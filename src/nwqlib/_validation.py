"""Scalar checks used at scientific input boundaries."""

from __future__ import annotations

import math
from numbers import Real
import operator
from typing import Any, Iterable

import numpy as np


# Dimensionless input convention, not a forward-error bound. It is the admitted
# excess beyond the endpoints of host-kernel probabilities, masses and QPE
# ancilla means, the normalization window of a marginal without a preparation
# receipt, the floor of exact_probability_window, and the window of subset-mass
# and QHD mass-partition comparisons. Revisit for changed precision
# or units.
NUMERICAL_RELATION_RTOL = 1e-12

# Binary64 unit roundoff u.
UNIT_ROUNDOFF = 2.0**-53

# The int64 count domain C = 2**63 - 1 of requested shots and stored counts.
# With R the requested shots and c_i the stored count values, a count chunk
# satisfies c_i integer, 0 <= c_i <= C, T = sum_i c_i computed exactly, and
# 0 <= T = returned_shots <= R <= C with R >= 1. The requested upper limit is
# enforced by the counts declaration (ObservationSpec._readout) and the full
# relation again by the chunk (ObservationChunk._statistics), before a reader
# creates an int64 array. For one validated chunk every intermediate sum of
# its nonnegative entries lies in [0, T], so an int64 sum of that chunk's
# weights cannot overflow; sums across acquisitions, or of scaled products,
# still need an overflow-safe accumulation. It is a representation boundary:
# the reader returns counts as int64 weights. execution.MAX_COUNT names the
# same value. Registered in docs/ENGINEERING_CONSTANTS.md ("Readout count
# representation"). Source: the NWQLib count-domain derivation.
MAX_COUNT = 2**63 - 1

# Exact-probability roundoff window. The bounds below are first order in u, and
# they measure the change of the squared state norm that roundoff can cause while
# a simulator executes a circuit and reads out the result. gamma_n =
# n*u/(1 - n*u), about n*u, is Higham's constant (Accuracy and Stability of
# Numerical Algorithms, 2nd ed., SIAM 2002, doi:10.1137/1.9780898718027,
# Lemma 3.1).
#
# Inner products. A simulator forms each new amplitude as an m-term complex inner
# product. Its real and imaginary parts are real sums of 2m products. In any
# summation order, with or without fused multiply-add, each product passes
# through at most 2m roundings, so each part has error at most gamma_(2m) times
# the sum of its |products|. Since (|ac| + |bd|)**2 + (|ad| + |bc|)**2 <=
# 2*|x|**2*|y|**2 for x = a + ib and y = c + id, the complex error is at most
# sqrt(2)*gamma_(2m)*sum_k |x_k||y_k|.
#
# Blocks. Let A be a b x b matrix that is unitary up to entry rounding, with at
# most s nonzeros per row and column. Each row and column of |A| has unit 2-norm,
# so || |A| ||_2 <= sqrt(s), and applying A to amplitudes x has 2-norm error at
# most sqrt(2)*gamma_(2b)*sqrt(s)*||x||_2. Multiplying A into a matrix with c unit
# columns adds that error to each column, and since sum_j |x_j| <=
# sqrt(c)*||x||_2 the product's action on a unit vector changes by at most
# sqrt(c) times the per-column error.
#
# Gate entries. Assume cos, sin and the exponential of an imaginary argument are
# accurate to one ulp. Aer 0.17.2 then computes every parameterized gate entry
# with relative error at most 5u (at most one cos or sin, one complex
# exponential and one product), and NWQ-Sim's U gate, which forms
# exp(i*phi)*exp(i*lambda) with one more complex product, with at most 9.83u.
# GATE_ENTRY_ERROR = 10 covers both. The entries of Aer's u, u2, u3, cu, cu2,
# cu3, mcu, mcu2 and mcu3 gates also round a sum of angles inside the
# exponential, which moves an entry's phase by at most 2*u*A, A being the sum of
# the gate's |parameters|. For A <=
# PHASE_SUM_LIMIT the entry error is at most (GATE_ENTRY_ERROR + 2*A)*u, and a
# gate with || |G| ||_2 <= 2 then moves ||G x||_2 by at most twice that.
#
# Aer (qiskit-aer 0.17.2 statevector method, src/transpile/fusion.hpp). Aer fuses
# only circuits wider than fusion_threshold, 14 qubits by default, so a circuit
# of at most 14 qubits applies every instruction as its own block. In a wider
# circuit four passes (diagonal, one-qubit, two-qubit and cost-based) fold
# instructions into fused operations on at most five qubits, d = 32, by applying
# them to the columns of a unitary-simulator state that starts as the identity.
# The first fold of each fused operation is exact because its products are by 0
# or 1. Every later fold merges two operations into one, so G native
# instructions give at most G - 1 inexact folds and at most G applied blocks. The
# worst fold is a dense five-qubit instruction, at most
# sqrt(2)*gamma_64*sqrt(32)*sqrt(32), and the worst application is a dense fused
# block, at most sqrt(2)*gamma_64*sqrt(32). Instructions on more than five qubits
# are never fused. A multi-controlled 2 x 2, diagonal or permutation instruction
# of any width costs at most sqrt(2)*gamma_4*sqrt(2) per application. A dense
# matrix on seven or more qubits exceeds the per-instruction charge, and only
# instructions carrying supplied matrices (``unitary``, ``kraus`` and
# ``quantum_channel``, all excluded) can be one.
#
# NWQ-Sim CPU/SV (efd0226, include/circuit_pass/fusion.hpp,
# include/svsim/sv_cpu.hpp and the U/CX matrices in
# include/private/gate_factory/sv_gate.hpp, with ValType = double and
# ENABLE_FUSION from include/config.hpp and include/nwq_util.hpp, and the gate
# records of include/circuit.hpp and include/private/sim_gate.hpp). A runner
# built from a revision that descends from efd0226 with these files unchanged is
# inside this derivation (_ROUNDOFF_SOURCES and _ROUNDOFF_BASE_REVISION in
# backends/nwqsim.py). The runner sends only U and
# CX gates. Fusion multiplies 2 x 2 and 4 x 4 gate matrices. Its Kronecker
# products with the identity and its control-target reversal are exact, and
# every product merges two gates, so G gates give at most G - 1 inexact products
# and at most G applied 4 x 4 blocks. A product of two 4 x 4 factors with
# || |.| ||_2 <= 2 changes the action on a unit vector by at most
# sqrt(2)*gamma_8*2*2, and
# applying a 4 x 4 block costs at most sqrt(2)*gamma_8*2. CX entries are exact,
# and a U gate has || |G| ||_2 <= sqrt(2).
#
# Each constant charges every native instruction its worst inexact fold, applied
# block and entry error, and doubles the sum because the total probability is
# the squared norm. The excluded instructions and the input unitarity of
# supplied matrices are listed in docs/ENGINEERING_CONSTANTS.md.
GATE_ENTRY_ERROR = 10.0
# Largest sum of |parameters| of an Aer phase-sum gate inside the derivation. It
# admits three angles each reduced to [-pi, pi], with pi to spare. A gate with a
# larger sum is listed as an exclusion instead of widening every window. Revisit
# if library constructions produce larger unreduced angles.
PHASE_SUM_LIMIT = 4 * math.pi


def _complex_inner_product(terms: int) -> float:
    """First-order error of an m-term complex inner product, in units of u*sum_k |x_k||y_k|."""
    return math.sqrt(2) * 2 * terms


_AER_FUSED_DIMENSION = 2**5
_AER_FOLD = _complex_inner_product(_AER_FUSED_DIMENSION) * math.sqrt(_AER_FUSED_DIMENSION) * math.sqrt(
    _AER_FUSED_DIMENSION)
_AER_APPLICATION = _complex_inner_product(_AER_FUSED_DIMENSION) * math.sqrt(_AER_FUSED_DIMENSION)
_AER_ENTRY = 2 * (GATE_ENTRY_ERROR + 2 * PHASE_SUM_LIMIT)
AER_STATEVECTOR_ROUNDOFF_PER_OPERATION = 2 * (_AER_FOLD + _AER_APPLICATION + _AER_ENTRY)
_NWQSIM_FOLD = _complex_inner_product(4) * 2 * 2
_NWQSIM_APPLICATION = _complex_inner_product(4) * 2
_NWQSIM_ENTRY = math.sqrt(2) * GATE_ENTRY_ERROR
NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION = 2 * (_NWQSIM_FOLD + _NWQSIM_APPLICATION + _NWQSIM_ENTRY)


# One multiplication of a unit state by a global phase exp(i*phi) moves it by at
# most 2u + sqrt(2)*gamma_2 = 4.83u in 2-norm, 5 in units of u after rounding up.
# The phase entry is accurate to 2u (one-ulp cos and sin), and each complex
# product is a two-term complex inner product (_complex_inner_product). Revisit
# with the complex exponential or complex-product model.
GLOBAL_PHASE_STATE_ROUNDOFF = 5

# Relative roundoff, in units of u, of one probability evaluated from one
# amplitude z. re*re + im*im makes two products and one nonnegative addition,
# gamma_2 = 2u to first order. Aer's full-register probabilities and the NWQ-Sim
# runner's probability bins evaluate a one-amplitude bin this way. np.abs(z)**2
# is (1 + 2u)**2 (1 + u) - 1 = 5u to first order, assuming NumPy's complex
# absolute value is accurate to 2u for normal nonzero magnitudes (one ulp of its
# scaled norm). Gradual underflow adds at most 2**-1074 per evaluation, and
# flush-to-zero is outside both. Revisit when a readout sums several amplitudes
# into one bin or NumPy changes its complex absolute value.
COMPONENT_SQUARES_ROUNDOFF = 2
ABSOLUTE_SQUARE_ROUNDOFF = 5


def exact_readout_roundoff(qubits: int) -> float:
    """First-order readout change of an exact probability total, in units of u, for n qubits.

    A probability total or Pauli expectation sums at most N = 2**n terms, each
    at most two amplitude products with exact sign changes, so it is a real sum
    of at most 2N products. In any summation tree, including per-bin sums
    followed by a sum of the bins, each product passes through at most 2N
    roundings, and a compensated sum does better, so the error is at most
    gamma_(2N) times the squared norm. The constant
    ``2*GLOBAL_PHASE_STATE_ROUNDOFF = 10`` covers one multiplication of the
    state by a global phase exp(i*phi), which changes the squared norm by at
    most 2*(2u + sqrt(2)*gamma_2) = 9.66u. Branch masses that NWQLib derives
    from amplitudes are outside this term (docs/ENGINEERING_CONSTANTS.md).
    """
    return 2.0 ** (operator.index(qubits) + 1) + 2 * GLOBAL_PHASE_STATE_ROUNDOFF


def exact_probability_window(operations: int | None, qubits: int, *,
                             per_operation: float = AER_STATEVECTOR_ROUNDOFF_PER_OPERATION) -> float:
    """Admitted |total - 1| of one exact binary64 statevector probability population.

    The window is ``(per_operation*G + exact_readout_roundoff(n))*u`` for G
    native instructions on n qubits, never below ``NUMERICAL_RELATION_RTOL``.
    ``per_operation`` is the derived constant of the executing simulator. The
    default is Aer's, the larger one, which a receipt from a target without a
    derivation of its own also uses. An unknown operation count keeps the floor.
    The instructions the derivation excludes are listed in the receipt's
    ``probability_window_exclusions``.
    """
    if operations is None:
        return NUMERICAL_RELATION_RTOL
    operations, qubits = operator.index(operations), operator.index(qubits)
    if operations < 0 or qubits < 0:
        raise ValueError("operation and qubit counts must be nonnegative")
    roundoff = (per_operation * operations + exact_readout_roundoff(qubits)) * UNIT_ROUNDOFF
    return max(NUMERICAL_RELATION_RTOL, roundoff)


def native_state_error(operations: int, per_operation: float) -> float:
    """First-order 2-norm error budget delta of the final state of one exact native execution.

    ``per_operation`` is a simulator's constant above, which doubles each
    instruction's 2-norm charge for the squared norm, so one instruction moves
    the state by at most ``per_operation/2`` units of u. One global-phase
    multiplication adds ``GLOBAL_PHASE_STATE_ROUNDOFF``, so
    ``delta = (per_operation*G/2 + 5)*u`` for G native operations (the
    receipt's ``native_operations``, which on Aer counts no save instruction), relative
    to the unitary native circuit with its binary64 gate parameters read
    exactly. It holds under the assumptions of those constants, for an
    execution whose assessed ``probability_window_exclusions`` are empty or
    bounded by the caller's own readout (``PreparedArtifact.state_error``).
    The readout term and the ``NUMERICAL_RELATION_RTOL`` floor of
    ``exact_probability_window`` bound a population total and are not part
    of it.
    """
    operations = operator.index(operations)
    if operations < 0:
        raise ValueError("operation count must be nonnegative")
    # Half of each doubled per-instruction charge, plus one global-phase product.
    return (per_operation * operations / 2 + GLOBAL_PHASE_STATE_ROUNDOFF) * UNIT_ROUNDOFF


def probability_difference_window(delta: float, evaluation: float) -> float:
    """Bound ``2*delta + delta**2 + evaluation*(1 + delta)**2`` on a computed probability difference.

    For a unit state psi and a computed state ``psi_hat`` with
    ``||psi_hat - exp(i phi) psi||_2 <= delta`` for some phase phi, the
    diagonal operator ``|i><i| - |j><j|`` has norm one, so the exact
    probabilities ``q`` of ``psi_hat`` satisfy
    ``|(q_i - q_j) - (p_i - p_j)| <= 2*delta*sqrt(p_i + p_j) + delta**2``,
    at most ``2*delta + delta**2``. If each probability is evaluated with
    relative error at most e, the two evaluations add at most
    ``e*(q_i + q_j) <= e*(1 + delta)**2``, and ``evaluation`` is that e. The
    same expression bounds the change of a whole population total when
    ``evaluation`` is the relative error of the summed total.

    An overflowing nonnegative state term returns infinity before the
    evaluation term is formed. A zero evaluation error contributes exactly
    zero, including when the state term is infinite. Products avoid the
    OverflowError that exponentiation can raise. A caller requiring a
    finite window checks the returned bound.
    """
    # Nonnegative contributions cannot cancel an overflowing state term.
    state = 2.0 * delta + delta * delta
    if state == float("inf") or evaluation == 0.0:
        return state
    return state + evaluation * ((1.0 + delta) * (1.0 + delta))


# SciPy's expm_multiply (scipy.sparse.linalg._expm_multiply, SciPy 1.18.1) is
# Algorithm 3.2 of Al-Mohy and Higham, "Computing the action of the matrix
# exponential, with an application to exponential integrators", SIAM J. Sci.
# Comput. 33 (2011) 488-511, doi:10.1137/100788860. With tol = 2**-53 it takes
# truncated Taylor sums
# of degree m at most m_max = 55, and it chooses the number of substeps from
# theta_m of the selected degree, at most theta_55 = 9.9, the largest entry of
# its Table 3.1 and of SciPy's _theta table. Both are paper and SciPy
# constants, not tuning choices. Revisit when SciPy changes m_max, the theta
# table or the algorithm.
EXPM_MULTIPLY_TERMS_MAX = 55
EXPM_MULTIPLY_THETA_MAX = 9.9


def expm_multiply_call_roundoff(norm: float, nonzeros: int) -> float:
    """First-order 2-norm error, in units of u, of one SciPy expm_multiply call on a unit vector.

    The call applies ``exp(-i G)`` for a generator ``G`` whose floating-point
    matrix is exactly Hermitian. ``norm`` is a bound N on
    ``||G - (trace(G)/D) I||_1`` and ``nonzeros`` a bound r on the nonzeros in
    a row of ``G``. The charge is

    ``c = N + sqrt(2)*2*r*N*e**theta_bar + S*(2*e**theta_bar + 7)``,
    ``theta_bar = min(N, EXPM_MULTIPLY_THETA_MAX)`` and
    ``S = EXPM_MULTIPLY_TERMS_MAX * max(1, ceil(N/EXPM_MULTIPLY_THETA_MAX))``,

    relative to the exact exponential of the shifted matrix that SciPy
    evaluates, up to a global phase. The rounding of the trace shift is not
    included (``expm_multiply_state_error`` charges it).

    Derivation, first order in u, for Algorithm 3.2 of Al-Mohy and Higham,
    doi:10.1137/100788860, with ``T_m`` the degree-m Taylor polynomial of
    ``exp`` (their Eq. (3.3)). SciPy shifts ``A = -iG`` by
    ``mu = trace(A)/D``, which is imaginary, so ``A' = A - mu I`` stays
    skew-Hermitian. It then takes s substeps of degree ``m <= 55``. The cost
    minimization (3.11) never takes more matrix products than the choice
    ``m = 55`` with ``ceil(alpha_p/theta_55)`` substeps, and its norm
    estimates never exceed ``||A'||_1 <= N``, so ``m*s <= S``. The analysis
    rests on two assumptions that SciPy's error control itself makes.

    1. Its backward-error control (3.9) chooses s with
       ``alpha_p(A')/s <= theta_m <= 9.9``, and the norm estimates in
       ``alpha_p`` are not below the spectral radius. When condition (3.13)
       holds, SciPy uses the exact ``||A'||_1``, which is at least
       ``rho(A') = ||A'||_2``. Each substep then has
       ``theta = ||A'||_2/s <= theta_bar``.
    2. The two-term test (3.15), which the paper introduces as an estimate
       of the omitted tail, bounds that tail in 2-norm, so an early exit
       changes a substep's result by at most u relative to its norm. The
       test itself compares infinity norms, so this is an assumption.

    - Truncation. ``T_m(A'/s)**s = exp(A' + dA)`` with ``||dA|| <= u*N``
      (Eqs. (3.5)-(3.9)), which moves the unit vector by at most ``u*N``.
    - Sparse products. In one substep of a unit vector the term
      ``B_j = (A'/(s j)) B_(j-1)`` has ``||B_(j-1)|| <= theta**(j-1)/(j-1)!``,
      and its sparse product errs by at most
      ``sqrt(2)*gamma_(2r)*(N/(s j))*||B_(j-1)||`` (see
      ``_complex_inner_product``). The rest of the finite sum carries that
      error through ``sum_(k=0)^(m-j) (A'/s)**k j!/(j+k)!``, whose norm is at
      most ``sum_k theta**k j!/(j+k)!``. Over j these factors total
      ``sum_(v=1)^m v theta**(v-1)/v! <= e**theta``, so the sparse products
      of all s substeps contribute at most ``sqrt(2)*2*r*N*e**theta_bar``.
    - Other roundings of one substep. Rounding each scale coefficient and its
      vector multiplication adds at most ``2*theta*e**theta``, the additions
      into the partial sums at most ``m + e**theta`` (the partial sums are
      bounded by the Taylor remainder on the imaginary axis), and the product
      with the unit phase ``exp(mu/s)`` and the early exit at most 6. The
      substep charge ``(2*theta + 1)*e**theta + m + 6`` is at most
      ``m*(2*e**theta + 7)`` because every entry of SciPy's theta table has
      ``theta_m <= m/4``, and ``m*s <= S`` gives the S term. A degree-zero
      call only multiplies by the phase, which S also covers.
    - Later substeps apply ``T_m(A'/s)``, whose norm is ``1 + O(u)``, so an
      earlier error is not amplified to first order.

    The factor ``e**theta_bar`` is the growth of the Taylor terms before they
    decay, the per-substep ``e**||A||`` of Al-Mohy and Higham,
    doi:10.1137/100788860, Lemma 4.1.
    Their Eq. (4.7) shows that for a normal matrix with a unitary exponential
    this factor exceeds what the problem's conditioning requires. Under the
    two assumptions the bound holds for every unit input state, so it can
    exceed the roundoff of a smooth state by several orders of magnitude.

    The charge is returned as inf when S or the sum exceeds the binary64
    range, and callers refuse the nonfinite budget.
    """
    norm = float(norm)
    if not math.isfinite(norm) or norm < 0.0:
        raise ValueError("expm_multiply generator norms must be finite and nonnegative")
    theta = min(norm, EXPM_MULTIPLY_THETA_MAX)
    products = EXPM_MULTIPLY_TERMS_MAX * max(1, math.ceil(norm / EXPM_MULTIPLY_THETA_MAX))
    growth = math.exp(theta)
    sparse = _complex_inner_product(operator.index(nonzeros))
    try:
        rounding = products * (2.0 * growth + 7.0)
    except OverflowError:
        # The integer S does not convert to a binary64 float.
        return math.inf
    # Truncation N, sparse products sqrt(2)*2*r*N*e**theta, other roundings S*(2*e**theta + 7).
    return norm + sparse * norm * growth + rounding


def expm_multiply_state_error(calls: Iterable[tuple[float, int]], *, start: float,
                              phase_multiplications: int = 0) -> float:
    """First-order 2-norm error budget delta of a unit start state evolved by expm_multiply calls.

    ``calls`` holds one pair ``(N, r)`` per call, as in
    ``expm_multiply_call_roundoff``. The computed start vector differs from
    the exact unit start state by at most ``start*u`` in 2-norm. For QHD,
    ``algorithms.qhd.initial_state.restricted_state_error`` derives it for
    each initial state: 2 for the uniform state, whose D equal entries
    ``1/sqrt(D)`` take one correctly rounded square root and one division of
    an exactly representable D, and the per-variable amplitude and product
    roundings of the other states. Each call adds its charge c and a further
    ``N*u`` for the rounded subtraction of the trace shift from the diagonal,
    whose exact value is a scalar phase choice. Each final multiplication of
    the state by a unit phase adds ``GLOBAL_PHASE_STATE_ROUNDOFF``:

    ``delta = u*(start + sum_calls (c + N) + 5*phase_multiplications)``.

    Exact unitary propagation carries each local error to the final state
    unchanged in norm, so the charges add. The budget is relative to the
    ordered product of the exact exponentials of the floating-point
    generators passed to SciPy, applied to the exact start state, up to a
    global phase. It excludes the error
    of those generators relative to exact coefficients and every time,
    splitting or grid discretization error. The centering charge is first
    order and assumes that the N of each call also bounds the shifted matrix
    that SciPy forms, so a huge scalar offset whose second-order terms, such
    as ``u**2*D*||G||``, dominate lies outside it. The one budget serves both
    consumers: ``state_mass_window`` turns it into the mass window and
    ``algorithms.qhd.method._host_window`` into the most-probable tie window
    of the QHD ``schrodinger`` kernel, which is also its Plan's ceiling
    ``algorithms.qhd.method._host_tie_window``.

    The budget is returned as inf when the charges sum beyond the binary64
    range, and callers refuse the nonfinite budget.
    """
    # Start vector, each call c + N (centering), each final phase product 5, in units of u.
    try:
        total = math.fsum(expm_multiply_call_roundoff(norm, nonzeros) + float(norm) for norm, nonzeros in calls)
    except OverflowError:
        # fsum raises when its finite partial sums exceed the binary64 range. Calls after that
        # point are not validated, and the inf budget is refused in any case.
        total = math.inf
    return (float(start) + total
            + GLOBAL_PHASE_STATE_ROUNDOFF * operator.index(phase_multiplications)) * UNIT_ROUNDOFF


def state_mass_window(delta: float, dimension: int) -> float:
    """Admitted |total - 1| of the D probabilities ``np.abs(z)**2`` of a state computed within delta of a unit state.

    The window is
    ``max(NUMERICAL_RELATION_RTOL, probability_difference_window(delta, (D + 4)*u))``
    with D = ``dimension``, the state length. A computed state within delta
    of a unit state in 2-norm has a squared norm within ``2*delta + delta**2``
    of one. Evaluating the D probabilities with ``np.abs(z)**2``
    (``ABSOLUTE_SQUARE_ROUNDOFF``, 5u) and summing them with at most D - 1
    additions adds at most ``(D + 4)*u*(1 + delta)**2`` to first order.
    Every QHD host kernel takes delta for its mass window from the Plan's
    state budget of its flavor (``algorithms.qhd.method._host_state_error``,
    which is ``expm_multiply_state_error``,
    ``algorithms.qhd.theory.binary_product_state_error`` or
    ``algorithms.qhd.split_step.state_error``), and the same delta gives the
    ceiling of its tie window (``algorithms.qhd.method._host_tie_window``).
    The ``split_step`` kernel selects its most probable point with the
    smaller budget that ``algorithms.qhd.split_step.evolve`` observes on its
    trajectory, and the other kernels with the Plan's budget.

    Raises:
        ValueError: delta is not finite and nonnegative.
    """
    if not (math.isfinite(delta) and delta >= 0):
        raise ValueError("a state error budget must be finite and nonnegative")
    # D - 1 additions of the D probabilities plus the 5u of each abs(z)**2 evaluation.
    evaluation = (operator.index(dimension) - 1 + ABSOLUTE_SQUARE_ROUNDOFF) * UNIT_ROUNDOFF
    return max(NUMERICAL_RELATION_RTOL, probability_difference_window(delta, evaluation))


def validate_normalized_mass(value: float | None, window: float | None = None) -> None:
    """Admit one independently known observed probability, keeping raw roundoff.

    Alone, the mass is only known to be nonnegative. How far roundoff may carry
    it above one depends on the execution that produced it, so a caller that
    holds that execution's window passes it. A circuit's window comes from its
    preparation receipt: direct probability and Pauli readouts use
    ``PreparedArtifact.probability_window``, and masses formed from a corrected
    saved state use ``PreparedArtifact.saved_state_probability_window``. A host
    kernel's window is ``NUMERICAL_RELATION_RTOL`` unless the kernel derives
    its own, as the QHD kernel does with ``state_mass_window``. This domain
    does not apply to unnormalized norms or reference predictions. None
    denotes unavailable evidence and does not constrain another mass.
    """
    if value is not None and not (0 <= value and (window is None or value <= 1 + window)):
        raise ValueError("normalized observed branch mass must be nonnegative and at most one plus "
                         "the roundoff window of its execution")


def boolean(value: Any, name: str) -> bool:
    """Return a native Boolean without interpreting numbers or strings as flags."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a native bool")
    return value


def integer(value: Any, name: str, minimum: int) -> int:
    """Return an exact integer at or above the caller's minimum."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    try:
        resolved = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer at least {minimum}") from None
    if resolved < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return resolved


def finite_real(value: Any, name: str) -> float:
    """Return a finite real scalar without accepting booleans or strings."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        resolved = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite real number") from None
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite real number")
    return resolved
