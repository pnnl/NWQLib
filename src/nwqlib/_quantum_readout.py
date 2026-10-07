"""Reduce actual selected quantum populations, without acquisition or replay.

An, Childs & Lin, arXiv:2312.03916v2, Appendix A.3 Lemma 24 Eq. (178), p. 36,
https://arxiv.org/html/2312.03916v2#A3, defines the encoded LCU amplitudes.
The reductions below derive physical moments by projection and positive scale
recovery from that convention. Statistical intervals require separate evidence.

An outcome index of ObservationChunk.histogram has observed position p at
bit p, so the bit at observed position p is (index >> p) & 1.

The exact scalar route registers the acquisition-time reduction
``projected_moments`` when this module is imported, and
``core.planning.BUILTIN_REDUCER_MODULES`` lists this module so that the name
resolves in a process that has not imported it. One saved state yields the
complete native norm, the success-only mass, the physical-slice mass and the
projected Pauli moment as scaled (mantissa, exponent) pairs, validated
against the producing receipt's qualified state error or probability-window
convention before publication (``reduce_scaled``, ``validate_saved_masses``).
The sampled route groups qubit-wise commuting labels (``qwc_groups``) and
reduces each group's counts to weighted moments (``weighted_group_moments``).
A sampled physical mass reduces one counts setting (``reduce_setting``).
Sampled original-coordinate outcomes reduce to sorted index and count arrays
(``reduce_sample_arrays``).
"""

import math
import sys
from fractions import Fraction as Q
from math import frexp, fsum, isfinite, ldexp

import numpy as np

from pydantic import model_validator

from nwqlib.core.records import Record, Text


class ReadoutSetting(Record):
    """One Pauli parity or original-coordinate mass measurement.

    physical_projection is measured before any basis rotation. A Pauli term for
    a non-power-of-two observable represents its zero-padded P O P operator.
    Original-coordinate validity cannot be inferred after rotating that basis.
    """

    name: Text
    label: Text
    physical_projection: bool = False

    @model_validator(mode="after")
    def _relation(self):
        if set(self.label) - set("IXYZ"):
            raise ValueError("readout Pauli labels use IXYZ")
        if self.physical_projection and any(axis != "I" for axis in self.label):
            raise ValueError("original-coordinate mass is measured before Pauli basis rotation")
        return self


def validate_readout_layout(chunk, *, observed, classical):
    """Match physical observation order and actual classical storage positions.

    A trajectory point chunk is checked against its point's one-point readout
    (``ObservationChunk.observation``).
    """
    readout = chunk.observation
    if readout.kind == "probabilities":
        if readout.qubits != observed:
            raise ValueError("probability observation differs from selected physical bit order")
    elif readout.kind == "counts":
        position, expected = 0, []
        for declaration in classical:
            if declaration.dtype != "bits":
                continue
            if type(declaration.width) is not int or not 1 <= declaration.width <= len(observed) - position:
                raise ValueError("readout requires resolved classical bit declarations")
            expected.append((declaration.name, tuple(range(position, position + declaration.width))))
            position += declaration.width
        if position != len(observed) or tuple((r.name, r.bits) for r in chunk.classical_layout) != tuple(expected):
            raise ValueError("count layout differs from selected classical declarations")


def _positions(observed, success, conditions, coordinates):
    """Return the observed position p of each observed physical bit.

    The bit is bit p of an outcome index. Every selector bit in success and conditions and every coordinate bit
    must be distinct and observed, and each selector value must be 0 or 1.
    """
    selected = tuple(bit for bit, _ in success + conditions) + coordinates
    if (len(set(observed)) != len(observed) or len(set(selected)) != len(selected)
            or any(type(bit) is not int or bit < 0 for bit in observed)
            or any(value not in (0, 1) for _, value in success + conditions)):
        raise ValueError("readout requires distinct coordinates and valued selector bits")
    positions = {bit: index for index, bit in enumerate(observed)}
    if any(bit not in positions for bit in selected):
        raise ValueError("selected physical bit is absent from the observed order")
    return positions


def _outcomes(chunk, observed):
    """Return the outcome indices and weights of the chunk, as Python numbers, after checking its width."""
    histogram = chunk.histogram()
    if histogram.entries and histogram.width != len(observed):
        raise ValueError("histogram width differs from selected observed bits")
    return zip(histogram.index_list(), histogram.weights.tolist(), strict=True)


def _coordinate(outcome, coordinates, positions):
    """Return the integer basis index whose bit i is the observed value of coordinates[i]."""
    return sum(((outcome >> positions[bit]) & 1) << index for index, bit in enumerate(coordinates))


def reduce_setting(chunk, *, observed, success, conditions=(), pivot=None,
                   coordinates=(), dimension=None):
    """Return algorithm mass, physical mass and signed Pauli moment.

    All three are fractions of the returned shots (counts) or of total
    probability. The algorithm mass keeps outcomes whose success selectors
    match. The physical mass further requires the condition selectors and,
    with dimension, a coordinate below dimension. The signed moment is that
    physical mass with outcomes weighted -1 where the pivot bit reads one.
    With an original-coordinate projection P, mass is sum_{j<dimension}|a_j|².
    Such a coordinate check is invalid after a Pauli basis rotation, so pivot
    and dimension cannot be combined. Zero-padded P O P terms are instead
    measured without this check, then normalized by a separately observed mass.
    """
    if dimension is not None and pivot is not None:
        raise ValueError("original-coordinate projection cannot follow a Pauli basis rotation")
    if dimension is not None and (type(dimension) is not int or dimension < 1
                                  or (dimension - 1).bit_length() > len(coordinates)):
        raise ValueError("physical dimension must fit the observed coordinate register")
    needed = coordinates if dimension is not None else (() if pivot is None else (pivot,))
    positions = _positions(observed, success, conditions, needed)
    if chunk.observation.kind == "pauli_expectation":
        if observed or success or conditions or pivot is not None or dimension is not None:
            raise ValueError("identity expectation cannot encode selected bit conditions")
        return (1.0, 1.0, 1.0) if chunk.values else (None, None, None)
    counts = chunk.observation.kind == "counts"
    if not counts and chunk.observation.kind != "probabilities":
        raise ValueError("setting reduction requires counts or probabilities")
    denominator = chunk.returned_shots if counts else 1.0
    if denominator == 0:
        return None, None, None
    algorithm, plus, minus = [], [], []
    narrower = bool(conditions) or dimension is not None
    for outcome, weight in _outcomes(chunk, observed):
        if any((outcome >> positions[bit]) & 1 != value for bit, value in success):
            continue
        if narrower:
            algorithm.append(weight)
        if any((outcome >> positions[bit]) & 1 != value for bit, value in conditions):
            continue
        if dimension is not None and _coordinate(outcome, coordinates, positions) >= dimension:
            continue
        (minus if pivot is not None and (outcome >> positions[pivot]) & 1 else plus).append(weight)
    positive, negative = fsum(plus), fsum(minus)
    mass = (positive + negative) / denominator
    return fsum(algorithm) / denominator if narrower else mass, mass, (positive - negative) / denominator


def physical_moment(value, recovery, exponent=0):
    """Recover gamma² times an encoded moment without first squaring gamma.

    From ACL arXiv:2312.03916v2, Eq. (178), a physical amplitude is gamma
    times its selected encoded amplitude. Its quadratic moment thus scales by
    gamma². frexp/ldexp express the same product using bounded binary64
    mantissas and integer exponents. An unrepresentable nonzero result
    remains unavailable, not clipped to zero.
    """
    if recovery is None:
        return None
    if value == 0:
        return 0.0
    mantissa, value_exponent = frexp(value)
    try:
        result = ldexp(mantissa * recovery.mantissa * recovery.mantissa,
                       value_exponent + exponent + 2 * recovery.exponent)
    except OverflowError:
        return None
    return result if isfinite(result) and result != 0 else None


# Scaled selected reduction. IEEE binary64 round to nearest with gradual
# underflow. u = 2**-53, eta = 2**-1074 and exact rational evaluation of a
# bounded number of scalar allowances.
UNIT = Q(1, 2**53)
ETA = Q(1, 2**1074)


def pow2(e):
    """Exact rational value of 2**e."""
    return Q(2)**int(e)


def gamma(k):
    """Rounding factor gamma(k) = k*u/(1 - k*u) with u = 2**-53, defined for k*u < 1."""
    if type(k) is not int or k < 0 or k * UNIT >= 1:
        raise ValueError("invalid rounding-bound domain")
    return k * UNIT / (1 - k * UNIT)


def pair_value(pair):
    """Exact value of one saved scaled scalar ``(mantissa, exponent)``.

    A saved scalar is a canonical pair denoting its exact binary value. Zero
    is ``(0, 0)``. A nonzero mantissa has absolute value in [1/2, 1). Masses
    and numerators carry separate exponents. The exponent check
    [-8192, 8192] is a derived envelope for finite binary64 inputs and the
    admitted lengths ``F, N, L < 2**51`` defined in ``readout_requirements``:
    F is the saved-state length, N the coordinate-space dimension and L the
    number of kept packed Pauli rows. The exponent check is not a precision cutoff.
    Recovered physical exponents lie outside it and are not checked here.
    """
    m, e = pair
    if (type(e) is not int or not -8192 <= e <= 8192
            or not math.isfinite(m)
            or (m == 0 and e != 0)
            or (m != 0 and not 0.5 <= abs(m) < 1.0)):
        raise ValueError("invalid binary-scale scalar")
    return Q(float(m)) * pow2(e)


def fraction_pair(value, *, upward=False):
    """Round an exact rational to its canonical scaled pair.

    With ``upward`` the mantissa is rounded toward +inf, so a published
    radius never understates its exact value.
    """
    value = Q(value)
    if upward and value < 0:
        raise ValueError("an outward radius must be nonnegative")
    if not value:
        return (0.0, 0)
    a = abs(value)
    e = a.numerator.bit_length() - a.denominator.bit_length() + 1
    if a < pow2(e - 1):
        e -= 1
    scaled = value / pow2(e)
    m = float(scaled)
    if upward and Q(m) < scaled:
        m = math.nextafter(m, math.inf)
    if abs(m) == 1:
        m *= 0.5
        e += 1
    return (m, e)


def sqrt_up(value, bits=128):
    """Upper bound on sqrt(value) with ``bits`` fractional bits, in exact integer arithmetic."""
    value = Q(value)
    if value < 0:
        raise ValueError("negative squared norm")
    scale = 1 << bits
    n, d = value.numerator * scale * scale, value.denominator
    root = math.isqrt(n // d)
    if root * root * d < n:
        root += 1
    return Q(root, scale)


def selected_tiles(native, fixed=(), *, tile=1024):
    """Yield bounded tiles of the native amplitudes whose valued selector bits match.

    Callers validate width, dtype and distinct selectors before iterating.
    Each tile forms at most ``tile`` native indices and one selected tile.
    No F-entry index or probability array is formed. With no selectors each
    tile is a borrowed view.
    """
    mask = np.uint64(sum(1 << b for b, _ in fixed))
    value = np.uint64(sum(v << b for b, v in fixed))
    for start in range(0, len(native), tile):
        part = native[start:start + tile]
        if fixed:
            ix = np.arange(start, start + len(part), dtype=np.uint64)
            part = part[(ix & mask) == value]
        yield part


def scaled_mass_tiles(make_tiles, count):
    """Return the scaled squared norm of one population and its scaling exponent.

    For M complex amplitudes with m = 2M real components x_i, the largest
    magnitude a sets e = frexp(a).exponent. The components are scaled
    exactly by 2**-e, squared and summed by tiled NumPy reductions, and
    ``frexp`` of the total gains 2e in its exponent. Under IEEE binary64
    round to nearest with gradual underflow, finite complex128 components
    and the same population on both passes, the computed mass obeys
    ``|M_hat - M| <= a_M M + U_M`` with ``a_M = gamma(m+1)`` and
    ``U_M = 2**(2e) 4 m eta / (1 - (m+1) u)``, eta = 2**-1074. The largest
    scaled square is at least 1/4, so a positive mass stays positive even
    below the ordinary binary64 range, and a zero output means an exactly
    zero population. The scale is fixed on the first pass, so the
    population is selected twice.
    """
    if type(count) is not int or count < 0:
        raise ValueError("invalid population count")
    gamma(2 * count + 1)
    largest, seen = 0.0, 0
    for part in make_tiles():
        if part.dtype != np.complex128 or part.ndim != 1 or not np.isfinite(part).all():
            raise ValueError("finite complex128 population required")
        seen += len(part)
        largest = max(largest,
                      float(np.max(np.abs(part.real), initial=0.0)),
                      float(np.max(np.abs(part.imag), initial=0.0)))
    if seen != count:
        raise ValueError("population differs from its declaration")
    if largest == 0:
        return (0.0, 0), 0
    e, total = math.frexp(largest)[1], 0.0
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        for part in make_tiles():
            for component in (part.real, part.imag):
                work = np.ldexp(component, -e)
                np.square(work, out=work)
                total += float(np.sum(work, dtype=np.float64))
                del work
    m, exponent = math.frexp(total)
    return (m, exponent + 2 * e), e


def saved_mass_allowance(pair, count):
    """Relative and absolute host allowance of a saved mass, from the saved pair alone.

    Since ``2**(2e) <= 4 M_hat`` for this kernel, the saved absolute
    allowance ``U_M = 16 m eta M_hat / (1 - (m+1) u)`` with m = 2*count
    bounds the scan's underflow without the acquisition's scaling
    exponent. The relative allowance is ``gamma(m+1)``. A zero output of
    this kernel has zero host error. Arithmetic is exact rational on a
    bounded number of scalars.
    """
    value = pair_value(pair)
    if type(count) is not int or count < 0 or value < 0:
        raise ValueError("invalid mass population")
    if count == 0 and value != 0:
        raise ValueError("nonzero empty-population mass")
    a = gamma(2 * count + 1)
    if a >= 1:
        raise ValueError("mass inversion needs a < 1")
    if value == 0:
        return Q(0), Q(0)
    # T_hat >= 1/4 implies 2**(2e) <= 4*M_hat.
    absolute = 16 * (2 * count) * ETA * value / (1 - (2 * count + 1) * UNIT)
    return a, absolute


def validate_saved_masses(complete, success, physical, counts, *,
                          delta=None, window=None):
    """Check the complete native norm and the nested masses of one computed state.

    Success and physical masses are subpopulations and need not be one.
    With a justified native-state bound delta, the complete norm obeys
    ``|F_hat - 1| <= 2 delta + delta**2 + a_F (1+delta)**2 + U_F``. Without
    it, the producing receipt's probability window omega supplies an
    input-validation convention, ``T_F = omega + a_F (1 + omega) + U_F`` with
    ``|F_hat - 1| <= T_F``, and the accepted norm bounds the computed state's
    squared norm by ``R**2 <= (F_hat + U_F)/(1 - a_F)``. The subset relations
    on the same computed state need only host errors:
    ``A_hat - F_hat <= (a_A + a_F) R**2 + U_A + U_F`` and
    ``P_hat - A_hat <= (a_P + a_A) R**2 + U_P + U_A``. Each projected mass is
    also bounded by ``(1 + a) R**2 + U``. The window convention is a
    validation policy, not a certificate of closeness to the ideal circuit.
    Returns the norm upper bound R**2 and the complete-norm window.

    Projected acquisition and QLS/LCHS reanalysis pass the same pair of the
    producing receipt's saved-state budgets: delta is
    ``PreparedArtifact.saved_state_error``, which adds the host phase
    product's charge to the native estimate, and omega is
    ``saved_state_probability_window``, omega_saved. When native exclusions
    make delta unavailable, ``T_F = omega_saved + a_F(1 + omega_saved) + U_F``
    and ``R**2 <= (F_hat + U_F)/(1 - a_F)``. Subset checks continue adding
    only their host mass-evaluation errors against this accepted norm. They
    do not add native error a second time.
    """
    F, A, P = map(pair_value, (complete, success, physical))
    nF, nA, nP = counts
    if not 0 <= nP <= nA <= nF:
        raise ValueError("populations are not nested")
    aF, uF = saved_mass_allowance(complete, nF)
    aA, uA = saved_mass_allowance(success, nA)
    aP, uP = saved_mass_allowance(physical, nP)
    if delta is not None:
        delta = Q(delta)
        if delta < 0:
            raise ValueError("negative native-state bound")
        norm_upper = (1 + delta)**2
        total_window = 2 * delta + delta**2 + aF * norm_upper + uF
    else:
        if window is None or Q(window) < 0:
            raise ValueError("the producing receipt's probability window is required")
        omega = Q(window)
        total_window = omega + aF * (1 + omega) + uF
        norm_upper = (F + uF) / (1 - aF)
    if abs(F - 1) > total_window:
        raise ValueError("complete native norm is outside the producing window")
    if (A - F > (aA + aF) * norm_upper + uA + uF
            or P - A > (aP + aA) * norm_upper + uP + uA
            or A > (1 + aA) * norm_upper + uA
            or P > (1 + aP) * norm_upper + uP):
        raise ValueError("saved masses violate their population relation")
    return norm_upper, total_window


def ordered_action_constants(n, terms):
    """Return exact-rational relative and absolute error allowances for one Pauli action.

    For an N-entry vector v and L terms with coefficient mass C1, the
    returned pair (e_A, U_A) bounds the action error by
    ``e_A*C1*||v||_2 + U_A``. N = n and L = terms. With L = 0 return
    exact zeros. For a nonempty action, N is a positive state dimension,
    L is a positive integer, and ``(2L + 4)u < 1`` is required.

    The model uses binary64 round to nearest, gradual underflow and
    finite successful arithmetic, with u = 2**-53 and eta = 2**-1074.
    Flush-to-zero arithmetic is outside this model. Define
    ``gamma(k) = k*u/(1-k*u)``, ``s(x) = sqrt_up(x)`` and
    ``mu = s(2)*gamma(2)``. Then

        e_A = mu + gamma(L-1) + mu*gamma(L-1),
        U_A = 8*L*eta*s(N)/(1-(2*L+4)*u).

    All scalar operations use exact rationals. ``sqrt_up`` rounds upward
    to a dyadic with 128 fractional bits. These constants are not rounded
    to binary64 here. The relative term multiplies C1*||v||, not ||Av||.

    The accumulation derivation is in ``operators._pauli.apply_terms``:
    one complex product contributes mu and the termwise sum contributes
    gamma(L-1), giving ``(1+mu)*(1+gamma(L-1))-1``. A partition into G
    groups of largest size h satisfies ``(h-1)+(G-1) <= L-1``. Its
    relative factor ``(1+gamma(h-1))*(1+mu)*(1+gamma(G-1))-1`` is therefore
    bounded by e_A. Its underflow allowance is
    ``4*(L+G)*eta*s(N)/(1-(L+G+4)*u)`` and is bounded by U_A because
    G <= L. The same ordered envelope covers either successful action
    path.
    """
    if not terms:
        return Q(0), Q(0)
    mu, g = sqrt_up(2) * gamma(2), gamma(terms - 1)
    r = 2 * terms + 4
    if r * UNIT >= 1:
        raise ValueError("invalid action underflow domain")
    return mu + g + mu * g, 8 * terms * ETA * sqrt_up(n) / (1 - r * UNIT)


def scaled_projected_moment(selected, table, physical, state_exponent):
    """Scaled numerator ``q = v^dagger O v`` on the physical slice and its host radius.

    The slice is framed by the physical mass pass's exponent e,
    ``r = RN(2**-e v)`` zero padded to N entries. The coalesced identity
    coefficient c_I contributes ``c_I P_hat`` with the saved physical mass.
    The other coefficients are framed by ``k = frexp(max |c_j|).exponent``,
    ``beta_j = RN(2**-k c_j)``, keeping every row. With
    ``kappa = sqrt(2d) eta / 2`` (zero for e <= 0), ``Delta_C = L eta / 2``
    (zero for k <= 0), ``C = (C_hat + L eta)/(1 - gamma(L-1))``,
    ``p_+ = (P_hat + U_P)/(1 - a_P)``, ``E_P = a_P p_+ + U_P``,
    ``s >= sqrt(p_+ / 2**(2e))`` and ``r_+ = s + kappa``, the framing and
    contraction errors are
    ``E_frame = (C + Delta_C)(2 s kappa + kappa**2) + Delta_C r_+**2`` and
    ``E_contract = [e_A + g(1 + e_A)] C r_+**2 + (1 + g) r_+ U_A + U_dot``
    with ``g = sqrt(2) gamma(2N)`` and ``U_dot = 8 N eta / (1 - (2N+4) u)``.
    The published numerator ``Q_0 = c_I P_hat + 2**(2e+k) Re(r^dagger B r)``
    has host radius
    ``E_q = |c_I| E_P + 2**(2e+k)(E_frame + E_contract) + E_pair`` against the
    stored Pauli observable on the computed physical slice, E_pair being the
    exact change of the final pair rounding. It is published upward. Framed
    inputs have magnitude below one, so for N, L < 2**51 the action and dot
    intermediates cannot overflow binary64. A nonfinite result is a violated
    premise. A signed contraction can round to zero with a nonzero radius,
    which is not evidence that the exact numerator is zero. Lost framed
    state components and coefficients are counted as diagnostics. The radius
    does not depend on them. Against an original dense observable add its
    conversion discrepancy times the physical mass. Against an ideal state
    add the native-state term. No native floating-point certificate is
    asserted.
    """
    from nwqlib.operators._pauli import PauliTerms, apply_terms

    p = pair_value(physical)
    N, d, L = 1 << table.num_qubits, len(selected), len(table)
    if not 0 <= d <= N < 2**51 or L >= 2**51 or table.x.shape[1] != 1:
        raise ValueError("matching single-word observable required")
    if selected.dtype != np.complex128 or not np.isfinite(selected).all():
        raise ValueError("finite physical slice required")
    if p == 0 or L == 0:
        return {"numerator": (0.0, 0), "host_radius": (0.0, 0),
                "lost_state_components": 0, "lost_coefficients": 0}
    coefficients = table.coefficients
    if not np.isfinite(coefficients).all() or np.any(coefficients.imag != 0):
        raise ValueError("finite real Pauli coefficients required")
    identity = (table.x[:, 0] == 0) & (table.z[:, 0] == 0)
    if np.count_nonzero(identity) > 1:
        raise ValueError("use the admitted coalesced observable table")
    cI = float(coefficients.real[identity][0]) if np.any(identity) else 0.0
    aP, uP = saved_mass_allowance(physical, d)
    p_upper = (p + uP) / (1 - aP)
    p_error = aP * p_upper + uP
    coeff = coefficients.copy()
    coeff[identity] = 0
    maximum = float(np.max(np.abs(coeff.real), initial=0.0))
    scalar, error = Q(cI) * p, abs(Q(cI)) * p_error
    lost_state = lost_coeff = 0
    if maximum:
        k = math.frexp(maximum)[1]
        e = state_exponent
        vector = np.zeros(N, dtype=np.complex128)
        with np.errstate(under="ignore", over="raise", invalid="raise"):
            for component in ("real", "imag"):
                before = getattr(selected, component)
                after = getattr(vector[:d], component)
                np.ldexp(before, -e, out=after)
                lost_state += int(np.count_nonzero((after == 0) & (before != 0)))
            before = coeff.real.copy()
            np.ldexp(coeff.real, -k, out=coeff.real)
            lost_coeff = int(np.count_nonzero((coeff.real == 0) & (before != 0)))
            del before
        beta = Q(float(np.sum(np.abs(coeff.real), dtype=np.float64)))
        C = (beta + L * ETA) / (1 - gamma(max(0, L - 1)))
        scaled = PauliTerms._snapshot(table.num_qubits, table.x, table.z, coeff)
        del coeff, identity
        with np.errstate(under="ignore", over="raise", invalid="raise"):
            action = apply_terms(scaled, vector, num_qubits=table.num_qubits)
            dot = float(np.vdot(vector, action).real)
        if not math.isfinite(dot):
            raise ValueError("scaled action violated its finite-arithmetic premise")
        s = sqrt_up(p_upper / pow2(2 * e))
        kappa = sqrt_up(2 * d) * ETA / 2 if e > 0 else Q(0)
        dc = L * ETA / 2 if k > 0 else Q(0)
        r = s + kappa
        action_error, action_underflow = ordered_action_constants(N, L)
        dot_error = sqrt_up(2) * gamma(2 * N)
        if (2 * N + 4) * UNIT >= 1:
            raise ValueError("invalid dot underflow domain")
        dot_underflow = 8 * N * ETA / (1 - (2 * N + 4) * UNIT)
        framing = (C + dc) * (2 * s * kappa + kappa**2) + dc * r**2
        contraction = ((action_error + dot_error * (1 + action_error)) * C * r**2
                       + (1 + dot_error) * r * action_underflow + dot_underflow)
        scale = pow2(2 * e + k)
        scalar += Q(dot) * scale
        error += scale * (framing + contraction)
    result = fraction_pair(scalar)
    error += abs(pair_value(result) - scalar)
    return {"numerator": result, "host_radius": fraction_pair(error, upward=True),
            "lost_state_components": lost_state, "lost_coefficients": lost_coeff}


def reduce_scaled(native, *, coordinates, success, conditions, dimension, table,
                  delta=None, window=None, tile=1024, has_moment=True):
    """Reduce one saved native state to its scaled masses and projected moment.

    The reducer scans success mass and the complete native norm in bounded
    tiles, then evaluates the observable on the success-and-physical slice.
    Admission includes the saved state, slice-index workspace, observable
    representations, coefficient framing and grouping metadata. Work counts
    the selected mass and gather branches and the grouped action on scaled
    inputs. The byte reserve includes the shared action's scratch envelope.
    Scaled scalar statistics carry the recovery information needed for reanalysis.

    Exact scalar readout evaluates the stored Pauli observable on the
    success-and-physical slice v, returning ``p = v^dagger v`` and
    ``q = v^dagger O_tilde v``. Normalized output is q/p for positive p. A
    dense observable is zero extended before coefficient conversion. Its
    conversion error contributes separately to error against the original
    dense observable. Projected and full-block Pauli sums coincide for an
    exact zero-extension representation, while rounded coefficients can
    change the cancellation of dummy-coordinate contributions. Recovery and
    numerical reduction errors follow their separate owners.

    The coordinates and the valued selector bits (success, then conditions)
    must partition the native wires, qubit zero least significant. The
    complete native norm, the success-only mass and the physical-slice mass
    are validated by ``validate_saved_masses`` before any statistic is
    returned. Identical success and physical populations (no conditions,
    unpadded coordinates) share one mass pair.
    """
    from nwqlib.amplitudes import _slice

    if (type(native) is not np.ndarray or native.dtype != np.complex128
            or native.ndim != 1 or len(native) < 1 or len(native) & (len(native) - 1)
            or type(tile) is not int or not 1 <= tile <= 1024):
        raise ValueError("admitted complex128 native vector and tile required")
    F, n = len(native), len(coordinates)
    w = F.bit_length() - 1
    fixed = success + conditions
    bits = tuple(coordinates) + tuple(b for b, _ in fixed)
    if (w >= 63 or len(bits) != w or set(bits) != set(range(w))
            or any(type(b) is not int for b in coordinates)
            or any(type(b) is not int or type(v) is not int or v not in (0, 1)
                   for b, v in fixed)
            or type(dimension) is not int or not 0 <= dimension <= 1 << n
            or (has_moment and (table is None or table.num_qubits != n))):
        raise ValueError("coordinates and valued selectors must partition native wires")
    countA = F >> len(success)
    complete, eF = scaled_mass_tiles(lambda: selected_tiles(native, tile=tile), F)
    if not success:
        algorithm, eA = complete, eF
    else:
        algorithm, eA = scaled_mass_tiles(
            lambda: selected_tiles(native, success, tile=tile), countA)
    # A contiguous layout borrows a view. Other layouts use the 48d gather
    # allowance of readout_requirements.
    selected = (native[:0] if dimension == 0
                else _slice(native, coordinates, fixed, size=dimension))
    if not conditions and dimension == 1 << n:
        # Use precisely one published mass for identical populations.
        # Its allowance is independent of coordinate order.
        physical, e = algorithm, eA
    else:
        physical, e = scaled_mass_tiles(lambda: selected_tiles(selected, tile=tile), dimension)
    validate_saved_masses(complete, algorithm, physical, (F, countA, dimension),
                          delta=delta, window=window)
    moment = (scaled_projected_moment(selected, table, physical, e) if has_moment else {})
    return {"kernel": "pow2-mass-pauli-moment/1", "complete": complete,
            "success": algorithm, "physical": physical, **moment}


def recover_scaled_pair(pair, recovery, *, upward=False):
    """Multiply a saved native pair by recovery**2 without forming recovery**2.

    For ``recovery = m_G * 2**e_G`` the pair's mantissa is multiplied by
    ``m_G**2`` exactly, normalized, and ``2 e_G`` is added to its exponent.
    Recovered exponents are arbitrary Python integers. A claim about the
    physical value also needs the recovery owner's own error.
    """
    m, e = pair
    if recovery is None:
        return None
    local = Q(float(m)) * Q(recovery.mantissa)**2
    result_m, result_e = fraction_pair(local, upward=upward)
    if result_m == 0:
        return (0.0, 0)
    return (result_m, result_e + int(e) + 2 * recovery.exponent)


def readout_requirements(F, n, d, L, payload, *, tile=1024, has_moment=True,
                         success_count=None, same_mass=False, contiguous=False,
                         binding_work=0):
    """Return bytes and input-visit work for a scaled projected reduction.

    F=2**w, N=2**n, d<=N and L describe the saved state, coordinate
    population, physical prefix and kept packed rows. payload includes
    the original packed table and all live observable representations.
    success_count is the success-only population A. None reserves A=F
    plus a selector scan. A=F denotes an empty success selector.
    same_mass means no conditions and d=N. contiguous means _slice
    returns its strided view. Omitted layout facts reserve their costly
    branches. binding_work funds JSON decoding and label packing.

    A mass pass costs at most 16 visits per complex entry. A nonempty
    success selector adds 8F visits. A gather adds (4n+4)d visits.
    A moment adds 16d+24N+32L and the grouped Pauli-action work. The
    scaled inputs have components below one and N,L<2**51, so the
    grouped action cannot overflow and its ordered retry is unreachable.
    The shared action byte envelope still reserves retry scratch.

    With I_p=1 when physical mass is separately scanned, I_g=1 when
    ``_slice`` gathers and g=min(L,N), the work law is
    ``W_mass=16F+1[success nonempty](8F+16A)+16d I_p``,
    ``W_gather=(4n+4)d I_g``,
    ``W_moment=16d+24N+32L+(4L+g)+N(L+g+1)`` and
    ``W_1=W_mass+W_gather+4w+L+W_bind+1[moment and L>0] W_moment``,
    where the last two terms of W_moment are the grouped action with its
    initial output zero fill. With t=min(F,tile), s=min(N,tile) and
    (B_A, W_A) from ``pauli_action_requirements``, an active moment
    reserves ``16F+P+65536+max{96L, 64t, 16N+48d I_g+B_A+32N+3P+48L+64s}``
    bytes and a mass-only output ``16F+P+65536+max{64t, 48d I_g+64s}``.
    This work law is an upper bound on input visits. It still
    overcounts early zero or identity exits, uses g as a bound on flip
    groups and keeps conservative framing reserves.

    Bytes keep the conservative 3*payload scaled-storage reserve and
    65536 fixed bookkeeping allowance. They cover known buffers and
    the qualified CPython objects priced by projected_requirements,
    not simulator storage, allocator arenas or process RSS.
    """
    from nwqlib.operators._pauli import pauli_action_requirements

    N, w = 1 << n, F.bit_length() - 1
    if not (0 <= d <= N <= F < 2**51 and F & (F - 1) == 0
            and 1 <= tile <= 1024 and 0 <= L < 2**51):
        raise ValueError("invalid selected reducer dimensions")
    if success_count is not None and not d <= success_count <= F:
        raise ValueError("physical population must fit the success population")
    t, s = min(F, tile), min(N, tile)
    held = 16 * F + payload
    gather_bytes = 0 if contiguous else 48 * d
    gather_work = 0 if contiguous else (4 * n + 4) * d
    if success_count is None:
        masses = 40 * F
    elif success_count == F:
        masses = 16 * F
    else:
        masses = 24 * F + 16 * success_count
    if not same_mass:
        masses += 16 * d
    work = masses + gather_work + 4 * w + L + binding_work
    if not L or not has_moment:
        return held + 65536 + max(64 * t, gather_bytes + 64 * s), work
    g = min(L, N)
    action_bytes, action_work = pauli_action_requirements(N, L, g, complex_input=True)
    action_work -= N + L * N
    scaled_bytes = action_bytes + 32 * N + 3 * payload + 48 * L
    work += 16 * d + 24 * N + 32 * L + action_work
    size = held + 65536 + max(96 * L, 64 * t,
        16 * N + gather_bytes + scaled_bytes + 64 * s)
    return size, work


# The one exact scalar reduction of LCHS and QLS: its registered name, kernel
# tag and the order of its saved scalar pairs.
PROJECTED_MOMENTS = "projected_moments"
PROJECTED_KERNEL = "pow2-mass-pauli-moment/1"
PROJECTED_SCALARS = ("complete_mass", "success_mass", "physical_mass", "numerator", "numerator_radius")

# The receipt exclusion labels this kernel resolves in saved_state_error: its
# registration declares them, and LCHS and QLS publication pass them for a
# projected_moments point.
#
# The label says the native readout term does not cover masses
# subsequently formed from saved amplitudes. It does not identify a defect in
# the native state itself. Write the computed saved state as z and the unit
# ideal native state, up to one common phase, as psi. The premise is
# ||z - psi||_2 <= delta, the receipt's saved-state budget
# (PreparedArtifact.saved_state_error), which includes the charge of a host
# phase correction of z. For a selected population X with n_X complex
# entries, put m_X = 2 n_X, u = 2**-53, eta = 2**-1074 and
# gamma_k = k u/(1 - k u). The two-pass mass kernel (scaled_mass_tiles)
# finds the largest real or imaginary component, frames every component by a
# power of two, squares the components and sums them in bounded tiles. A
# product and the subsequent nonnegative summation tree are covered by
# a_X = gamma(m_X + 1), and the stored-pair-only bound is
# U_X = 16 m_X eta M_hat_X/(1 - (m_X + 1) u) (saved_mass_allowance), so
# |M_hat_X - M_X| <= a_X M_X + U_X for the exact squared norm M_X of the
# computed components. The state premise gives ||z||_2 <= 1 + delta and
# | ||z||_2**2 - 1 | <= 2 delta + delta**2, hence the complete-mass rule
# |F_hat - 1| <= 2 delta + delta**2 + a_F (1 + delta)**2 + U_F. The subset
# checks compare masses of the same computed state and need no additional
# native delta charge (validate_saved_masses). The success and physical
# populations are subsets of the complete one for every admitted layout, so
# the same proof covers each declared population. Resolving this host-readout
# label does not upgrade the receipt's first-order native-state budget to an
# all-orders simulator theorem or cover the algorithm's approximation error,
# and it resolves no other label: with any other exclusion, delta stays
# unavailable and the probability-window convention applies.
PROJECTED_MASS_EXCLUSIONS = ("amplitude-derived masses",)


def projected_parameters(*, coordinates, success, conditions=(), dimension, terms, moment=True):
    """Canonical reducer parameters binding the selected reduction's populations and observable.

    ``coordinates`` are the ordered coordinate qubits (qubit zero least
    significant), ``success`` and ``conditions`` the valued selector bits,
    ``dimension`` the original dimension d and ``terms`` the stored Pauli
    labels and real coefficients (qubit zero rightmost). ``moment`` False
    selects a mass-only reduction.
    """
    return {"kernel": PROJECTED_KERNEL, "coordinates": list(coordinates),
            "success": [list(item) for item in success], "conditions": [list(item) for item in conditions],
            "dimension": dimension, "terms": [[label, float(value)] for label, value in terms],
            "moment": bool(moment)}


def _projected_fields(parameters):
    """Validated tuples of one ``projected_moments`` parameter mapping."""
    if parameters.get("kernel") != PROJECTED_KERNEL or set(parameters) != {
            "kernel", "coordinates", "success", "conditions", "dimension", "terms", "moment"}:
        raise ValueError("projected moments need their kernel tag and selected bindings")
    coordinates = tuple(parameters["coordinates"])
    success = tuple(tuple(item) for item in parameters["success"])
    conditions = tuple(tuple(item) for item in parameters["conditions"])
    terms = tuple((label, value) for label, value in parameters["terms"])
    dimension, moment = parameters["dimension"], parameters["moment"]
    n = len(coordinates)
    if (not n or any(type(bit) is not int for bit in coordinates) or type(moment) is not bool
            or any(len(item) != 2 for item in success + conditions)
            or type(dimension) is not int or not 1 <= dimension <= 1 << n
            or any(type(label) is not str or len(label) != n or set(label) - set("IXYZ")
                   or type(value) not in (int, float) for label, value in terms)):
        raise ValueError("projected moments need nonempty integer coordinates, valued selectors, a fitting "
                         "dimension and real IXYZ terms of the coordinate width")
    return coordinates, success, conditions, dimension, terms, moment


def _pauli_table(terms, num_qubits):
    """Packed single-word table of real labeled terms, with zero coefficients dropped."""
    import numpy as np
    from nwqlib.operators._pauli import PauliTerms, _pauli_masks

    kept = tuple((label, value) for label, value in terms if value != 0)
    x = np.zeros((len(kept), 1), dtype=np.uint64)
    z = np.zeros((len(kept), 1), dtype=np.uint64)
    coefficients = np.zeros(len(kept), dtype=np.complex128)
    for index, (label, value) in enumerate(kept):
        flip, phase, _ = _pauli_masks(label)
        x[index, 0], z[index, 0], coefficients[index] = flip, phase, float(value)
    return PauliTerms._snapshot(num_qubits, x, z, coefficients)


def projected_populations(parameters, width):
    """Declared population sizes (complete F, success-only, physical d) of one reduction."""
    coordinates, success, _, dimension, _, _ = _projected_fields(parameters)
    return 1 << width, (1 << width) >> len(success), dimension


def projected_requirements(parameters, width):
    """Price separate packing and numerical phases after callback payload release.

    Requires _projected_execute to delete parameters and terms after its
    packed table is built and before reduce_scaled. The parameter string
    remains owned by the point. Parsed rows overlap packing but not the
    numerical phase. A qualified 64-bit CPython object allowance supplements
    the shared action owner's logical group metadata.

    M counts every serialized row, including zero coefficients. L counts
    nonzero packed rows. H is the sum of the parsed outer-list size, the
    row-list, label and coefficient sizes and the validated outer and row
    tuple sizes, counting a label or coefficient once per row, which
    safely overcounts sharing. J=512+64w+(n+40)M bounds canonical JSON
    characters for native float coefficients, IXYZ labels and wire indices
    below 51. Integer coefficients use their magnitude-bit count plus two
    in place of 32. Two JSON decodes, two label validations and one packing
    pass are included in ``W_bind=2J+2(n+4)M+(n+8)L+8w+64``. The work unit
    is a character or kernel-input visit, not a CPU-instruction or
    elapsed-time estimate.

    With T=32L, S=64+J and H_action=64L+192g+1024 for the extra action
    objects, an active moment requires
    ``16F+T+S+65536+max{H+96L, 64t, 16N+48d I_g+B_A+32N+3T+48L+H_action+64s}``
    bytes. A mass-only output sets T=0 and omits packing and the numerical
    action, leaving ``16F+S+65536+max(H, 64t, 48d I_g+64s)``. Work is that of
    ``readout_requirements`` with payload T+H+64+J. The direct production
    registry path, this callback and the checked interpreter are premises of
    the release. A wrapper that keeps another reference to the mapping needs
    the larger envelope P=32L+H+64+J throughout.
    """
    from nwqlib.operators._pauli import pauli_action_requirements

    coordinates, success, conditions, d, terms, moment = _projected_fields(parameters)
    n, M = len(coordinates), len(terms)
    F, N = 1 << width, 1 << n
    L = sum(value != 0 for _, value in terms)
    rows = parameters["terms"]
    H = sys.getsizeof(rows) + sum(
        sys.getsizeof(row) + sys.getsizeof(label) + sys.getsizeof(value)
        for row, (label, value) in zip(rows, rows, strict=True))
    H += sys.getsizeof(terms) + sum(sys.getsizeof(row) for row in terms)
    J = 512 + 64 * width + sum(
        n + 8 + (32 if type(value) is float else abs(value).bit_length() + 2)
        for _, value in terms)
    binding_work = 2 * J + 2 * (n + 4) * M + (n + 8) * L + 8 * width + 64
    contiguous = coordinates == tuple(range(coordinates[0], coordinates[0] + n))
    T = 32 * L if moment else 0
    _, work = readout_requirements(F, n, d, L, T + H + 64 + J,
        has_moment=moment, success_count=F >> len(success),
        same_mass=not conditions and d == N, contiguous=contiguous,
        binding_work=binding_work)
    held = 16 * F + T + 64 + J + 65536
    packing = H + (96 * L if moment else 0)
    gather = 0 if contiguous else 48 * d
    t, s = min(F, 1024), min(N, 1024)
    if not moment or not L:
        return held + max(packing, 64 * t, gather + 64 * s), work
    g = min(L, N)
    action_bytes, _ = pauli_action_requirements(N, L, g, complex_input=True)
    objects = 64 * L + 192 * g + 1024
    numerical = 16 * N + gather + action_bytes + 32 * N + 3 * T + 48 * L + objects + 64 * s
    return held + max(packing, 64 * t, numerical), work


def _projected_shape(parameters):
    from nwqlib.core.planning import ReducerOutput

    _projected_fields(parameters)
    return (ReducerOutput("float64", (len(PROJECTED_SCALARS),)), ReducerOutput("int64", (len(PROJECTED_SCALARS),)),
            ReducerOutput("int64", (2,)))


def _projected_work(parameters, width):
    return projected_requirements(parameters, width)[1]


def _projected_execute(state, parameters, bindings, *, context):
    """Reduce the saved state using the producing receipt's qualified mass window.

    The registry declares amplitude-derived masses resolved because
    scaled_mass_tiles and saved_mass_allowance bound the populations this
    kernel squares and sums. The context carries the receipt's saved-state
    budget after that resolution. The reducer stays registered phase
    sensitive because its lost-component diagnostic can change under a phase
    rotation, so its context has no ``state_error``. Its masses and
    quadratic-form target are invariant, and their finite host errors are
    priced separately, so the mass checks use
    ``context.modulo_phase_state_error``, which includes the host phase
    product's charge. When native exclusions make that budget unavailable,
    they use the propagated window ``context.probability_window``, and the
    receipt's exclusions stay visible. This validates the saved masses under
    the stated window convention, without claiming phase-defined or general
    native accuracy.
    """
    import numpy as np

    coordinates, success, conditions, dimension, terms, moment = _projected_fields(parameters)
    n = len(coordinates)
    table = _pauli_table(terms, n) if moment else None
    # The packed table owns byte copies of the terms. Releasing the parsed
    # rows here is the premise of projected_requirements' numerical phase.
    del terms, parameters
    has_moment = table is not None and len(table) > 0
    result = reduce_scaled(
        state, coordinates=coordinates, success=success, conditions=conditions,
        dimension=dimension, table=table, delta=context.modulo_phase_state_error,
        window=context.probability_window, has_moment=has_moment,
    )
    pairs = [result["complete"], result["success"], result["physical"],
             result.get("numerator", (0.0, 0)), result.get("host_radius", (0.0, 0))]
    return (np.array([m for m, _ in pairs], dtype=np.float64), np.array([e for _, e in pairs], dtype=np.int64),
            np.array([result.get("lost_state_components", 0), result.get("lost_coefficients", 0)], dtype=np.int64))


def projected_statistics(chunk, parameters, width, *, delta=None, window):
    """Validated saved pairs of one ``projected_moments`` point chunk.

    Reads the chunk's three components by their registered order, checks
    each pair, and applies ``validate_saved_masses``.
    The populations come from the bound declaration. Mass validation uses the
    producing receipt's qualified delta when supplied and its probability
    window otherwise. Returns dict(name -> pair) for PROJECTED_SCALARS plus
    the two loss counts.
    """
    values = {item.component: item for item in chunk.values if getattr(item, "kind", None) == "reduced"}
    if set(values) != {0, 1, 2} or len(chunk.values) != 3:
        raise ValueError("projected moments need their three registered components")
    mantissas, exponents, losses = values[0].real, values[1].integers, values[2].integers
    if len(mantissas) != len(PROJECTED_SCALARS) or len(exponents) != len(PROJECTED_SCALARS) or len(losses) != 2:
        raise ValueError("projected moments differ from their registered shape")
    pairs = {name: (float(m), int(e)) for name, m, e in zip(PROJECTED_SCALARS, mantissas, exponents, strict=True)}
    for name, pair in pairs.items():
        value = pair_value(pair)
        if name != "numerator" and value < 0:
            raise ValueError(f"saved {name} is negative")
    if any(count < 0 for count in losses):
        raise ValueError("saved loss counts are negative")
    validate_saved_masses(
        pairs["complete_mass"], pairs["success_mass"], pairs["physical_mass"],
        projected_populations(parameters, width), delta=delta, window=window,
    )
    return dict(pairs, lost_state_components=int(losses[0]), lost_coefficients=int(losses[1]))


def pair_ratio(numerator, mass):
    """Binary64 value of numerator/mass from two saved pairs, or None when unrepresentable or mass is zero."""
    if mass[0] == 0:
        return None
    try:
        value = ldexp(numerator[0] / mass[0], numerator[1] - mass[1])
    except OverflowError:
        return None
    return value if isfinite(value) and (value != 0 or numerator[0] == 0) else None


def pair_float(pair):
    """Binary64 value of a saved pair, or None when a nonzero value is unrepresentable."""
    try:
        value = ldexp(pair[0], pair[1])
    except OverflowError:
        return None
    return value if isfinite(value) and (value != 0 or pair[0] == 0) else None


def _register_projected_moments():
    from nwqlib.core.planning import READOUT_REDUCERS, Reducer

    READOUT_REDUCERS[PROJECTED_MOMENTS] = Reducer(
        shape=_projected_shape,
        work=_projected_work,
        execute=_projected_execute,
        receives_context=True,
        state_error_resolutions=PROJECTED_MASS_EXCLUSIONS,
    )


_register_projected_moments()


def qwc_groups(terms, num_qubits, *, max_comparisons, max_bytes, limit_name="max_comparisons"):
    """First-fit qubit-wise commuting groups of the nonzero nonidentity labels.

    Returns ``(groups, bases, comparison_count)``: each group's member labels
    in first-fit order, its accumulated basis (qubit zero rightmost) and the
    actual evaluated comparison count. ``limit_name`` is forwarded to
    ``PauliTerms.group``, which admits each candidate tile against
    ``comparisons + charge <= max_comparisons`` and names the caller's limit
    on refusal. With no grouped label no comparison is needed, so only the cap
    is validated and zero comparisons are returned.
    """
    labels = tuple(dict.fromkeys(label for label, value in terms
                                 if value != 0 and any(axis != "I" for axis in label)))
    if not labels:
        if type(max_comparisons) is not int or max_comparisons < 1:
            raise ValueError("max_comparisons must be a positive integer")
        return (), (), 0
    table = _pauli_table(tuple((label, 1.) for label in labels), num_qubits)
    grouping = table.group(strategy="qwc", max_bytes=max_bytes, max_comparisons=max_comparisons,
                           limit_name=limit_name)
    groups, bases = [], []
    for members in grouping.groups:
        x = z = 0
        for index in members:
            x |= int(table.x[index, 0])
            z |= int(table.z[index, 0])
        groups.append(tuple(labels[index] for index in members))
        bases.append("".join("Y" if (x >> bit) & (z >> bit) & 1 else "X" if (x >> bit) & 1
                             else "Z" if (z >> bit) & 1 else "I" for bit in reversed(range(num_qubits))))
    return tuple(groups), tuple(bases), grouping.comparison_count


def weighted_group_moments(bits, counts, selected, masks, coefficients):
    """Weighted outcome mean, second moment and mean variance of one QWC group.

    Reduce all Pauli parities in one qubit-wise-commuting group from the same
    returned counts. The group's basis rotates each active coordinate once.
    A label's support mask selects its parity, and valued success selectors
    define its population. Weighted group variances include within-shot
    covariance. A physical-prefix mass is measured in an unrotated
    coordinate setting. Counts from a rotated basis do not identify
    dummy-coordinate membership.

    The parity of label j is ``(-1)**popcount(outcome & support_j)``. For a
    group of n shots with observed weighted values y, the unbiased
    mean-variance estimate is ``(mean(y*y) - mean(y)**2)/(n-1)`` for n > 1.
    The variance of the summed estimator is
    ``sum_g (sum_{j,k in g} c_j c_k Cov(X_j, X_k))/n_g``. Floating-point
    second moments are diagnostics and can be slightly inconsistent by
    rounding. The population is summed with Python integers.
    """
    import numpy as np

    bits = np.asarray(bits, dtype=np.uint64)
    counts = np.asarray(counts, dtype=np.int64)
    population = sum(map(int, counts))
    if population <= 0:
        raise ValueError("a group needs a positive returned population")
    y = np.zeros(bits.size, dtype=float)
    for mask, coefficient in zip(masks, coefficients, strict=True):
        parity = np.bitwise_count(bits & np.uint64(mask)) & 1
        y += coefficient * (1 - 2 * parity.astype(np.int8))
    y *= selected
    mean = float(np.dot(counts, y)) / population
    second = float(np.dot(counts, y * y)) / population
    variance = None if population == 1 else (second - mean * mean) / (population - 1)
    return mean, second, variance


def reduce_sample_arrays(chunk, *, observed, coordinates, success, conditions=(), dimension):
    """Original-coordinate sample indices and counts as sorted int64 arrays, plus shot populations.

    Returns ``(indices, counts, algorithm, physical)``: the distinct
    original-coordinate indices below ``dimension`` of outcomes whose
    selectors match, sorted increasingly, their summed counts (each at most
    ``MAX_COUNT``), the number of shots that match the success selectors,
    and the number of shots in the physical slice, which is the sum of the
    returned counts. Both shot numbers are Python integers. Dummy
    coordinates and zero counts are excluded.
    """
    import numpy as np
    from nwqlib.execution import MAX_COUNT

    if chunk.observation.kind != "counts":
        raise ValueError("sample reduction requires integer counts")
    if type(dimension) is not int or dimension < 1 or (dimension - 1).bit_length() > len(coordinates):
        raise ValueError("physical sample dimension must fit the coordinate register")
    positions = _positions(observed, success, conditions, coordinates)
    histogram = chunk.histogram()
    if histogram.entries and histogram.width != len(observed):
        raise ValueError("histogram width differs from selected observed bits")
    if histogram.width > 64:
        raise ValueError("sample arrays read at most 64 observed bits")
    outcomes = histogram.indices() if histogram.entries else np.zeros(0, dtype=np.uint64)
    weights = histogram.weights
    keep = weights > 0
    for bit, value in success:
        keep &= ((outcomes >> np.uint64(positions[bit])) & np.uint64(1)) == np.uint64(value)
    algorithm = sum(map(int, weights[keep]))
    for bit, value in conditions:
        keep &= ((outcomes >> np.uint64(positions[bit])) & np.uint64(1)) == np.uint64(value)
    index = np.zeros(outcomes.shape, dtype=np.uint64)
    for place, bit in enumerate(coordinates):
        index |= ((outcomes >> np.uint64(positions[bit])) & np.uint64(1)) << np.uint64(place)
    keep &= index < np.uint64(dimension)
    index, weights = index[keep].astype(np.int64), weights[keep]
    unique, inverse = np.unique(index, return_inverse=True)
    totals = [0] * len(unique)
    for slot, count in zip(inverse.tolist(), weights.tolist(), strict=True):
        totals[slot] += count
    if any(total > MAX_COUNT for total in totals):
        raise ValueError("a sample count exceeds MAX_COUNT")
    return unique, np.array(totals, dtype=np.int64), algorithm, sum(totals)
