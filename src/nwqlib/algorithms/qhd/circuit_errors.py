"""Numerical error entries of one native QHD circuit: preparation, block angles, omitted blocks and phase.

For the binary encoding the angle-formation and phase entries live here
(``binary_angle_formation``, ``binary_phase_allowance``). Its pruning and
AQFT bounds are those of the reconstruction
(``records.QHDReconstruction.pruning_error_bound`` and ``aqft_error_bound``),
and ``resources.circuit_resources`` names its other entries. The rest of
this module concerns the one-hot circuit.

These entries complete the stage chain of ``evolution_bounds`` for the
native circuit, on the valid one-hot subspace and with the physical phase
included. After the exact split product S with the exponents
``alpha^_k = delta a^_k`` and ``beta^_k = delta b^_k`` (exact products of the
binary64 ``delta = fl(total_time/num_steps)`` and the stored step weights),
the chain continues, each stage against the next:

1. S against the ideal blocks with their stored angles for the blocks the
   circuit keeps, and with their exact intended exponents for the blocks it
   omits (``angle_formation``).
2. those blocks against the ideal product without the omitted blocks, the
   blocks that ``rotation_threshold`` pruned or that planning omitted below
   the normal binary64 range (``rotation_pruning``).
3. the ideal kept blocks and the stored scalar phase ``exp(i Phi^)`` against
   the native circuit: the fused hopping gate's own parameter
   (``block_errors``), the phase-diagonal provider's phase wrap, which has no
   qualified bound, and the phase bookkeeping (``phase_allowance``).

Pruning precedes numerical lowering, because two separately lowered circuits
could acquire different phase roundings that the dropped angles alone do not
bound. The preparation compares input states and is added once to the
evolution error (``preparation_error``). Every exponential perturbation
below uses ``||exp(-i x G) - exp(-i y G)|| <= |x - y| ||G||`` for a Hermitian
G, which also gives ``||exp(-i y G) - I|| <= |y| ||G||``. The values are
exact rationals of the stored binary64 data, converted upward by the caller.
Backend simulation roundoff and later compilation are separate stages.
"""

import math
from fractions import Fraction
from math import inf, isqrt, nextafter, pi, sqrt

import numpy as np

# u = 2**-53 and the smallest positive subnormal 2**-1074, named _TAU here and
# lambda in the docstrings and comments of ``preparation_error`` and
# ``phase_allowance``. In ``block_errors``, lambda* and lambda^ are instead
# the intended and stored hopping exponents. The tau of ``phase_allowance`` is a
# different quantity, the binary64 2 pi (``math.tau``).
_U = Fraction(1, 2**53)
_TAU = Fraction(1, 2**1074)
# Upper rationals of pi and sqrt(2): one binary64 step above the correctly
# rounded values math.pi and math.sqrt(2), which differ from the true values by
# less than one step. math.pi lies below pi, so it is a lower rational of pi.
_PI_UP = Fraction(nextafter(pi, inf))
_PI_LOW = Fraction(pi)
_SQRT2_UP = Fraction(nextafter(sqrt(2.0), inf))


def preparation_error(reconstruction, method, grid):
    """Return a first-order estimate of structured one-hot preparation error against the exact product target.

    For one register, let w be the exact nonnegative unit target, v the
    stored nonzero vector and z = v/||v|| the direction prepared by the
    exact chain. If e bounds ||v-w|| and eta bounds abs(||v||-1), then
    ||z-w|| <= e+eta by the triangle inequality and <= 2e by the reverse
    triangle inequality. Nonnegative unit vectors independently give
    ||z-w|| <= sqrt(2). Thus d_dir <= min(sqrt(2), e+eta, 2e).

    Write u = 2**-53, lambda = 2**-1074 and
    eta_K = 3u/(1-2u) + sqrt(K)*lambda, the norm allowance of _normalized.
    Uniform and periodic kinetic-ground vectors have exact uniform
    directions. A Dirichlet kinetic-ground register uses 8u+eta_K, with
    8u the first-order vector estimate of variable_errors.

    For Gaussian preparation, A = u*reconstruction.initial_state_error
    is the saved aggregate from restricted_state_error. Its construction
    bounds min(G,Q) upward, where G = prod_j(1+e_j)-1+P, P >= 0, and
    Q = sqrt(N**2+1), N >= 1. Therefore A >= min(sum_j e_j, sqrt(2)).
    If A < sqrt(2), sum_j e_j <= A. Otherwise the nonnegative unit-product
    cap applies. The complete product-direction error is consequently at
    most D_G = min(sqrt(2), A+d*eta_K). This uses the aggregate construction
    of restricted_state_error and its Gaussian arithmetic assumptions,
    without another amplitude evaluation or a native Kronecker-product
    operation.

    Tensor-product telescoping of unit directions gives a direction
    allowance D capped at sqrt(2). It is zero for a uniform direction,
    min(sqrt(2), d*(8u+eta_K)) for Dirichlet kinetic-ground preparation,
    and D_G for Gaussian preparation. The stored-vector chain contributes
    d*c_K*u to first order, c_K = K*(K-1)/2 + pi*(K-1), under the tail-norm
    and scalar-function assumptions of append_amplitude_chain, which is
    conservative for a chain that the lower-range cutoff shortens. That
    cutoff adds its exact charge Delta per register
    (``initial_state.chain_selection``), and the per-register charges add by
    the same telescoping. The result is min(2, D+d*c_K*u+sum Delta), still
    an estimate because higher-order chain terms are omitted. Later
    synthesis of preparation rotations belongs to the synthesis entry.

    The implementation uses exact rational arithmetic, upper constants
    for pi and sqrt(2), and ceil(sqrt(K)) in the subnormal allowance.
    Its caller converts the final value upward.
    """
    from .initial_state import GaussianState, KineticGroundState, UniformState, chain_selection

    state, k, d = method.initial_state, grid.num_grid_points, grid.num_variables
    root = isqrt(k) if isqrt(k) ** 2 == k else isqrt(k) + 1
    if isinstance(state, UniformState) or (isinstance(state, KineticGroundState) and grid.boundary == "periodic"):
        direction = Fraction(0)
    elif isinstance(state, GaussianState):
        # A + d eta_K with A = u initial_state_error, capped at sqrt(2) below.
        direction = _U * Fraction(reconstruction.initial_state_error) + d * (3 * _U / (1 - 2 * _U) + root * _TAU)
    else:
        # 8u + eta_K per register, eta_K = 3u/(1 - 2u) + sqrt(K) lambda.
        direction = d * (8 * _U + 3 * _U / (1 - 2 * _U) + root * _TAU)
    # c_K = K (K - 1)/2 + pi (K - 1) per register.
    chain = d * (Fraction(k * (k - 1), 2) + _PI_UP * (k - 1)) * _U
    # sum Delta over the registers whose chain the lower-range cutoff shortens
    cutoff = sum((chain_selection(alpha).charge for alpha in reconstruction.initial_amplitudes), Fraction(0))
    return min(Fraction(2), min(_SQRT2_UP, direction) + chain + cutoff)


def block_errors(reconstruction, method, grid):
    """Return the exact block-angle entries of the native circuit and the counts that its phase allowance needs.

    Formation (stage 1). An occurrence has the nominal duration ``t = r delta``,
    an exact rational with r = 1/2 for a second-order potential factor and
    for an odd-indexed link in the second-order kinetic layers, and r = 1
    otherwise (``compiler.QHDCompiler``). Planning admits ``delta/2`` as a
    normal number, so the stored half duration is exact
    (``schedules.step_weights``), and the nominal durations stay the
    reference of this stage. A hopping occurrence on a link of variable j intends
    the signed exponent ``lambda* = -t a^_k/(2 h_j**2)`` of
    ``G = (XX + YY)/2``, whose restriction has norm 1, and the stored
    ``angle`` is ``lambda^`` (``kinetic.KineticCompiler``), so it costs
    ``min(2, |lambda^ - lambda*|)``. A projector on s qubits for table value v
    intends ``zeta* = t b^_k v`` of the traceless ``P - I/2**s`` of norm
    ``1 - 2**-s`` (the identity part belongs to the phase ledger), so it costs
    ``min(2, (1 - 2**-s) |zeta^ - zeta*|)``. Both discrepancies are exact
    rationals, about ``3u |lambda*|`` and ``2u |zeta*|``, since planning
    admits every product in the normal range (``validation._normal_range``).

    Omitted blocks (stage 2). Every step intends each link for a total
    duration delta (first order once, second order the odd links twice for
    delta/2 and the even ones once) and each table entry for a total delta
    (one or two potential factors), so the intended exponent weight of a step
    is ``delta |a^_k| sum_j L_j/(2 h_j**2) + delta |b^_k| sum_S (1 - 2**-|S|) sum_v |v|``
    for L_j links per variable. The kept blocks' ``|lambda*|`` and
    ``(1 - 2**-s) |zeta*|`` at their nominal durations subtracted from it
    leave the exact weight of the blocks the circuit omits, pruned or omitted
    below the normal range (``potential.PotentialCompiler.select_occurrences``),
    which bounds their removal at their intended exponents. It is
    nonnegative because every kept block is one of the intended occurrences.

    Hopping parameters (stage 3). The native gate is
    ``XXPlusYYGate(theta^)`` with ``theta^ = 2 fl(2 t^ c)`` of the stored
    duration t^ and coefficient (``pauli_evolution._hopping_parameter``),
    whose restricted exponent is ``theta^/2``, so a kept block adds
    ``min(2, |theta^/2 - lambda^|)``. The builder doubles the stored angle's
    own product, so this is zero whenever theta^ is finite.

    Returns:
        A dict with ``formation``, ``omitted`` and ``hopping`` (exact
        rationals) and ``dense``, the kept projector blocks with the
        phase-diagonal provider, both potential halves included.

    Raises:
        ValueError: A hopping gate parameter is not a finite binary64
            number, so the native circuit cannot be built.
    """
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import _hopping_parameter

    k = grid.num_grid_points
    inverse_squares = [1 / Fraction(grid.spacing(j)) ** 2 for j in range(grid.num_variables)]
    tables = {table.support: table.values.array for table in reconstruction.support_values}
    dt = Fraction(method.total_time / method.num_steps)
    second = method.trotter_order == 2
    # Nominal durations: delta/2 for a second-order potential factor and odd-indexed link, delta otherwise.
    link_durations = {link: dt / 2 if second and index % 2 else dt for index, link in enumerate(grid.links())}
    potential_duration = dt / 2 if second else dt
    formation = hopping = kept = Fraction(0)
    dense = 0
    kinetic_weights = potential_weights = Fraction(0)
    for group, (_time, a, b) in zip(reconstruction.steps, reconstruction.step_weights, strict=True):
        a, b = Fraction(a), Fraction(b)
        kinetic_weights += abs(a)
        potential_weights += abs(b)
        for block in group:
            stored = Fraction(block.angle)
            if block.kind == "kinetic":
                duration = link_durations[(block.support[0] % k, block.support[1] % k)]
                # lambda* = -t a^_k/(2 h_j**2)
                intended = -duration * a * inverse_squares[block.support[0] // k] / 2
                formation += min(2, abs(stored - intended))
                kept += abs(intended)
                # theta^ as the native builder forms it, and its exact half.
                theta = _hopping_parameter(block.time_step, block.coefficient)
                hopping += min(2, abs(Fraction(theta) / 2 - stored))
                continue
            variables = tuple(q // k for q in block.support)
            position = 0
            for q in block.support:
                position = position * k + q % k
            # zeta* = t b^_k v and the traceless norm 1 - 2**-s.
            intended = potential_duration * b * Fraction(tables[variables][position])
            norm = 1 - Fraction(1, 2 ** len(block.support))
            formation += min(2, norm * abs(stored - intended))
            kept += norm * abs(intended)
            dense += block.provider == "diagonal_synthesis"
    links = grid.num_links
    intended = dt * kinetic_weights * links * sum(inverse_squares) / 2 + dt * potential_weights * sum(
        (1 - Fraction(1, 2 ** len(support))) * sum(abs(Fraction(v)) for v in values)
        for support, values in tables.items())
    return dict(formation=formation, omitted=intended - kept, hopping=hopping, dense=dense)


def phase_allowance(reconstruction, method, grid, counts):
    """Return an upper allowance, in radians, on the native circuit's global phase against the exact identity phase.

    The reference is ``Phi = -sum_k delta (a^_k sum_j h_j**-2 + b^_k c +
    b^_k sum_(S,r) 2**-|S| v_(S,r))`` with exact products of the stored data.
    ``method._compiled_phase_allowance`` bounds ``|Phi^ - Phi|`` for the
    stored ``physical_phase`` Phi^, the formation and compensated
    accumulation of the ledger (E_ledger), from the ledger's contribution
    count M (``method._phase_contributions``).
    ``method._native_phase_allowance`` adds the top-level native assignments,
    one per kept projector block and one for Phi^,
    ``E_top = 2u (Y + |Phi^|) + 20u (B + 1)`` with u = 2**-53, B kept
    projectors and ``Y = sum_b |zeta^_b| 2**-s_b``, under Qiskit 2.5.2's
    rounded ``rem_euclid`` phase setter. Both evaluate every operation upward.

    The top-level allowance leaves out the internal assignment of each
    phase-diagonal child, which assigns its wrapped angle divided by 2**s,
    4 <= s <= 7, of magnitude below 1/4. With lambda = 2**-1074 and tau the
    binary64 2 pi (``math.tau``), as in ``_native_phase_allowance``, an
    increment of magnitude v formed within lambda has
    ``h = tau + v + lambda``, addition error at most ``a = u h + lambda``, a
    removed-turn count at most ``k = (h + a)/tau + 1``, and both
    ``|tau - 2 pi|`` and the remainder rounding at most 4u, so it costs at
    most ``lambda + a + 4u k + 4u <= 2u v + 20u`` radians (``6 < tau < 7``).
    Repeated halving of the child's angle errs by less than lambda
    (``e_(j+1) <= e_j/2 + lambda/2``), so each child adds at most 21u, and the
    allowance is ``E_native + 21u B_dense``, charged once. It assumes
    round-to-nearest, gradual underflow, finite operations and finite exact
    divisors. A phase allowance E costs at most E in 2-norm, since
    ``|exp(i E) - 1| <= E``. ``counts`` is the dict of ``block_errors``.

    Planning establishes the stated relative-error premises for admitted
    contributions. Lower-range omissions are charged by their exact omitted
    identity action, separately from that arithmetic allowance
    (``compiler.QHDCompiler``, ``validation._normal_range``). The ledger
    formation and accumulation charges apply to the stored duration, weights
    and coefficients. Native assignment arithmetic has its separate
    allowance, including gradual underflow. The value bounds this phase
    stage only. It does not qualify dense lowering or the other stages of
    the circuit's state error.

    Returns:
        The allowance as an exact rational capped at 2.
    """
    from .method import _compiled_phase_allowance, _native_phase_allowance, _phase_contributions

    r = reconstruction
    ledger = _compiled_phase_allowance(r, method, grid, _phase_contributions(r, method))
    native = _native_phase_allowance(r, ledger)
    if native == inf:
        return Fraction(2)
    # E_native + 21u per phase-diagonal child.
    return min(Fraction(2), Fraction(native) + 21 * _U * counts["dense"])


def binary_phase_allowance(reconstruction, method, grid):
    """Return an upper allowance, in radians, on a native binary circuit's global phase.

    The Walsh terms are the Plan's stored ``QHDReconstruction.walsh_phase``,
    the ``binary.WalshPhaseTerms`` that ``binary.compile_binary_steps``
    returned for the stored blocks at planning. The binary phase ledger holds only the
    objective constant's contributions ``-fl(dt fl(b_k c))``, M of them
    (``method._phase_contributions``), and
    the Walsh diagonals keep their identity phases apart
    (``binary.WalshPhaseTerms``). ``method._native_phase_allowance`` with
    those Walsh terms gives
    ``E_native = E_ledger + F_W + 2u (Y_W + |Phi^|) + 20u (B_W + 1)``, which
    charges the formation of the Walsh identity phases once, their
    assignments to the circuit and the final assignment of the stored
    physical phase Phi^. The reconstruction allowance ``R_W`` belongs to the
    ``ir_product`` reference, not to the circuit, and is not added. A dense
    diagonal keeps its identity phase inside its gate definition, which is
    part of its lowering and not of this allowance.

    Planning establishes the stated relative-error premises for admitted
    contributions. Lower-range omissions are charged by their exact omitted
    identity action, separately from that arithmetic allowance
    (``binary.compile_binary_steps``, ``validation._normal_range``). The Walsh
    formation allowance charges the identity phases' own underflow.

    Returns:
        The allowance as an exact rational capped at 2.
    """
    from .method import _compiled_phase_allowance, _native_phase_allowance, _phase_contributions

    r = reconstruction
    native = _native_phase_allowance(
        r, _compiled_phase_allowance(r, method, grid, _phase_contributions(r, method)), r.walsh_phase)
    return Fraction(2) if native == inf else min(Fraction(2), Fraction(native))


def _trig_interval(value, argument, ratio):
    """Return ``[low, high]`` enclosing ``f(pi ratio)`` from the returned ``value = f(argument)``, f = sin or cos.

    Under the one-ulp premise the returned value is within ``2u`` of
    ``f(argument)`` for a value in [-1, 1], and both functions are
    1-Lipschitz, so with ``eta = max(|argument - ratio pi_low|,
    |argument - ratio pi_high|)`` the exact ``f(pi ratio)`` lies within
    ``2u + eta`` of the value. The exact angles lie in ``[0, pi/2]``, so the
    interval is intersected with [0, 1].
    """
    a, y = Fraction(argument), Fraction(value)
    eta = max(abs(a - ratio * _PI_LOW), abs(a - ratio * _PI_UP))
    return max(Fraction(0), y - 2 * _U - eta), min(Fraction(1), y + 2 * _U + eta)


def _spectral_scale():
    """Return ``[2 pi_low**2, 2 pi_high**2]``, which encloses ``2 pi**2``, the spectral scale times ``L**2``."""
    return 2 * _PI_LOW**2, 2 * _PI_UP**2


def _kinetic_walsh_summary(model, bits, spacing):
    """Return ``(C, B_c)`` of one variable's kinetic Walsh coefficients, or ``(None, None)`` when an enclosure is missing.

    ``C = sum_(m != 0) |c^_m|`` over the actual coefficients of
    ``binary.kinetic_walsh`` and ``B_c = sum_(m != 0) rho_m`` with
    ``rho_m = max(|c^_m - low_m|, |c^_m - high_m|)`` for an interval
    ``[low_m, high_m]`` that encloses the exact analytic coefficient c_m, so
    ``rho_m >= |c^_m - c_m|`` covers every rounding that formed c^_m. The bit
    reversal of ``qft_bit_reversal="relabel"`` permutes coefficients and
    targets together and leaves both sums unchanged.

    Spectral: ``c_m = s_m 2 pi**2/L**2`` with ``L = K h`` and the exact Walsh
    coefficients of the signed square, ``s_(2**l) = w_l/2`` and
    ``s_(2**l + 2**j) = w_l w_j/2`` for ``l < j`` with ``w_l = 2**l`` except
    ``w_(b-1) = -2**(b-1)``, and ``s_m = 0`` for every other nonidentity mask.
    The interval is ``s_m G`` with ``G`` enclosing ``2 pi**2/L**2``.

    Finite difference: ``c_m = s_w(pi/K) prod_(l in m, l < b-1) sin a_l
    prod_(l not in m, l < b-1) cos a_l/h**2`` with ``a_l = pi 2**l/K``,
    ``w = popcount(m)`` and ``s_w = cos, -sin, -cos, sin`` for
    ``w mod 4 = 0, 1, 2, 3``, for a mask that contains the top bit, and zero
    for any other. At b = 1 the only nonidentity coefficient is exactly
    ``-1/h**2``. Here h is the spacing and b the register's bit count.
    For b > 1, each factor is recomputed with the same call on the same
    binary64 argument as ``kinetic_walsh`` and enclosed by ``_trig_interval``
    under the one-ulp premise for ``math.sin`` and ``math.cos``, the premise
    ``binary.qft_error_bound`` states for its sines. On this branch,
    ``kinetic_fd_summary_up`` multiplies outward binary64 enclosures of the
    nonnegative factors and divides by outward binary64 enclosures of the
    exact ``h**2``, using the known sign for the coefficient's error radius.
    At b = 1 it forms the exact discrepancy from ``-1/h**2`` before
    converting the radius upward.

    The spectral sums stay exact rationals. Only masks of weight one or two
    have a nonzero analytic coefficient, so the enclosure is formed on that
    sparse set and every other mask adds its exact coefficient magnitude,
    which is zero for the exact zeros of ``kinetic_walsh``. The
    finite-difference summary is ``kinetic_fd_summary_up``: upper binary64
    values of C and B_c, converted exactly to Fractions for the later chain.
    Its C can exceed the exact coefficient-magnitude sum, and its ``B_c`` can
    exceed the sum of radii obtained by propagating the same factor
    intervals in exact rational arithmetic. An infinite upper value
    returns ``(None, None)``, which ``binary_angle_formation`` caps at 2 for a
    nonzero contribution.
    """
    from .binary import kinetic_walsh

    k = 1 << bits
    values = kinetic_walsh(model, bits, spacing)
    if model != "spectral":
        c_up, b_up = kinetic_fd_summary_up(values, bits, spacing)
        if not (math.isfinite(c_up) and math.isfinite(b_up)):
            return None, None
        return Fraction(c_up), Fraction(b_up)
    h2 = Fraction(spacing) ** 2
    low, high = (scale / (k * k * h2) for scale in _spectral_scale())
    weights = [Fraction(1 << level) for level in range(bits)]
    weights[-1] = -weights[-1]
    enclosed = {1 << l: weights[l] / 2 for l in range(bits)}
    enclosed.update({(1 << l) | (1 << j): weights[l] * weights[j] / 2
                     for l in range(bits) for j in range(l + 1, bits)})
    total_c = radii = Fraction(0)
    for mask in np.flatnonzero(values[1:]).tolist():
        value = abs(Fraction(float(values[mask + 1])))
        total_c += value
        if mask + 1 not in enclosed:
            radii += value
    for mask, s in enclosed.items():
        c = Fraction(float(values[mask]))
        radii += max(abs(c - s * low), abs(c - s * high))
    return total_c, radii


def kinetic_fd_summary_up(coefficients, bits, spacing):
    """Return upper binary64 values ``(C_up, B_up)`` of ``_kinetic_walsh_summary`` for finite differences.

    This finite-difference summary implements the interval recurrence
    below. It consumes the already formed, admitted
    coefficients of ``kinetic_walsh("finite_difference", ...)`` in their
    original order and returns upper floats. Every trigonometric factor gets
    one exact ``_trig_interval`` call, converted downward/upward, and each
    mask's interval starts from its base interval and propagates
    ``L' = down(L L_f)``, ``U' = up(U U_f)`` level by level, with a lower
    endpoint clamped at zero. The final lower bound is divided by an upper
    enclosure of the exact ``h**2`` and the upper bound by a lower one, both
    computed once from ``Fraction(spacing)**2``. A lower product rounded to
    zero stays zero, and an upper positive underflow moves to the smallest
    subnormal. For the stored coefficient magnitude c the upward radius is
    ``up(max(0, c - L, U - c))``, also the error radius of the signed
    coefficient, since all modeled factors are nonnegative before the known
    sign, and the radii are added upward. Coefficients whose top bit is
    absent are structurally zero and are not included in the radius pass. At
    b = 1 the exact analytic target is ``-1/Fraction(h)**2``. If an upper
    float is infinite it stays an upper bound. These formulas preserve an
    upper bound but not the exact-rational B_c or necessarily its final ulp.
    """
    from .binary import _scaled_absolute_sum_up
    from .evolution_bounds import _upward

    def down(q):
        x = float(q)
        return math.nextafter(x, -math.inf) if Fraction(x) > q else x

    def up(q):
        return _upward(q)

    def enclosure(value, argument, ratio):
        lo, hi = _trig_interval(value, argument, ratio)
        return max(0.0, down(lo)), up(hi)

    coefficients = np.asarray(coefficients, dtype=np.float64)
    c_up = _scaled_absolute_sum_up(coefficients[1:], 1.0)
    h2 = Fraction(spacing) ** 2
    if bits == 1:
        return c_up, up(abs(Fraction(float(coefficients[1])) + 1 / h2))
    k, top = 1 << bits, bits - 1
    masks = np.arange(1 << top, k, dtype=np.int64)
    base = math.pi / k
    slo, shi = enclosure(math.sin(base), base, Fraction(1, k))
    clo, chi = enclosure(math.cos(base), base, Fraction(1, k))
    odd = (np.bitwise_count(masks) & 1) != 0
    low = np.where(odd, slo, clo)
    high = np.where(odd, shi, chi)
    for level in range(top):
        if level != 0:
            angle = math.pi * (1 << level) / k
            ratio = Fraction(1 << level, k)
            slo, shi = enclosure(math.sin(angle), angle, ratio)
            clo, chi = enclosure(math.cos(angle), angle, ratio)
        selected = ((masks >> level) & 1) != 0
        lower_factor = np.where(selected, slo, clo)
        upper_factor = np.where(selected, shi, chi)
        with np.errstate(all="ignore"):
            low = np.maximum(0.0, np.nextafter(low * lower_factor, -np.inf))
            high = np.nextafter(high * upper_factor, np.inf)
    with np.errstate(all="ignore"):
        low = np.maximum(0.0, np.nextafter(low / up(h2), -np.inf))
        high = np.nextafter(high / down(h2), np.inf)
        actual = np.abs(coefficients[1 << top:])
        radius = np.maximum(0.0, np.maximum(
            np.nextafter(actual - low, np.inf), np.nextafter(high - actual, np.inf)))
    return c_up, _scaled_absolute_sum_up(radius, 1.0)


def _kinetic_energy_summary(grid, var_index, model):
    """Return ``(M_E, H_E)`` of one variable's kinetic energies for dense kinetic formation.

    ``M_E = max_q |E^_q|`` over the energies of ``split_step.kinetic_eigenvalues``
    and ``H_E = max_q max(|E^_q - low_q|, |E^_q - high_q|)`` for intervals
    enclosing the exact energies. Spectral: ``E_q in q**2 G`` for the signed
    index q with ``G`` enclosing ``2 pi**2/L**2``. Finite difference:
    ``E_q = 2 sin(pi |q|/K)**2/h**2``, exact 0 at q = 0 and exactly ``2/h**2``
    at ``|q| = K/2``. Otherwise the sine returned by NumPy at the source's
    argument ``x_q = pi (|q|/K)`` is enclosed by ``_trig_interval`` under the
    one-ulp premise for the actual NumPy ``sin``, and the interval
    ``[2 s_low**2/h**2, 2 s_high**2/h**2]`` follows.
    """
    from .split_step import kinetic_eigenvalues

    k, h = grid.num_grid_points, grid.spacing(var_index)
    h2 = Fraction(h) ** 2
    energies = [Fraction(e) for e in kinetic_eigenvalues(grid, var_index, model).tolist()]
    index = np.arange(k)
    signed = np.where(index < -(-k // 2), index, index - k)
    # The source's arguments x_q = pi (|q|/K) and their sines, formed with the same array operations.
    arguments = np.pi * (np.abs(signed) / k)
    sines = np.sin(arguments)
    radius = Fraction(0)
    scale_low, scale_high = (scale / (k * k * h2) for scale in _spectral_scale())
    for q, (energy, magnitude) in enumerate(zip(energies, np.abs(signed).tolist(), strict=True)):
        if model == "spectral":
            ends = (magnitude**2 * scale_low, magnitude**2 * scale_high)
        elif magnitude == 0:
            ends = (Fraction(0),)
        elif 2 * magnitude == k:
            ends = (2 / h2,)
        else:
            s_low, s_high = _trig_interval(float(sines[q]), float(arguments[q]), Fraction(magnitude, k))
            ends = (2 * s_low**2 / h2, 2 * s_high**2 / h2)
        radius = max(radius, *(abs(energy - end) for end in ends))
    return max(abs(e) for e in energies), radius


def binary_angle_formation(reconstruction, method, grid, model):
    """Bound binary angle formation against the exact stored-weight split product.

    The reference uses ``X = r Fraction(dt) Fraction(weight)``, r = 1, or 1/2
    for a second-order potential half, with ``dt = fl(total_time/N)`` and the
    step's stored weight, the stored potential tables as exact data, the
    analytic kinetic energies and mathematical pi in the QFT. The emitted
    exponent x (``QHDBinaryBlock.exponent``) uses the actual stored duration,
    so ``eps_x = |Fraction(x) - X|`` is the product rounding, the stored half
    duration being exact (``schedules.step_weights``). With u = 2**-53 and
    ``gamma_n = n u/(1 - n u)``, every quantity below is an exact rational
    of the stored binary64 data, each block occurrence's allowance is
    converted upward, and the sum is exact. A coefficient omitted by the
    Walsh normalization enters here through ``rho_m``. A rotation or dense
    phase entry omitted because its product would fall below ``2**-1022``
    is compared through its exact value before omission, which has no
    formation error, and its removal is charged in ``rotation_pruning``
    (``binary.PhaseTable.synthesize``), so no omission is charged twice.

    Walsh blocks. The identity phase stays at ``-X c0`` on both sides, since
    ``identity_phase`` charges its formation (``F_W``). Let rho_m enclose
    ``|c^_m - c_m|``, ``C = sum_(m != 0) |c^_m|``, ``B_c = sum_(m != 0) rho_m``
    and ``R_theta = (1/2) sum_(m != 0) |theta^_m - 2 x c^_m|`` with the
    computed angles ``theta^_m = fl((2 x) c^_m)``. The Z strings commute and
    ``Rz(theta) = exp(-i theta Z/2)``, so splitting each discrepancy as
    ``theta^_m/2 - x c^_m + (x - X) c^_m + X (c^_m - c_m)`` bounds every
    diagonal phase error by ``L_W = |X| B_c + eps_x C + R_theta``, and a phase
    diagonal with errors delta_z differs from the target by
    ``max_z 2 |sin(delta_z/2)|``, so

        ``e_W <= min(2, |X| B_c + eps_x C + R_theta)``.

    Every mask participates, including spurious nonzero coefficients, and
    the population is taken before pruning. The doubling of x is exact, a
    kept or threshold-pruned angle is a normal product within
    ``u |2 x c^_m|`` of ``2 x c^_m``, and a range-omitted angle keeps its
    exact virtual value ``2 x c^_m``, so ``R_theta <= u |x| C``. For a
    potential table v of length ``M = 2**n``, each output of the butterfly
    (``walsh_coefficients``) is a signed sum along paths of n roundings, a
    subnormal sum or difference of binary64 numbers is exact, and the
    normalization by M is an exact power-of-two scaling or an omission
    (``binary.walsh_admission``), so ``rho_m = gamma_n a + d_m`` with
    ``a = sum_z |v_z|/M`` and ``d_m`` the exact normalization omission
    (Higham, The Accuracy of Floating Point Summation, SIAM J. Sci. Comput.
    14 (1993) 783-799, doi:10.1137/0914050, Sec. 3, Eq. (3.6), p. 788, and
    the pairwise bound of ``binary.walsh_coefficients``), which needs
    ``n u < 1``, and
    ``B_c = (M - 1) gamma_n a + sum_(m != 0) d_m``.
    Kinetic coefficients take the enclosures of ``_kinetic_walsh_summary``.

    The table reductions return upper bounds on the exact absolute sums of
    the stored binary64 entries. A correctly rounded ``fsum`` is moved upward
    before division, and every later positive operation is evaluated
    outward. When an unscaled sum is not representable, the reduction uses
    scaled upper terms or exact integer units. The resulting angle-formation
    allowance may be larger than the exact-rational evaluation but cannot be
    smaller under these arithmetic premises. The upper values a and C are
    computed once per table and converted exactly to Fractions, and the rest
    of the scalar chain, exact exponent discrepancies and per-block caps
    included, stays exact. An infinite upper a or C caps a nonzero block
    contribution at 2.

    Dense blocks. A potential block forms ``phi^_z = fl(-x v_z)`` against the
    target ``-X v_z``, whose largest discrepancy is at most
    ``(eps_x + u |x|) max_z |v_z|``, the rounding of the product and of the
    exponent, with an omitted phase entry compared through its exact
    virtual value. A kinetic block with ``M_E``, ``H_E`` of
    ``_kinetic_energy_summary`` forms ``phi^_q = fl(-x E^_q)`` against
    ``-X E_q``, and
    ``phi^_q + X E_q = (phi^_q + x E^_q) + (X - x) E^_q + X (E_q - E^_q)``
    gives ``min(2, u |x| M_E + eps_x M_E + |X| H_E)``. Dense
    formation covers the whole diagonal, its mean included. The wrap,
    multiplexor rounding and the definition's global phase of the dense
    lowering belong to the later ``diagonal_wrap`` entry.

    QFT angles. Each controlled phase takes ``ldexp(math.pi, -r)``, exact
    scaling of ``p_low = math.pi``, and ``||CP(a) - CP(b)|| =
    2 |sin((a - b)/2)| <= |a - b|``, so a full b-bit QFT with its b - r
    gates at distance r errs by at most
    ``eps_pi sum_(r=1..b-1) (b - r) 2**-r = eps_pi (b - 2 + 2**(1-b))`` with
    ``eps_pi = p_high - p_low >= |math.pi - pi|``, and a conjugation
    ``F^dagger D F`` of a unitary diagonal by twice that. The d N_s
    conjugations give ``min(2, 2 d N_s eps_pi (b - 2 + 2**(1-b)))``, zero for
    b = 1. This charge uses the full QFT, and the later ``aqft`` entry
    compares the full stored-angle QFT with its truncation.

    The result is ``min(2, sum_b e_b + E_(Q,pi))`` over every selected block
    occurrence, both potential halves and every kinetic variable of every
    step included. The Walsh identity coefficient and the objective constant
    are excluded because ``identity_phase`` charges their formation and
    native assignments. The work is two upward ``fsum`` reductions over each
    distinct potential table (``binary._scaled_absolute_sum_up``), ``O(b K)``
    array operations per finite-difference kinetic table with one exact
    enclosure per trigonometric factor (``kinetic_fd_summary_up``), and
    constant work per occurrence, with no transform, circuit or state. The trigonometric
    enclosures rest on the one-ulp premise for ``math.sin``, ``math.cos``
    and NumPy's ``sin``, a platform assumption like that of
    ``binary.qft_error_bound``. The premise is conditional. The bound was
    tested against an independent exact evaluation on macOS arm64 and Linux
    aarch64, and NumPy's SIMD ``sin`` on Linux x86-64 hosts has not been
    checked (docs/dependency_issues.md, "Platform math libraries").

    Returns:
        The bound as an exact rational capped at 2, or None when a table value,
        kinetic energy or coefficient is not finite.
    """
    from .evolution_bounds import _upward

    r = reconstruction
    tables = (*model.potential.values(), *model.kinetic)
    if not all(np.all(np.isfinite(t.values)) and np.all(np.isfinite(t.coefficients)) for t in tables):
        return None
    dt = Fraction(method.total_time / method.num_steps)
    bits = model.bits
    from .binary import _scaled_absolute_sum_up

    # max |v| of each support table, its cached exact extremum (records.SupportValues.magnitude).
    magnitudes = {tuple(t.support): t.magnitude for t in r.support_values}
    potential = {}
    for support, table in model.potential.items():
        n, size = table.qubits, table.values.size
        # Upper a and C, computed once and converted exactly, or None when not finite.
        a_up = _scaled_absolute_sum_up(table.values, float(size))
        c_up = _scaled_absolute_sum_up(table.coefficients[1:], 1.0)
        gamma = n * _U / (1 - n * _U)
        # C, B_c = (M - 1) gamma_n a + sum_(m != 0) d_m and max |v|
        finite = math.isfinite(a_up) and math.isfinite(c_up)
        potential[support] = (
            Fraction(c_up) if finite else None,
            (size - 1) * gamma * Fraction(a_up) + table.omission.charge if finite else None,
            Fraction(magnitudes[support]),
        )
    kinetic, energies = {}, {}
    total = Fraction(0)
    for (_time, kinetic_weight, potential_weight), group in zip(r.step_weights, r.steps, strict=True):
        for block in group:
            is_kinetic = block.kind == "binary_kinetic"
            half = not is_kinetic and method.trotter_order == 2
            exact = (dt / 2 if half else dt) * Fraction(kinetic_weight if is_kinetic else potential_weight)
            x = Fraction(block.exponent)
            eps_x = abs(x - exact)
            if block.synthesis == "walsh_rotations":
                if is_kinetic:
                    j = block.variables[0]
                    if j not in kinetic:
                        kinetic[j] = _kinetic_walsh_summary(model.kinetic_model, bits, model.spacings[j])
                    total_c, radii = kinetic[j]
                else:
                    total_c, radii, _largest = potential[tuple(block.variables)]
                if total_c is None:
                    # An infinite upper C or B_c caps a nonzero contribution at 2. A zero
                    # exponent with zero target still contributes zero.
                    bound = Fraction(0) if exact == 0 and x == 0 else Fraction(2)
                else:
                    # e_W = min(2, |X| B_c + eps_x C + u |x| C)
                    bound = abs(exact) * radii + eps_x * total_c + _U * abs(x) * total_c
            elif is_kinetic:
                j = block.variables[0]
                if j not in energies:
                    energies[j] = _kinetic_energy_summary(grid, j, model.kinetic_model)
                largest, radius = energies[j]
                # u |x| M_E + eps_x M_E + |X| H_E
                bound = _U * abs(x) * largest + eps_x * largest + abs(exact) * radius
            else:
                largest = potential[tuple(block.variables)][2]
                # (eps_x + u |x|) max |v|
                bound = (eps_x + _U * abs(x)) * largest
            total += Fraction(_upward(min(Fraction(2), bound)))
    # E_(Q,pi) = min(2, 2 d N_s eps_pi (b - 2 + 2**(1-b)))
    conjugations = method.num_steps * grid.num_variables
    qft = min(Fraction(2), 2 * conjugations * (_PI_UP - _PI_LOW) * (bits - 2 + Fraction(2, 1 << bits)))
    return min(Fraction(2), total + qft)
