"""Physical application facts and constant-source branch grids that passed the limit checks, without state PREP or evolution."""

from nwqlib._limits import ResourceLimitError, exceeds, optional_limit

from dataclasses import dataclass
from math import expm1, fsum, isfinite, ldexp, sqrt

import numpy as np

from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.operators.inputs import _freeze_array
from .solution_error_budget import admitted_exp, extracted_phase, scaled_sum


@dataclass(frozen=True, kw_only=True)
class LCHSApplication:
    """One application of the LCHS propagator to a physical input, with its magnitude kept range-safe.

    Attributes:
        kind: ``"initial"`` or ``"source"``.
        start_time: Time at which the input enters: 0 for the initial
            state, the Gauss-Legendre node r_q for a source application.
        elapsed_time: Evolution time t_a = T - start_time.
        weight: Quadrature weight w_a: 1 for the initial state, the
            Gauss-Legendre weight w_q (time units) for a source application.
        input_norm: Physical norm nu_a of the input, as its PhysicalScale.
        mass: m_a = w_a*nu_a*exp(s*t_a) for the signed shift s, as a
            PhysicalScale whose integer exponent can lie outside binary64.
    """

    kind: str
    start_time: float
    elapsed_time: float
    weight: float
    input_norm: object
    mass: object


@dataclass(frozen=True, kw_only=True)
class LCHSApplicationFacts:
    """The physical applications of one LCHS solution, shared by the coherent and sampled routes.

    Masses are PhysicalScale values: the base-2 logarithm of a mass is its
    integer ``exponent`` plus log2 of its mantissa, and no absolute binary64
    logarithm is formed. ``application_facts`` builds this record.

    Attributes:
        kind: ``"homogeneous"``, ``"discrete_source"`` or
            ``"continuous_source"``.
        final_time: Physical final time T.
        active_time: T_e, the largest elapsed time of any application, or of
            the continuous source interval.
        signed_shift: Signed shift s (``solution_error_budget.signed_shift``),
            negative under known decay, in inverse time.
        dissipative_scalar: c_L, the identity coefficient taken out of the
            evolved dissipative child L_c = L - c_L*I.
        hamiltonian_scalar: c_H, the identity coefficient taken out of
            H_c = H - c_H*I.
        applications: ``LCHSApplication`` records: the initial application
            when its norm is nonzero, then one per source node for
            ``"discrete_source"``.
        source_interval: (t_lo, t_hi), the active elapsed-time interval of a
            continuous source, else None.
        source_norm: nu_b of a continuous source, else None.
        source_mass: nu_b times the integral of exp(s*t) over source_interval,
            as a PhysicalScale, for a continuous source, else None.
    """

    kind: str
    final_time: float
    active_time: float
    signed_shift: float
    dissipative_scalar: float
    hamiltonian_scalar: float
    applications: tuple
    source_interval: tuple | None = None
    source_norm: object = None
    source_mass: object = None


def _source_mass(norm, shift, low, high):
    """Return M_b = nu_b * integral over [low, high] of exp(s*t) dt as a PhysicalScale.

    With Delta = high - low, sigma = |s| and
    Phi(x) = (1 - exp(-x))/x for x > 0, Phi(0) = 1, so 0 < Phi(x) <= 1,

        M_b = nu_b*Delta*Phi(sigma*Delta)*exp(s*a)   for s < 0 (anchor a = low),
        M_b = nu_b*Delta                             for s = 0,
        M_b = nu_b*Delta*Phi(sigma*Delta)*exp(s*b)   for s > 0 (anchor b = high).

    Exact source zero or empty support returns the zero scale before any
    exponential work. Otherwise E = exp(s*anchor) comes from ``scaled_exp``,
    and a nonfinite s*anchor or s*anchor/ln 2 is the existing explicit range
    refusal. The bounded factor is chosen from the floating product
    x = sigma*Delta, and every positive factor is combined by
    compose_recovery before any binary64 result:

    - x == 0 (product underflow): nu_b*Delta*E, the limit Phi = 1. The
      exact product is then at most 2**-1075, and 0 <= 1 - Phi(x) <= x/2.
    - 0 < x <= 1: nu_b*Delta*Phi(x)*E with Phi(x) = -expm1(-x)/x, which lies
      between 1 - exp(-1) and 1.
    - finite x > 1: nu_b*(1 - exp(-x))/sigma*E, the identical relation
      Delta*Phi(sigma*Delta) = (1 - exp(-sigma*Delta))/sigma, with the
      numerator -expm1(-x) between 1 - exp(-1) and 1 and the ratio passed
      as a pair without forming a reciprocal.
    - x == inf from two finite operands (reachable only for s < 0, since
      for s > 0 the anchor exponential s*b >= s*Delta is refused first):
      nu_b*E/sigma. Setting the numerator to 1 is a numerical saturation,
      not a zero tail; its exact defect nu_b*E*exp(-x)/sigma is below
      nu_b*E*exp(-64)/sigma, since the exact x exceeds 64.

    Both nonzero-sign forms evaluate expm1 only at a nonpositive argument.
    Endpoint subtraction, the floating x and expm1 keep their numerical
    rounding qualification; no factor is an outward enclosure. The forms
    follow from integral over [a, b] of exp(s*t) dt = exp(s*a)*(exp(s*Delta) - 1)/s
    = exp(s*b)*(1 - exp(-s*Delta))/s, the source term of ACL
    arXiv:2312.03916v2, Eq. (2), for a constant source.
    """
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    if norm.mantissa == 0 or low == high:
        return PhysicalScale(mantissa=0.0, exponent=0)
    delta = high - low
    if shift == 0:
        return compose_recovery(norm, delta)
    sigma = abs(shift)
    exponential = admitted_exp(shift*(low if shift < 0 else high))
    x = sigma*delta
    if x == 0:
        return compose_recovery(norm, delta, exponential)
    if x <= 1:
        return compose_recovery(norm, delta, -expm1(-x)/x, exponential)
    if isfinite(x):
        return compose_recovery(norm, (-expm1(-x), sigma), exponential)
    return compose_recovery(norm, (1.0, sigma), exponential)


def application_facts(*, kind, final_time, signed_shift, scalars, initial_norm, source_norm=None,
                      source_nodes=(), source_weights=(), source_interval=None):
    """Return the range-safe physical application facts of one LCHS solution.

    The constant-source solution is
    u(T) = exp(-A*T) u0 + integral over r in [0, T] of exp(-A*(T - r)) b dr
    (ACL arXiv:2312.03916v2, Eq. (2)). Application a applies the propagator
    for its elapsed time t_a to the direction of its input, with
    m_a = w_a*nu_a*exp(s*t_a): the initial state has t = T and w = 1, and
    source node r_q has t = T - r_q and the Gauss-Legendre weight w_q. Each
    mass is the PhysicalScale product of nu_a, w_a and exp(s*t_a)
    (``solution_error_budget.scaled_exp``), so neither a huge or tiny input
    norm nor the compensation is converted to binary64. Integer exponents
    stay exact. A ``"continuous_source"`` descriptor keeps the initial
    application and the source as nu_b and its interval, with
    source_mass = nu_b*integral of exp(s*t) dt over that interval
    (``_source_mass``). active_time is the largest elapsed time of a
    nonzero application, and a nonzero continuous source on a nonempty
    interval contributes its upper endpoint; a zero source or an empty
    interval contributes nothing, and with neither present it is 0.

    The caller selects and admits the source rule: this function computes
    no quadrature and does O(applications) scalar work. It reads no array
    of the system dimension.

    Args:
        kind: ``"homogeneous"``, ``"discrete_source"`` or
            ``"continuous_source"``.
        final_time: T.
        signed_shift: s.
        scalars: (c_L, c_H), the identity coefficients removed from the
            original L and H to form the evolved children L - c_L*I and
            H - c_H*I, whose extracted phase is -t*(k*(c_L + s) + c_H). The
            dense recipe, which evolves L + s*I and H itself, has
            (c_L, c_H) = (-s, 0).
        initial_norm: PhysicalScale of the initial vector. A zero scale
            means no initial application.
        source_norm: PhysicalScale of the source, None for homogeneous.
        source_nodes: Gauss-Legendre start times r_q on [0, T].
        source_weights: Matching weights w_q.
        source_interval: (t_lo, t_hi) of a continuous source,
            0 <= t_lo <= t_hi <= T.

    Raises:
        ApplicabilityError: When s*t or s*t/ln 2 is not finite for an
            application, the explicit range refusal.
        ValueError: For an unknown kind or an interval outside [0, T].
    """
    from nwqlib.problems.inputs import compose_recovery

    if kind not in ("homogeneous", "discrete_source", "continuous_source"):
        raise ValueError("application facts kind must be homogeneous, discrete_source or continuous_source")

    def application(role, start, elapsed, weight, norm):
        return LCHSApplication(kind=role, start_time=start, elapsed_time=elapsed, weight=weight, input_norm=norm,
                               mass=compose_recovery(norm, weight, admitted_exp(signed_shift*elapsed)))

    applications = [application("initial", 0., final_time, 1., initial_norm)] if initial_norm.mantissa > 0 else []
    if kind == "discrete_source" and source_norm.mantissa > 0:
        applications += [application("source", float(node), final_time-float(node), float(weight), source_norm)
                         for node, weight in zip(source_nodes, source_weights, strict=True)]
    active = max((item.elapsed_time for item in applications), default=0.)
    interval = norm = mass = None
    if kind == "continuous_source":
        low, high = (float(value) for value in source_interval)
        if not 0 <= low <= high <= final_time:
            raise ValueError("continuous source interval must satisfy 0 <= t_lo <= t_hi <= T")
        interval, norm = (low, high), source_norm
        mass = _source_mass(source_norm, signed_shift, low, high)
        if source_norm.mantissa > 0 and high > low:
            active = max(active, high)
    dissipative, hamiltonian = scalars
    return LCHSApplicationFacts(kind=kind, final_time=final_time, active_time=active, signed_shift=signed_shift,
        dissipative_scalar=dissipative, hamiltonian_scalar=hamiltonian, applications=tuple(applications),
        source_interval=interval, source_norm=norm, source_mass=mass)


@dataclass(frozen=True, kw_only=True)
class SourceBranchLayout:
    """Frozen coefficient state, time grid and the physical applications it represents.

    Address slot (a, j) holds kernel node k_j of application a, whose
    magnitude m_a (``LCHSApplication.mass``) is a PhysicalScale. The slot
    coefficient is c_aj = (m_a/2**scale_exponent)*c_j*exp(i*phi_aj), with
    c_j the stored raw kernel coefficient and phi_aj = -t_a*(k_j*a + b) the
    extracted scalar phase. The layout keeps the PREP amplitudes
    sqrt(|c_aj|/coefficient_l1_norm) and the phases arg(c_j) + phi_aj
    separately, and the recovery multiplies by
    gamma = coefficient_l1_norm*2**scale_exponent, so neither a magnitude
    nor a coefficient outside the binary64 range has to be formed
    (``select_source_branches`` derives both).

    Attributes:
        amplitudes: Normalized nonnegative PREP amplitudes, zero-padded to
            the power-of-two address count.
        phases: Phase arg(c_j) + phi_aj of each address slot in radians,
            reduced to [-pi, pi], 0 on padding, zero-coefficient and omitted
            slots.
        coefficient_l1_norm: sum over slots of |c_aj|, relative to
            2**scale_exponent.
        k_values: k-node of each address slot, zero on padding.
        elapsed_times: Evolution time of each address slot, T for the initial
            state and T - r_q for Duhamel node r_q.
        branch_to_node: Physical branch index of each address slot, or None for padding.
        branch_kinds: "initial" or "source" for each physical branch, which
            selects the input direction that branch prepares.
        applications: ``LCHSApplication`` of each retained physical
            application.
        application_slots: Address slots of each retained application.
        source_nodes: Gauss-Legendre Duhamel times r_q on [0, T].
        source_weights: Matching Gauss-Legendre weights w_q.
        initial_norm: Physical norm of the initial vector, as a PhysicalScale.
        source_norm: Physical norm of the constant source, as a PhysicalScale.
        kind_control: Address bit that selects initial or source preparation
            when both are present in a padded layout, otherwise None.
        compact: Whether the address slots list physical branches consecutively without
            power-of-two field padding (branch-controlled SELECT).
        scale_exponent: The integer exponent E - k of the common binary
            scale: E is the largest application mass exponent minus one and
            k the exact lift of the kernel magnitudes.
        dropped_bound: Upper bound, as a PhysicalScale, on the physical
            contribution of slots whose PREP amplitude is not representable
            in binary64 and that were omitted under ``drop_allowance``, or
            None when none was omitted.
    """

    amplitudes: object
    phases: object
    coefficient_l1_norm: float
    k_values: object
    elapsed_times: object
    branch_to_node: tuple[int | None, ...]
    branch_kinds: tuple[str, ...]
    applications: tuple[LCHSApplication, ...]
    application_slots: tuple[tuple[int, ...], ...]
    source_nodes: tuple[float, ...]
    source_weights: tuple[float, ...]
    initial_norm: object
    source_norm: object
    kind_control: int | None
    compact: bool
    scale_exponent: int = 0
    dropped_bound: object = None

    @property
    def branch_count(self):
        return len(self.branch_kinds)


def omission_upper_float(bound):
    """Return a binary64 upper bound on a positive PhysicalScale bound, or None when it overflows.

    A bound below the binary64 range is replaced by the smallest positive
    subnormal 2**-1074, which is still an upper bound, so an omitted
    contribution never becomes zero or unknown merely by being tiny.
    """
    value = bound.as_float()
    if value is not None:
        return value
    return 5e-324 if bound.exponent < 0 else None


def select_source_branches(*, coefficients, k_nodes, initial_scale, source_scale, final_time,
                           shift, duhamel_nodes, scalar_phase=(0.0, 0.0), compact=False, drop_allowance=None,
                           max_bytes=None, max_quadrature_work=None):
    """Reuse one Duhamel quadrature and distinguish physical branches from padding.

    The constant-source solution is exp(-A*T)u0 + integral over s of
    exp(-A*(T-s))b (ACL arXiv:2312.03916v2, Eq. (2)). Discretizing k as in
    Eq. (61) and s with one Gauss-Legendre panel of duhamel_nodes points
    gives Eq. (72) with
    T/h2 = 1 and coefficients c'_q = w_q*||b||. Every k-node is repeated for
    the initial application and for each Duhamel node. The node count is the
    caller's choice rather than the Lemma 13 prescription, and its remainder
    is bounded separately (inhomogeneous_theory).

    Each application a carries exp(s*t_a) for the signed shift s and its own
    elapsed time t_a, never one common exp(s*T), which would change the
    solution. Its mass m_a comes from ``application_facts`` as a
    PhysicalScale. The input directions keep their complex phases, so the
    relative phase of the initial and source contributions is kept.
    ``scalar_phase = (a, b)`` multiplies slot (a, j) by
    exp(-i*t_a*(k_j*a + b)), the scalar a compact recipe takes out of its
    centered children (``selection.LCHSGenerator`` derives the
    factorization).

    Scale and amplitudes. The masses m_a = mu_a*2**e_a are the
    PhysicalScale values of ``application_facts``, never recomputed. The
    common scale is 2**E with E = max_a e_a - 1, an exact integer, and the
    relative masses r_a = m_a/2**E = ldexp(mu_a, e_a - E) lie in (0, 2),
    the largest in [1, 2), so their sum Z = sum_a r_a lies in [1, 2*A) for
    A applications and cannot overflow. The kernel magnitudes are
    K_j = 2**k*|c_j| with the exact lift k of ``providers.coefficient_lift``,
    so a subnormal raw coefficient keeps its magnitude: |(1+i)*2**-1074| is
    sqrt(2)*2**-1074, while abs rounds it to 2**-1074. The PREP state
    factorizes over applications and kernel nodes:

        |p> = sum_a sqrt(r_a/Z)|a> (x) sum_j sqrt(K_j/beta)|j>,
        beta = sum_j K_j, gamma = 2**(E - k)*Z*beta,

    so sum over slots of p_aj**2 = 1. With the integer split
    e_a - E = 2*h_a + d_a, d_a in {0, 1}, each amplitude is
    p_aj = ldexp(sqrt(ldexp(mu_a, d_a)/Z)*sqrt(K_j)/sqrt(beta), h_a):
    the square-root argument lies in [1/(4A), 2), so no probability is
    formed or squared, and the one ldexp rounds only when the amplitude is
    subnormal, which is kept as a positive value. The amplitude exponents
    use E, and the layout records scale_exponent = E - k, so the lift is
    removed once, in gamma.

    Phases. Slot (a, j) has phase arg(c_j) + phi_aj modulo 2*pi. The raw
    argument atan2(Im c_j, Re c_j) is computed from the stored parts, and
    phi_aj enters as the angle of the unit number that extracted_phase
    returns, so both lie in [-pi, pi] and their sum is reduced with
    ``math.remainder``. Neither the magnitude nor the product
    c_j*exp(i*phi_aj) enters, because rotating a subnormal part rounds it
    by up to order one: the registered grid with four c_j = 2**-1074 and
    phi = -.6 gives the angle -pi/4 that way. Adding arg(c_j) to an
    unreduced large phase argument would lose it at that argument's
    spacing, 256 radians at 2**60. Square roots, quotients, the binary64
    sums and the phase arithmetic add relative errors of a few units in
    the last place, a numerical estimate without an enclosure.

    Omission. A slot whose amplitude is not representable even as a
    subnormal, although c_j != 0, can be omitted only within
    ``drop_allowance``, an absolute physical L2 allowance. Its contribution
    is at most m_a*|c_j| = m_a*K_j*2**-k, because each branch
    exp(-i t (k L_s + H)) is unitary. These bounds are added per
    application: with K_omit,a the fsum of K_j over the omitted nodes of
    application a, its omitted slots contribute at most
    m_a*2**-k*K_omit,a, one PhysicalScale, and ``dropped_bound`` is the
    ``scaled_sum`` of these at most A scales, the sum over omitted slots
    of m_a*|c_j|. Each nonzero K_j is a normal binary64 number and the
    admitted table sum is finite, so no omitted magnitude underflows in
    this regrouping. The sums are binary64 evaluations with ordinary
    rounding, not outward enclosures. Without an allowance that covers
    the summed bound, planning raises ApplicabilityError. An
    application with every slot omitted is removed from the layout. Each
    omitted amplitude is below 2**-1074, so its share of the normalization
    is below 2**-2148 and leaves the binary64 amplitudes and 1-norm of the
    retained slots unchanged. Exact zero kernel coefficients and zero
    inputs are structural zeros, never omissions.

    In the padded layout the low address bits index the k-node, the next bits
    the Duhamel node, and one top bit (kind_control) chooses the initial or
    source input when both are present. Each field is padded to a power of
    two, so the input choice is one address bit and one controlled
    preparation per input kind serves every branch. The compact layout, used
    with branch-controlled SELECT, instead prepares the input inside each
    branch.
    """
    from math import atan2, remainder, tau
    from .inhomogeneous_theory import duhamel_quadrature
    from .providers import lifted_magnitudes
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    has_initial, has_source = initial_scale.mantissa > 0, source_scale.mantissa > 0
    if not has_initial and not has_source:
        raise ValueError("constant-source circuit requires a nonzero initial or source input")
    m, s = len(k_nodes), duhamel_nodes if has_source else 0
    from nwqlib.operators.access import _check_bytes

    def slot_count(initial, sources):
        node_bits, source_bits = max(m-1, 0).bit_length(), max(sources-1, 0).bit_length()
        return m*(int(initial)+sources) if compact else 1 << (node_bits+source_bits+int(initial and sources > 0))

    # Bytes of this stage, a coarse estimate of its dominant arrays checked
    # before the Gauss rule. With A = int(has_initial) + s applications, M = m
    # kernel nodes, N = A*M physical slots, S the padded slot count and P the
    # power of two at least S (omission only removes slots): 128 bytes per
    # Duhamel node for the rule, 96 per kernel node for its magnitudes, roots
    # and angles, 128 per physical slot for the amplitude, phase, kind and
    # slot rows and their kept copies, and 16*P + 64*S for the four layout
    # arrays, their byte copies and the branch map. Input arrays that the
    # caller holds are outside it.
    # Work: s**2 for the tridiagonal rule, 8*s for mapping the nodes, and 12 per
    # slot, an engineering allowance. The omission pass below visits each
    # physical slot at most six times and composes at most A scales.
    applications_max, size = int(has_initial)+s, slot_count(has_initial, s)
    slots = applications_max*m
    _check_bytes(128*s+96*m+16*(1 << max(size-1, 0).bit_length())+64*size+128*slots, max_bytes,
                 "constant-source quadrature and branch grid")
    work = s**2+8*s+12*size
    cap = optional_limit(max_quadrature_work, "max_quadrature_work")
    if exceeds(work, cap):
        raise ResourceLimitError(
            f"constant-source branch selection exceeds max_quadrature_work: it needs {work} work units "
            f"(s**2+8*s+12*slots with s={s}, slots={size}), LCHS.max_quadrature_work={cap}. "
            f"Raise LCHS.max_quadrature_work to at least {work}.")
    nodes, weights = duhamel_quadrature(final_time, s) if has_source else ((), ())
    dissipative, hamiltonian = scalar_phase
    applications = list(application_facts(kind="discrete_source", final_time=final_time, signed_shift=shift,
        scalars=(dissipative-shift, hamiltonian), initial_norm=initial_scale, source_norm=source_scale,
        source_nodes=nodes, source_weights=weights).applications)
    magnitudes, lift = lifted_magnitudes(coefficients)
    angles = np.angle(np.asarray(coefficients, dtype=np.complex128))
    roots, beta = np.sqrt(magnitudes), fsum(magnitudes)
    common = max(item.mass.exponent for item in applications) - 1

    def relative(item):
        # Exact integer exponent difference. A mass far below the largest
        # gives a zero amplitude row through ldexp, and its nodes are omitted.
        return item.mass.exponent - common

    def total(items):
        return fsum(ldexp(item.mass.mantissa, relative(item)) for item in items)

    def joined(angle, k, time):
        unit = extracted_phase(time, k, scalar_phase)
        return remainder(angle + atan2(unit.imag, unit.real), tau)

    z = total(applications)
    amplitude_rows, phase_rows = np.zeros((len(applications), m)), np.zeros((len(applications), m))
    for index, item in enumerate(applications):
        half, odd = divmod(relative(item), 2)
        row = amplitude_rows[index]
        # np.ldexp takes a C int exponent, and half can be any integer. The
        # scaled roots x are below 2 (ldexp(mu_a, d_a) < 2 <= 2*Z, K_j <= beta),
        # so for half <= -1076 every x*2**half is below 2**-1075, half the
        # smallest subnormal 2**-1074, and rounds to exactly zero. Saturating
        # half at -1076 therefore changes no row, and half >= -1075 is used
        # as is.
        row[:] = np.ldexp(sqrt(ldexp(item.mass.mantissa, odd)/z)/sqrt(beta)*roots, max(half, -1076))
        if dissipative or hamiltonian:
            phase_rows[index] = [joined(angle, k, item.elapsed_time) if amplitude > 0 else 0.0
                                 for angle, k, amplitude in zip(angles, k_nodes, row, strict=True)]
        else:
            phase_rows[index] = np.where(row > 0, angles, 0.0)
    # One omission scale per application, from the row's omitted nodes.
    positive, unlift = magnitudes > 0, PhysicalScale(mantissa=0.5, exponent=1-lift)
    omitted, bounds = 0, []
    for item, row in zip(applications, amplitude_rows, strict=True):
        mask = (row == 0) & positive
        count = int(np.count_nonzero(mask))
        if count:
            omitted += count
            bounds.append(compose_recovery(item.mass, fsum(magnitudes[mask]), unlift))
    dropped_bound = None
    if omitted:
        dropped_bound = scaled_sum(bounds)
        upper = omission_upper_float(dropped_bound)
        if drop_allowance is None or upper is None or upper > drop_allowance:
            raise ApplicabilityError(
                f"{omitted} constant-source address slot(s) have PREP amplitudes below the binary64 range; "
                f"set source_drop_allowance to at least their physical bound {dropped_bound.mantissa}*2**"
                f"{dropped_bound.exponent} to omit them")
        keep = np.flatnonzero(amplitude_rows.any(axis=1))
        applications = [applications[index] for index in keep]
        amplitude_rows, phase_rows = amplitude_rows[keep], phase_rows[keep]
        has_initial = any(item.kind == "initial" for item in applications)
        has_source = any(item.kind == "source" for item in applications)
    retained_sources = sum(1 for item in applications if item.kind == "source")
    node_bits, source_bits = max(m-1, 0).bit_length(), max(retained_sources-1, 0).bit_length()
    size = slot_count(has_initial, retained_sources)
    amplitudes, phases = np.zeros(1 << max(size-1, 0).bit_length()), np.zeros(size)
    selected_k, times = np.zeros(size), np.zeros(size)
    branch_to_node = [None]*size
    kinds, slots_by_application = [], []
    kind_control = node_bits+source_bits if has_initial and has_source and not compact else None
    next_slot, source_index = 0, 0
    for row, item in enumerate(applications):
        if compact:
            base = next_slot
        elif item.kind == "initial":
            base = 0
        else:
            base = (0 if kind_control is None else 1 << kind_control) | (source_index << node_bits)
            source_index += 1
        slots = tuple(range(base, base+m))
        amplitudes[base:base+m], phases[base:base+m] = amplitude_rows[row], phase_rows[row]
        selected_k[base:base+m], times[base:base+m] = k_nodes, item.elapsed_time
        for slot in slots:
            branch_to_node[slot] = len(kinds)
            kinds.append(item.kind)
        slots_by_application.append(slots)
        next_slot += m
    return SourceBranchLayout(amplitudes=_freeze_array(amplitudes), phases=_freeze_array(phases),
        coefficient_l1_norm=total(applications)*beta, k_values=_freeze_array(selected_k), elapsed_times=_freeze_array(times),
        branch_to_node=tuple(branch_to_node), branch_kinds=tuple(kinds), applications=tuple(applications),
        application_slots=tuple(slots_by_application), source_nodes=tuple(map(float, nodes)),
        source_weights=tuple(map(float, weights)), initial_norm=initial_scale, source_norm=source_scale,
        kind_control=kind_control, compact=compact, scale_exponent=common-lift, dropped_bound=dropped_bound)
