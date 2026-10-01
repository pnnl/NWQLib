"""The GCiM weighted-pencil pair reducer, registered as ``block_matrix_elements``.

The reducer is registered when this module is imported, and
``core.planning.BUILTIN_REDUCER_MODULES`` lists this module so that the name
resolves in a process that has not imported it.

Exact pair acquisition publishes the offset-free Hamiltonian entry and overlap
entry. An off-diagonal pair uses the two branches of one phase-faithful joint
preparation. The reducer applies the selected H0 once to the right branch with
the grouped Pauli kernel, then contracts with the left branch. It computes
overlap directly from the two branches. The Hamiltonian entry's error combines
preparation error, grouped-action rounding and complex-dot rounding. The overlap
entry has only preparation and dot error. A diagonal pair uses one system
preparation and returns its raw squared norm as the acquired overlap. Pencil
assembly uses unit diagonal overlap under the normalized-basis convention,
while the raw acquired scalars and selected padded coordinates remain
associated with the acquisition.

Joint state (Zheng et al., Phys. Rev. Research 5, 023200 (2023), Eqs. (14)-(15),
with a phase-faithful controlled preparation): starting from ``|0>|0>``, H on
the ancilla, X on the ancilla, controlled ``U_i``, X on the ancilla and
controlled ``U_j`` give ``|chi_ij> = (|0>|phi_i> + |1>|phi_j>)/sqrt(2)``. The
ancilla is native bit 0, so the two branches are the strided views
``joint[0::2]`` and ``joint[1::2]``, and ``2*vdot(x_left, P x_right) =
<phi_i|P|phi_j>`` with the actual preparation phases: for
``U_k = exp(i theta_k) V_k`` the target is
``exp(i(theta_j - theta_i)) <0|V_i^dagger P V_j|0>``. The saved state's one
common global phase cancels in both returned values.

Declaration: the reduction parameters name the layout (ancilla at native bit 0
with strided branches, or the diagonal system state with the ancilla in
``|0>``, or the system register alone for a shared full-chain query), the
pair ``(left, right)`` in the requested public orientation, the
system width, the content identity of the stored Pauli table and its term and
flip-group counts, and the reduction version. The output is two complex128
scalars ``(H0_ij, S_ij)`` for every pair, diagonal included. Reanalysis
consumes these scalars without native execution.
"""

from __future__ import annotations

import math
import weakref
from fractions import Fraction as F

import numpy as np

from nwqlib.core.planning import READOUT_REDUCERS, Reducer, ReducerOutput

REDUCER = "block_matrix_elements"
VERSION = 1
LAYOUT_OFF_DIAGONAL = "ancilla bit 0, strided branches"
LAYOUT_DIAGONAL = "system state, ancilla |0> at bit 0"
# A shared full-chain query saves the system register alone.
LAYOUT_SYSTEM = "system state, no ancilla"

# Coordinate tile of the Pauli action scratch allowance, the shared kernel's
# tile length. An engineering choice, not a scientific threshold.
TILE = 1024

# Stored Pauli tables of bound Plans, by admitted content identity. Every
# Plan bound to one operator holds the same table object, and the registry
# forgets it with the last such Plan.
_TABLES = weakref.WeakValueDictionary()


class PairTable:
    """The packed nonidentity Pauli table a bound Plan's pair reductions act with."""

    def __init__(self, identity, table):
        self.identity = identity
        self.table = table


def bind_table(identity, table):
    """Return the registered table of ``identity``, registering ``table()`` when no live Plan holds one.

    The identity is the admitted content identity of the operator, so an
    equal identity names equal table contents, and later Plans share the
    live holder instead of replacing it.
    """
    holder = _TABLES.get(identity)
    if holder is None:
        holder = PairTable(identity, table())
        _TABLES[identity] = holder
    return holder


def pair_parameters(*, left, right, qubits, identity, terms, flips, c1, ancilla=True):
    """Return the reduction parameters of one pair acquisition.

    ``terms`` and ``flips`` are L and the flip-group upper bound
    ``g = min(L, N)``, and ``c1`` is the outward coefficient mass
    ``sum |c_t|`` as a hexadecimal float, the input C of the entry bounds.
    """
    return {
        "c1": c1,
        "flips": flips,
        "layout": (LAYOUT_OFF_DIAGONAL if left != right else LAYOUT_DIAGONAL if ancilla else LAYOUT_SYSTEM),
        "left": left,
        "qubits": qubits,
        "right": right,
        "table": identity,
        "terms": terms,
        "version": VERSION,
    }


def pair_work(parameters, width=None):
    """Return the host work of one pair reduction.

    In the kernel/pass convention, the attempted group pass plus ordered
    retry and two dots cost ``W_pair = (L+F) + N(L+F+1) + N(1+L) + 2N``, and
    Pauli group construction before the first coordinate tile adds its 3L
    AND, popcount and rotation visits. For L = 0 the action and the H0 dot
    are skipped, giving one N-entry overlap reduction. F is the flip-group
    upper bound ``g = min(L, N)``, valid without inspecting the masks.
    """
    n = 1 << parameters["qubits"]
    terms, flips = parameters["terms"], parameters["flips"]
    if not terms:
        return n
    return (terms + flips) + 3 * terms + n * (terms + flips + 1) + n * (1 + terms) + 2 * n


def pair_bytes(parameters):
    """Return the private reduction workspace ``B_pair`` beyond the saved state, admitted before acquisition.

    With ``beta = 1`` on the diagonal and 2 off it, ``N = 2**n``, F the
    flip-group bound and ``t = min(N, tile)``, packed group metadata is
    ``B_groups = 24L + 16F + 8`` and a sufficient numerical reducer byte law
    is ``B_pair = 16*beta*N + 17N + 96t + B_groups + B_wrapper + B_operator +
    H0``. The 17N covers one action output and a finite mask. Group
    construction's 9L frontier shares a maximum with the 96t tile scratch.
    The strided branches are views, and ``np.vdot`` reduces them without a
    contiguous copy. The saved joint input
    ``16*beta*N`` is the simulator's saved state, admitted by the trajectory
    memory check, and ``B_operator`` is the Plan's resident table, so neither
    is counted again here. ``H0 = 65536`` is the fixed heap allowance.
    """
    n = 1 << parameters["qubits"]
    terms, flips = parameters["terms"], parameters["flips"]
    t = min(n, TILE)
    if not terms:
        action = 0
    else:
        action = 17 * n + max(9 * terms, 96 * t) + 24 * terms + 16 * flips + 8
    return action + 65536


def _pair_execute(state, parameters, bindings):
    """Return ``(H0_ij, S_ij)`` as one complex128 array of shape (2,).

    Inputs, coefficient domain, resource admission and fallback finiteness
    checks belong to the kernel's integration contract. It returns the two raw
    acquired scalars, with diagonal normalization applied only by the pencil
    assembler. The action output is complex128, as the binary64 entry bound
    requires.
    """
    from nwqlib.operators._pauli import apply_terms

    q = parameters["qubits"]
    n = 1 << q
    layout = parameters["layout"]
    diagonal = layout != LAYOUT_OFF_DIAGONAL
    width = n if layout == LAYOUT_SYSTEM else 2 * n
    beta = 1 if diagonal else 2
    if state.dtype != np.complex128 or state.shape != (width,):
        raise ValueError("joint state must match the selected pair layout")
    left = state if layout == LAYOUT_SYSTEM else state[0::2]
    right = left if diagonal else state[1::2]
    overlap = beta * np.vdot(left, right)
    h0 = 0j
    if parameters["terms"]:
        holder = _TABLES.get(parameters["table"])
        if holder is None:
            raise ValueError("the pair reduction's Pauli table is not bound to a live Plan")
        action = apply_terms(holder.table, right, num_qubits=q, out=np.empty(n, dtype=np.complex128))
        h0 = beta * np.vdot(left, action)
    if not (np.isfinite(h0) and np.isfinite(overlap)):
        raise ValueError("pair contraction requires finite results")
    return (np.array([h0, overlap], dtype=np.complex128),)


READOUT_REDUCERS[REDUCER] = Reducer(
    shape=lambda parameters: (ReducerOutput("complex128", (2,)),),
    work=pair_work,
    execute=_pair_execute,
    # The common saved-state phase cancels in both points.
    phase_invariant=True,
)


U = F(1, 2**53)
ETA = F(1, 2**1074)


def sqrt_up(value, bits=128):
    value = F(value)
    if value < 0:
        raise ValueError("a squared norm must be nonnegative")
    scale = 1 << bits
    n = value.numerator * scale * scale
    d = value.denominator
    root = math.isqrt(n // d)
    if root * root * d < n:
        root += 1
    return F(root, scale)


def upward(value):
    value = F(value)
    try:
        result = float(value)
    except OverflowError:
        return math.inf
    if math.isfinite(result) and F(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def gamma(count):
    if type(count) is not int or count < 0 or count*U >= 1:
        raise ValueError("invalid gamma domain")
    return count*U/(1-count*U)


def dot_underflow(n):
    if (2*n+4)*U >= 1:
        raise ValueError("invalid underflow domain")
    return 8*n*ETA/(1-(2*n+4)*U)


def action_constants(n, terms, groups, largest, *, ordered=False):
    if not terms:
        return F(0), F(0)
    if not (1 <= groups <= terms and 1 <= largest <= terms):
        raise ValueError("invalid action census")
    mu = sqrt_up(2)*gamma(2)
    if ordered:
        a = gamma(terms-1)
        error = mu+a+mu*a
        r = 2*terms+4
        numerator = 8*terms*ETA*sqrt_up(n)
    else:
        a, b = gamma(largest-1), gamma(groups-1)
        error = a+mu+b+a*mu+a*b+mu*b+a*mu*b
        r = terms+groups+4
        numerator = 4*(terms+groups)*ETA*sqrt_up(n)
    if r*U >= 1:
        raise ValueError("invalid action underflow domain")
    return error, numerator/(1-r*U)


def entry_bounds(delta, c1, n, terms, groups, largest, *, diagonal=False,
                 ordered=False):
    """Return outward ``(E_S, E_H)`` of one pair entry, or None when ``delta`` is unavailable.

    With ``D(delta) = 2*delta + delta**2``, ``g_N = sqrt(2)*gamma(2N)``,
    ``U_dot = 8N*eta/(1 - (2N+4)u)``, ``beta`` 1 on the diagonal and 2 off it,
    and ``(e, U_A)`` the successful action path's constants,

        E_S = D(delta) + g_N(1+delta)**2 + beta*U_dot,
        E_H = C{D(delta) + [e + (1+e)g_N](1+delta)**2}
              + (sqrt(beta) + beta*delta)(1+g_N)U_A + beta*U_dot.

    ``E_H = 0`` when L = 0 and H0 is returned algebraically zero. The pair
    reducer uses the ordered-action error envelope for either successful
    Pauli action path. This envelope also bounds grouped action because its
    largest-group and group-count contributions sum to at most the termwise
    reduction length. The allowance includes gradual underflow and both
    complex contractions. An unavailable state receipt leaves the derived
    entry bound unavailable. Callers pass ``ordered=True`` with
    ``groups=1, largest=L``, valid bound inputs whose grouped values are
    unused in the ordered branch. Exact rational arithmetic evaluates the
    expanded products and a dyadic square-root upper bound, and a strictly
    positive rational below half a subnormal publishes as the minimum
    positive subnormal.

    For a preparation synthesized from a supplied matrix, the comparison uses
    the exact-to-rounding construction convention of subroutines._dense_synthesis.
    The native receipt and this host reduction do not supply an independent
    synthesis-residual bound relative to an arbitrary input matrix. A finite
    entry allowance is conditional on that convention and on the producing
    receipt's operation and parameter premises.

    The receipt's first-order model and its exclusions
    still apply: outward evaluation cannot convert a first-order receipt
    premise into an all-orders guarantee.
    """
    if delta is None:
        return None
    delta, c1 = F(delta), F(c1)
    if delta < 0 or c1 < 0 or n < 1:
        raise ValueError("invalid bound inputs")
    e, underflow = action_constants(n, terms, groups, largest, ordered=ordered)
    beta = 1 if diagonal else 2
    dot = sqrt_up(2)*gamma(2*n)
    d = 2*delta+delta*delta
    overlap = d+dot*(1+delta)**2+beta*dot_underflow(n)
    h0 = (c1*(d+(e+(1+e)*dot)*(1+delta)**2)
          +(sqrt_up(beta)+beta*delta)*(1+dot)*underflow
          +beta*dot_underflow(n)) if terms else F(0)
    return upward(overlap), upward(h0)


def pooled_bound(values, bounds, populations):
    """Return the outward bound of a pooled complex estimate of m acquisitions, or None.

    With exact rational weights ``w_a = n_a/sum n_a``,

        E_pool = sum_a w_a E_a
                 + [2u + u**2 + gamma(max(m-1, 2))(1 + 2u + u**2)] sum_a w_a|z_a|
                 + U_dot,m,

    the nearest-rounded-weight, product and componentwise-addition model. A
    single value uses its original bound with no pooling arithmetic. Apply
    it separately to H0 and S. Outward complex magnitudes use
    ``sqrt_up(real**2 + imag**2)``.
    """
    if not (len(values) == len(bounds) == len(populations)) or not values:
        raise ValueError("invalid pool")
    if any(type(n) is not int or n <= 0 for n in populations):
        raise ValueError("pool populations must be positive integers")
    if any(b is None for b in bounds):
        return None
    if len(values) == 1:
        return bounds[0]
    total = sum(populations)
    weights = [F(n, total) for n in populations]
    mags = [sqrt_up(F(z.real)**2+F(z.imag)**2) for z in values]
    g = gamma(max(len(values)-1, 2))
    kappa = 2*U+U*U+g*(1+2*U+U*U)
    result = sum((w*F(b) for w, b in zip(weights, bounds)), F(0))
    result += kappa*sum((w*z for w, z in zip(weights, mags)), F(0))
    return upward(result+dot_underflow(len(values)))
