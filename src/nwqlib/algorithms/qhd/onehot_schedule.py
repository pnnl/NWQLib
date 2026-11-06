"""The blocks of a one-hot QHD schedule, derived step by step from a Plan's tables and step rows.

A one-hot Plan records no block. The blocks of step k follow from the
support tables (``QHDReconstruction.support_values``), the step row
``(t_k, a_k, b_k)`` (``step_weights``), the Method's duration
``dt = fl(total_time/num_steps)``, Trotter order and ``rotation_threshold``,
the grid's links and the projector provider of each support size
(``QHDOneHotSchedule.providers``). ``OneHotSchedule.blocks`` forms one step
at a time with the same binary64 operations, in the same association, as
the circuit applies them, so ``step_blocks``, the emitted gates and every
count and bound that planning records (``OneHotSchedule.summary``) describe
the same numbers. Table entries are read in tiles of at most
``ENTRY_TILE`` entries, one tile live at a time, so the transient storage
of a derivation is that of one tile of the largest table
(``derivation_bytes``).

Order. First order applies ``V(dt)`` and then ``K(dt)``, the operator order
of Eq. (E.3) in Leng et al., arXiv:2303.01471v1. Second order applies
``V(dt/2) K(dt) V(dt/2)``, where ``K(dt)`` is the symmetric parity split of
``hopping_layers``. Under the integrated rule a step's weights are its
whole-step averages ``A/dt`` and ``B/dt`` (``schedules.step_weights``), so
each potential half applies ``B/2`` and the odd, even and odd link layers
apply ``A/2``, ``A`` and ``A/2``. The step is then exactly the symmetric
(Strang) split of the first Magnus exponent ``-i (A K + B V)``, whose
splitting error starts at third order in the exponents. The time-ordering
error of the step (the Magnus terms from ``Omega_2`` on) and the
product-formula error therefore stay separate, as ``method._error_model``
declares them. Potential halves ``B_1`` and ``B_2`` with ``B_1 + B_2 = B``,
such as separate half-step integrals, would add the second-order BCH term
``(A (B_1 - B_2)/2) [V, K]``, of the same order ``dt**3`` as ``Omega_2``, so
such a step would be a different method that mixes the two error sources.

Potential. With ``n = (I - Z)/2`` on each qubit, Wu et al.,
arXiv:2605.12066v1, Sec. IV.A, Eqs. (9)-(10), encode a single-variable
function as ``sum_k f(g_k) n_k``, and the text after their Eq. (10) builds
multivariate functions from products of single-variable operators. NWQLib
extends this to a general support term: ``f_S`` becomes
``sum_g f_S(g) prod_{j in S} n_{j, g_j}``, which equals ``f_S(x_S)`` on every
state with one excitation per register, because exactly one product is
nonzero there. This realizes the diagonal finite-grid potential of Eqs.
(F.9) and (F.15) of Leng et al. in the one-hot register, not their
Hamming-weight encoding of Eq. (F.36). Each table entry v of a support of
size s gives, for duration t and step weight b, the intended exponent
``Z = t b v`` of the projector ``prod n``. A zero value or weight adds no
gate. Otherwise the construction forms ``p = fl(b v)``, ``zeta = fl(t p)``,
the identity coefficient ``q = p/2**s`` and its phase ``fl(-t q)``, and
needs ``|zeta| 2**-s >= 2**-1022`` for the projector's power-of-two
parameter scalings and, with the phase-diagonal provider, the same for its
nonzero wrapped phase (``pauli_evolution.wrapped_projector_phase``). When
any of these would be nonzero and below ``2**-1022``, the whole occurrence
is omitted, the projector and its identity event together. Overflow and
nonfinite results raise. The projector splits as
``P = (P - 2**-s I) + 2**-s I`` with ``||P - 2**-s I|| = 1 - 2**-s``, so an
omitted occurrence costs ``(1 - 2**-s) |Z|`` in its traceless part, counted
with the pruned blocks, and ``2**-s |Z|`` in its identity phase, counted
in the phase total, together ``|Z|``. A kept occurrence whose
``|zeta|`` is below ``rotation_threshold`` adds no gate and is counted
as pruned. Most entries pass every range check, so they are formed and
decided as arrays. An entry that is not strictly inside the normal range
at any product, or whose angle or wrapped phase lies below ``2**-900``,
takes the scalar checks of ``_kept_products`` instead, which decide it
with exact arithmetic where a product reaches the edge of the range.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from .validation import (
    _NORMAL_MAX,
    _NORMAL_MIN,
    _kinetic_range_error,
    _normal_product,
    _normal_quotient,
    _normal_range,
    _underflows,
    angle_units,
)

# Table entries read together by one vector step. Every transient array of a
# step is at most this long.
ENTRY_TILE = 1 << 16
# Python scalars formed from one array at a time, so no list grows with a tile.
_CHUNK = 256
# The derivation's storage (derivation_bytes): per entry of the largest tile
# and per link occurrence of one step, plus a fixed part for Python objects,
# the chunk lists and NumPy bookkeeping. Engineering allowances above the
# tracemalloc peaks of blocks() and summary(), at most 60 bytes per entry of a
# full tile, 44 per link and 122,001 bytes in all for 1,296 entries, on
# CPython 3.12.14 and NumPy 2.5.2. Registered in docs/ENGINEERING_CONSTANTS.md,
# not a bound on process RSS.
_TILE_ENTRY_BYTES = 96
_LINK_BYTES = 64
_DERIVATION_FIXED_BYTES = 1 << 17

# Below this magnitude an angle or wrapped phase takes the scalar checks,
# which compare with the normal range exactly. Larger values are at least
# 2**122 times every range threshold 2**(s - 1022) of a support of fewer
# than 100 variables, so their vector decision needs no exact comparison.
_SUSPECT = 2.0**-900


class Block(NamedTuple):
    """One emitted block of a one-hot step, with the fields of ``records.QHDBlock``."""

    kind: str
    time_step: float
    time: float
    support: tuple[int, ...]
    coefficient: float
    angle: float
    provider: str
    cx: int


class HoppingLayer(NamedTuple):
    """One link layer of one variable in one step, with the hopping angle of each of its links."""

    variable: int
    duration: float
    coefficient: float
    angle: float
    links: tuple[tuple[int, int], ...]
    kept: bool
    pruned: bool


def hopping_layers(grid, *, dt, kinetic_weight, trotter_order, threshold):
    """Return the hopping layers of one step, variable by variable, each in circuit order.

    The restricted matrix is the finite-difference kinetic operator for
    ``-Delta / 2`` in Eq. (1) of Leng et al., arXiv:2303.01471v1, with the
    grid's boundary condition. Per variable the operator is ``-(1/2) L``
    with ``L = (A - 2I)/h**2`` as in their Remark 6, where A is the adjacency
    matrix of the grid's links (``OneHotGrid.links``). On the Dirichlet
    grids the links form a chain, the stencil of their Eqs. (F.7)/(F.9). On
    the periodic grid they form a cycle, the circulant matrix whose hopping
    sum including the wrap link is Eq. (12) of Liu et al.,
    arXiv:2607.16996v1, Sec. III C. On the one-hot subspace
    ``(X_i X_j + Y_i Y_j)/2`` acts as ``|e_i><e_j| + h.c.``, a direct
    two-qubit calculation, so each link carries the coefficient
    ``c_h = -kinetic_weight/(4 h**2)`` on XX and on YY, and the stored angle
    ``fl(fl(2 t) c_h)`` equals the duration times the stencil off-diagonal.
    The hopping conserves the excitation number of each register. The
    stencil diagonal ``kinetic_weight/h**2`` is the same on every one-hot
    state, a global phase that the phase total records (``_step_phases``).

    Links that share a point do not commute. First order applies the links
    in increasing link order for the full dt, the wrap link last. Second
    order applies the odd-indexed links for dt/2, the even-indexed links for
    dt and the odd-indexed links for dt/2. Links of one parity are disjoint
    on the chain and, for even K, on the cycle, so each layer is exact and
    the split symmetric. The QHD Method therefore requires an even K on the
    periodic grid and excludes K = 2 there (``method.QHD._periodic_grid``).

    Range checks. The coefficient ``c_h = fl(-a/fl(fl(4 h) h))``, the
    doubled duration ``fl(2 t)``, the angle and the circuit's gate parameter
    ``2 angle`` must be finite, and the coefficient and angle normal. The
    duration is positive, so the angle is zero only for a zero weight, which
    adds no block. A refusal names its remedy (``validation._kinetic_range_error``,
    or a shorter ``total_time`` for an overflowing ``fl(2 t)``). The circuit
    doubles the angle to form its gate parameter
    (``pauli_evolution._hopping_parameter``), so half of a finite gate
    parameter equals the angle. ``kept`` marks a nonzero angle at least
    ``threshold``, and ``pruned`` a nonzero angle below it.
    """
    links = grid.links()
    segments = (((links[1::2], dt / 2), (links[0::2], dt), (links[1::2], dt / 2)) if trotter_order == 2
                else ((links, dt),))
    layers = []
    for j in range(grid.num_variables):
        spacing = grid.spacing(j)
        try:
            # c_h = -a/D with D = fl(fl(4 h) h), which the grid keeps finite and normal (OneHotGrid).
            coefficient = _normal_quotient(-kinetic_weight, 4.0 * spacing * spacing,
                                           f"the hopping coefficient -a/(4 h**2) of variable {j}")
        except ValueError as error:
            raise _kinetic_range_error(error) from error
        for layer, duration in segments:
            try:
                doubled = _normal_product(2.0, duration, "twice the hopping duration 2 t")
            except ValueError as error:
                raise ValueError(f"{error}. Choose a shorter total_time") from error
            try:
                angle = _normal_product(doubled, coefficient, f"the hopping angle 2 t c of variable {j}")
                _normal_product(2.0, angle, f"the hopping gate parameter 4 t c of variable {j}")
            except ValueError as error:
                raise _kinetic_range_error(error) from error
            nonzero = angle != 0.0
            layers.append(HoppingLayer(j, float(duration), coefficient, angle, tuple(layer),
                                       nonzero and abs(angle) >= threshold, nonzero and abs(angle) < threshold))
    return layers


class _Table(NamedTuple):
    position: int
    support: tuple[int, ...]
    values: np.ndarray
    size: int
    provider: str
    cx: int
    diagonal: bool


class _Tile(NamedTuple):
    """The decisions and emitted blocks of one tile of one table in one potential factor.

    ``emitted`` marks the entries that give a block, ``positions`` are their
    C-order table positions, and ``phases`` holds the identity event of every
    entry. ``omitted`` counts the range omissions and ``charge`` is their
    exact ``sum |t b v|``.
    """

    table: _Table
    start: int
    values: np.ndarray
    emitted: np.ndarray
    positions: np.ndarray
    coefficients: np.ndarray
    angles: np.ndarray
    phases: np.ndarray
    pruned: np.ndarray
    omitted: int
    charge: Fraction


def _kept_products(value, duration, weight, scale, s, diagonal, key):
    """Return ``(p, zeta, q)`` of one occurrence, or None when a product would fall below the normal range.

    The scalar decision for an entry that the vector step cannot decide.
    Four results strictly inside the normal range pass every ``_underflows``
    check, so only an occurrence with a result at or beyond either end takes
    those checks, in their order, which raise or omit. ``key`` is the
    occurrence's ``(support, grid_indices)``, named in an error.
    """
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import wrapped_projector_phase

    coefficient = weight * value
    angle = duration * coefficient
    identity = coefficient * scale
    phase = duration * identity
    if not (_NORMAL_MIN < abs(coefficient) < _NORMAL_MAX and _NORMAL_MIN < abs(angle) < _NORMAL_MAX
            and _NORMAL_MIN < abs(identity) < _NORMAL_MAX and _NORMAL_MIN < abs(phase) < _NORMAL_MAX):
        where = f"support {key[0]} at grid indices {key[1]}"
        if _underflows(coefficient, f"the potential coefficient b v of {where}",
                       lambda: Fraction(weight) * Fraction(value)):
            return None
        if _underflows(angle, f"the projector angle t b v of {where}",
                       lambda: Fraction(duration) * Fraction(coefficient)):
            return None
        if _underflows(identity, f"the projector identity coefficient b v/2**s of {where}",
                       lambda: Fraction(coefficient) * Fraction(scale)):
            return None
        if _underflows(-duration * identity, f"the projector identity phase of {where}",
                       lambda: Fraction(duration) * Fraction(identity)):
            return None
    # |zeta| 2**-s >= 2**-1022, compared through the exponent of zeta.
    if math.frexp(abs(angle))[1] - 1 < s - 1022:
        return None
    if diagonal:
        wrapped = wrapped_projector_phase(angle)
        if not math.isfinite(wrapped):
            raise ValueError(f"the wrapped projector phase of support {key[0]} at grid indices {key[1]} is "
                             f"{wrapped!r}, not finite")
        if wrapped != 0.0 and math.frexp(wrapped)[1] - 1 < s - 1022:
            return None
    return coefficient, angle, identity


def _inside(x):
    return (np.abs(x) > _NORMAL_MIN) & (np.abs(x) < _NORMAL_MAX)


class _Neumaier:
    """Neumaier's compensated running sums of the phase total, one per name.

    For ``t = fl(s + x)`` the residual is ``fl(fl(s - t) + x)`` when
    ``|s| >= |x|`` and ``fl(fl(x - t) + s)`` otherwise, which is exactly
    ``s + x - t`` without overflow. The correction adds the residual, and the
    total is ``fl(t + c)``. An overflow raises, since cancellation later
    could not make an overflowed partial sum valid. The stored sum and
    correction and the added value are finite, so a nonfinite t, residual or
    correction makes ``fl(t + c)`` nonfinite, and checking that total alone
    is the same test.

    Rounding. Assume round-to-nearest binary64 arithmetic, no overflow and
    no inexact underflow, which planning ensures for every event
    (``validation._normal_range``). The residual formed with the
    larger-magnitude operand first is then exactly the rounding error of
    each addition. For m >= 1 added values with ``(m - 1) u < 1``,
    u = 2**-53, let ``X = sum_j x_j``, ``A = sum_j |x_j|`` and
    ``g_m = gamma_(m-1)``, ``gamma_r = r u/(1 - r u)``. The absolute
    residuals total at most ``g_m A``, summing them with ordinary additions
    adds at most ``g_m**2 A``, and the final addition adds at most ``u |X|``
    plus u times the correction's error, so the total differs from X by at
    most ``u |X| + (1 + u) g_m**2 A <= beta_m A`` with
    ``beta_m = u + (1 + u) gamma_(m-1)**2``. The algorithm is Neumaier's
    (Z. Angew. Math. Mech. 54 (1974) 39-51, doi:10.1002/zamm.19740540106).
    Its result equals that of the cascaded summation of Ogita, Rump and
    Oishi (SIAM J. Sci. Comput. 26 (2005) 1955-1988, doi:10.1137/030601818,
    Algorithm 4.1), whose Proposition 4.5 proves the smaller bound
    ``u |X| + gamma_(m-1)**2 A`` for the equivalent Algorithm 4.4. An
    ordinary running sum has the allowance ``gamma_(m-1) A``, first order in
    ``m u``, while beta_m is u plus a term of second order in ``m u``.

    The projector identities enter as one correctly rounded ``math.fsum``
    per tile (``OneHotSchedule.summary``), which errs by at most u times the
    tile's absolute sum, and the tile sums are the added values. With m the
    number of tile sums and single events (``method._phase_contributions``),
    the total therefore differs from the exact sum of the events by at most
    ``u A + beta_m (1 + u) A = beta' A`` with ``beta' = u + (1 + u) beta_m``,
    for A the absolute sum of the events (``method._phase_ledger``).
    """

    def __init__(self):
        self.sums = {}

    def add(self, name, value):
        high, correction = self.sums.get(name, (0.0, 0.0))
        total = high + value
        residual = (high - total) + value if abs(high) >= abs(value) else (value - total) + high
        correction += residual
        result = total + correction
        if not math.isfinite(result):
            raise ValueError(
                "the compiled physical phase of the QHD circuit exceeds the binary64 range"
                + ("" if name == "total" else f" in its {name} part")
                + ". The native circuit and the ir_product reference apply it before any readout"
                + (". The objective constant does not affect the optimization, so removing it from the "
                   "objective removes its part" if name == "objective_constant" else "")
                + ". The classical schrodinger or split_step flavor without keep_state reads the probabilities "
                "without this phase")
        self.sums[name] = (total, correction)
        return result

    def value(self, name):
        high, correction = self.sums.get(name, (0.0, 0.0))
        return high + correction


def _listed(array):
    """Yield the Python scalars of a one-dimensional array, converted ``_CHUNK`` entries at a time."""
    for start in range(0, array.size, _CHUNK):
        yield from array[start:start + _CHUNK].tolist()


def _exact_units(magnitudes):
    """Return the exact sum of nonnegative finite binary64 values as an integer count of ``2**-1074``."""
    values, counts = np.unique(magnitudes, return_counts=True)
    return sum((count * angle_units(value) for value, count in zip(_listed(values), _listed(counts), strict=True)), 0)


class OneHotSchedule:
    """The data that fix a one-hot Plan's blocks, with their per-step derivation and planning summary.

    ``tables`` are the Plan's ``SupportValues`` in their order, ``providers``
    maps each support size to its ``(provider, cx)``, and the remaining
    arguments are the reconstruction's step rows and constant and the
    Method's options.
    """

    def __init__(self, grid, *, tables, providers, step_weights, constant, total_time, num_steps,
                 trotter_order, rotation_threshold):
        self.grid = grid
        self.k = grid.num_grid_points
        self.dt = total_time / num_steps
        self.order = trotter_order
        self.threshold = float(rotation_threshold)
        self.step_weights = tuple(step_weights)
        self.constant = constant
        self.tables = tuple(
            _Table(position, tuple(t.support), np.asarray(t.values.array), len(t.support),
                   *providers[len(t.support)], providers[len(t.support)][0] == "diagonal_synthesis")
            for position, t in enumerate(tables))
        # Wrapped projector phases of the scalar checks, by (angle.hex(), s).

    @classmethod
    def of(cls, reconstruction, method, grid):
        """Return the schedule of a one-hot Plan's reconstruction with compiled steps."""
        record = reconstruction.onehot
        return cls(grid, tables=reconstruction.support_values,
                   providers={p.support_size: (p.provider, p.cx) for p in record.providers},
                   step_weights=reconstruction.step_weights, constant=reconstruction.constant,
                   total_time=method.total_time, num_steps=method.num_steps, trotter_order=method.trotter_order,
                   rotation_threshold=method.rotation_threshold)

    def factors(self):
        """Return the potential durations of one step, one for first order and two equal halves for second."""
        return (self.dt,) if self.order == 1 else (self.dt / 2.0, self.dt / 2.0)

    def _tiles(self, duration, weight):
        """Yield the ``_Tile`` of every table tile of one potential factor, tables in order, entries in C order.

        ``_tile`` forms each tile, so only the yielded arrays outlive it, and
        each consumer releases its tile before asking for the next. One tile
        is then live at a time (``derivation_bytes``).
        """
        for table in self.tables:
            for start in range(0, table.values.size, ENTRY_TILE):
                yield self._tile(table, start, duration, weight)

    def _tile(self, table, start, duration, weight):
        """Return the ``_Tile`` of the entries of ``table`` from ``start`` in one potential factor.

        The vector decision holds p, zeta and the identity events (8 bytes
        per entry each) with two Boolean masks, plus the magnitudes and masks
        of one comparison at a time. The test of the phase-diagonal provider
        adds one complex array and then its argument. The identity event of
        an entry, ``fl(-t q)`` for ``q = fl(p/2**s)``, overwrites q, and an
        entry that the scalar checks keep has the same p, zeta and q, which
        ``_kept_products`` forms with the same operations.
        """
        s, scale = table.size, 1.0 / (2**table.size)
        v = table.values[start:start + ENTRY_TILE]
        nonzero = (v != 0.0) & (weight != 0.0)
        p = weight * v
        kept = nonzero & _inside(p)
        zeta = duration * p
        kept &= _inside(zeta)
        kept &= np.abs(zeta) >= _SUSPECT
        phases = p * scale
        kept &= _inside(phases)
        np.multiply(phases, -duration, out=phases)
        kept &= _inside(phases)
        if table.diagonal and kept.any():
            with np.errstate(all="ignore"):
                wrapped = -1j * zeta
                np.exp(wrapped, out=wrapped)
                wrapped = np.angle(wrapped)
            kept &= np.abs(wrapped, out=wrapped) >= _SUSPECT
            del wrapped
        omitted, charge = 0, Fraction(0)
        for index in _listed(np.flatnonzero(nonzero & ~kept)):
            value = float(v[index])
            key = (table.support, tuple(int(i) for i in np.unravel_index(start + index, (self.k,) * s)))
            if _kept_products(value, duration, weight, scale, s, table.diagonal, key) is None:
                omitted += 1
                charge += abs(Fraction(duration) * Fraction(weight) * Fraction(value))
            else:
                kept[index] = True
        del nonzero
        # The identity events are zero for a zero value or weight and for an omitted occurrence.
        np.copyto(phases, 0.0, where=~kept)
        magnitude = np.abs(zeta)
        emitted = kept & (magnitude >= self.threshold)
        pruned = magnitude[kept & (magnitude < self.threshold)]
        del kept, magnitude
        positions = np.flatnonzero(emitted)
        coefficients = p[positions]
        del p
        angles = zeta[positions]
        del zeta
        positions += start
        return _Tile(table, start, v, emitted, positions, coefficients, angles, phases, pruned, omitted, charge)

    def blocks(self, step):
        """Yield the ``Block`` records of one step in circuit order."""
        time, kinetic_weight, potential_weight = self.step_weights[step]
        factors = self.factors()

        def potential(duration):
            for tile in self._tiles(duration, potential_weight):
                table = tile.table
                shape = (self.k,) * table.size
                for position, coefficient, angle in zip(_listed(tile.positions), _listed(tile.coefficients),
                                                        _listed(tile.angles), strict=True):
                    indices = np.unravel_index(position, shape)
                    support = tuple(j * self.k + int(i) for j, i in zip(table.support, indices, strict=True))
                    yield Block("number_projector", float(duration), time, support, coefficient, angle,
                                table.provider, table.cx)
                del tile

        yield from potential(factors[0])
        for layer in hopping_layers(self.grid, dt=self.dt, kinetic_weight=kinetic_weight, trotter_order=self.order,
                                    threshold=self.threshold):
            if layer.kept:
                for left, right in layer.links:
                    yield Block("kinetic", layer.duration, time,
                                (layer.variable * self.k + left, layer.variable * self.k + right),
                                layer.coefficient, layer.angle, "XXPlusYYGate", 2)
        if len(factors) == 2:
            yield from potential(factors[1])

    def _step_phases(self, step, ledger, totals):
        """Add the kinetic-diagonal and constant phase events of one step and return the omitted constant charge.

        The kinetic diagonal ``sum_j a/h_j**2`` is the same on every one-hot
        state, hence a phase, on both boundaries. Leng et al. drop the
        corresponding constant as a global phase in the sentence that
        introduces Eq. (F.14) of arXiv:2303.01471v1, and Liu et al.,
        arXiv:2607.16996v1, Sec. V A 1, omit it as an overall phase of the
        one-hot sector. Here it enters the phase total, so that the circuit's
        global phase matches the evolution of the finite model. Each
        ``a/h**2`` is a normal binary64 number and their sum is Python's
        compensated ``sum``. The constant's coefficient is ``fl(b c)`` with
        the stored constant c, and when ``fl(b c)`` or its product with dt
        would be nonzero and below ``2**-1022`` the step records a zero event
        and returns the exact ``|dt b c|``, as for an omitted projector.
        """
        _time, kinetic_weight, potential_weight = self.step_weights[step]
        dt = self.dt
        try:
            coefficient = sum(
                _normal_quotient(kinetic_weight, self.grid.spacing(j) ** 2,
                                 f"the kinetic diagonal a/h**2 of variable {j}")
                for j in range(self.grid.num_variables))
            _normal_range(coefficient, "the kinetic diagonal sum_j a/h_j**2")
            _normal_product(-dt, coefficient, "the kinetic_diagonal phase contribution -t q")
        except ValueError as error:
            raise _kinetic_range_error(error) from error
        _record(ledger, totals, "kinetic_diagonal", coefficient, dt)
        charge = Fraction(0)
        if self.constant != 0:
            coefficient = potential_weight * self.constant
            omitted = _underflows(coefficient, "the objective constant's coefficient b c",
                                  lambda: Fraction(potential_weight) * Fraction(self.constant),
                                  zero=potential_weight == 0)
            if not omitted:
                omitted = _underflows(-dt * coefficient, "the objective constant's phase contribution -dt b c",
                                      lambda: Fraction(dt) * Fraction(coefficient), zero=coefficient == 0)
            if omitted:
                charge = abs(Fraction(dt) * Fraction(potential_weight) * Fraction(self.constant))
                coefficient = 0.0
            _record(ledger, totals, "objective_constant", coefficient, dt)
        return charge

    def summary(self):
        """Return the ``OneHotSummary`` of every step, formed in one pass without a block object.

        Counts. Kept hopping blocks, kept projector blocks by support size and
        the kept blocks with the phase-diagonal provider, both potential halves
        included.

        Phase total. Each step adds its kinetic-diagonal event, its constant
        event when the constant is nonzero, and the identity events of every
        table entry in every potential factor (zero for a zero value or an
        omitted occurrence). The identity events of one tile are summed with
        ``math.fsum``, correctly rounded, and every such tile sum and every
        kinetic and constant event is added to the running totals overall and
        per source with Neumaier's compensation (``_Neumaier``), in step
        order. ``method._phase_contributions`` counts these partial sums and
        ``method._phase_ledger`` bounds the accumulation.

        Pruning and omissions. The threshold-pruned hopping and projector
        occurrences give the dropped count and their exact absolute angles in
        units of ``2**-1074``, and the range omissions give the omitted
        projector count with the exact traceless and identity charges.

        Block errors (``circuit_errors``). Formation, stage 1: an occurrence
        has the nominal duration ``t = r delta``, r = 1/2 for a second-order
        potential factor and for an odd-indexed link in the second-order
        kinetic layers and r = 1 otherwise. A hopping occurrence on a link of
        variable j intends ``lambda* = -t a^_k/(2 h_j**2)`` of
        ``G = (XX + YY)/2``, whose restriction has norm 1, and a projector on
        s qubits for value v intends ``zeta* = t b^_k v`` of ``P - I/2**s`` of
        norm ``1 - 2**-s``. When every rounded product and quotient lies
        strictly inside the normal binary64 range, the scalings by 2 and 4 are
        exact and the standard model gives ``lambda^ = lambda* (1 + theta_3)``
        and ``zeta^ = zeta* (1 + theta_2)`` with ``|theta_n| <= gamma_n``,
        u = 2**-53 (Higham, *Accuracy and Stability of Numerical Algorithms*,
        2nd ed., doi:10.1137/1.9780898718027, Lemma 3.1), so a hopping block
        costs at most ``3u/(1 - 6u) |lambda^|`` and a projector
        ``2u/(1 - 4u) (1 - 2**-s) |zeta^|``. The absolute angles are summed
        upward per kind and support size. Any other block costs its exact
        ``min(2, |lambda^ - lambda*|)`` or ``min(2, (1 - 2**-s) |zeta^ - zeta*|)``.
        Omitted blocks, stage 2: every pruned or range-omitted occurrence
        costs its exact ``|lambda*|`` or ``(1 - 2**-s) |zeta*|``. Gate
        parameters, stage 3: the gate is ``XXPlusYYGate(theta^)`` with
        ``theta^ = 2 fl(2 t c)`` (``pauli_evolution._hopping_parameter``), so a
        kept hopping block adds ``min(2, |theta^/2 - lambda^|)``, zero when
        ``theta^`` is finite and its half normal.

        Each stage compares two products of unitaries on the one-hot
        subspace, which differ by at most 2 in operator norm, so each stage
        total is capped at 2 after its blocks' bounds are added. An upward
        angle sum that reaches infinity exceeds ``2**1000``, so its scaled
        stage-1 bound exceeds 2 and the stage takes the cap. The upward
        ``phase_assignments`` sum is then infinite too, and
        ``method._native_phase_allowance`` turns it into an infinite
        allowance.
        """
        from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import _hopping_parameter

        from .binary import _scaled_absolute_sum_up
        from .method import _add_up

        u = Fraction(1, 2**53)
        grid, dt = self.grid, self.dt
        delta = Fraction(dt)
        half = dt / 2
        fast = _NORMAL_MIN < abs(half) < _NORMAL_MAX
        inverse_squares = [1 / Fraction(grid.spacing(j)) ** 2 for j in range(grid.num_variables)]
        totals = _Neumaier()
        sources = {}
        hopping_blocks = dense = 0
        projector_blocks = {}
        dropped_count = dropped_units = 0
        omitted_count, projector_charge, identity_charge, constant_events = 0, Fraction(0), Fraction(0), 0
        hopping_sum = 0.0
        projector_sums = {}
        assignments = 0.0
        formation = omitted_weight = hopping = Fraction(0)
        for step, (_time, kinetic_weight, potential_weight) in enumerate(self.step_weights):
            charge = self._step_phases(step, sources, totals)
            if charge:
                constant_events += 1
                identity_charge += charge
            for duration in self.factors():
                nominal = delta / (1 if self.order == 1 else 2)
                for tile in self._tiles(duration, potential_weight):
                    table = tile.table
                    s = table.size
                    norm = 1 - Fraction(1, 2**s)
                    partial = math.fsum(tile.phases)
                    _record_partial(sources, totals, "projector_identity", partial)
                    kept = tile.positions.size
                    projector_blocks[s] = projector_blocks.get(s, 0) + kept
                    dense += kept if table.diagonal else 0
                    if kept:
                        assignments = _add_up(assignments, _scaled_absolute_sum_up(tile.angles, 2.0**s))
                    # Stage 1: a block whose p and zeta are strictly inside the normal range, with a normal half
                    # duration, takes the a priori bound, and any other block its exact discrepancy.
                    a_priori = _inside(tile.coefficients) & _inside(tile.angles) & fast
                    if a_priori.any():
                        projector_sums[s] = _add_up(projector_sums.get(s, 0.0),
                                                    _scaled_absolute_sum_up(tile.angles[a_priori], 1.0))
                    for index in _listed(np.flatnonzero(~a_priori)):
                        value = float(tile.values[tile.positions[index] - tile.start])
                        intended = nominal * Fraction(potential_weight) * Fraction(value)
                        formation += min(Fraction(2), norm * abs(Fraction(float(tile.angles[index])) - intended))
                    # Stage 2: pruned and range-omitted occurrences at their exact intended exponents.
                    missing = (tile.values != 0.0) & (potential_weight != 0.0) & ~tile.emitted
                    if missing.any():
                        units = _exact_units(np.abs(tile.values[missing]))
                        omitted_weight += norm * abs(nominal * Fraction(potential_weight)) * Fraction(units, 1 << 1074)
                    dropped_count += tile.pruned.size
                    dropped_units += _exact_units(tile.pruned)
                    omitted_count += tile.omitted
                    projector_charge += tile.charge * norm
                    identity_charge += tile.charge / 2**s
                    # Release this tile's arrays before the next tile is formed.
                    del tile, a_priori, missing
            for layer in hopping_layers(grid, dt=dt, kinetic_weight=kinetic_weight, trotter_order=self.order,
                                        threshold=self.threshold):
                count = len(layer.links)
                share = 2 if layer.duration != dt else 1
                intended = -(delta / share) * Fraction(kinetic_weight) * inverse_squares[layer.variable] / 2
                if layer.pruned:
                    dropped_count += count
                    dropped_units += count * angle_units(layer.angle)
                if not layer.kept:
                    if layer.angle != 0.0:
                        omitted_weight += count * abs(intended)
                    continue
                hopping_blocks += count
                spacing = grid.spacing(layer.variable)
                denominator = 4.0 * spacing * spacing
                if fast and _NORMAL_MIN < abs(denominator) < _NORMAL_MAX and _NORMAL_MIN < abs(layer.coefficient) \
                        < _NORMAL_MAX and _NORMAL_MIN < abs(layer.angle) < _NORMAL_MAX:
                    hopping_sum = _add_up(hopping_sum, _scaled_absolute_sum_up(np.full(count, layer.angle), 1.0))
                else:
                    formation += count * min(Fraction(2), abs(Fraction(layer.angle) - intended))
                theta = _hopping_parameter(layer.duration, layer.coefficient)
                if not (theta * 0.5 == layer.angle and abs(layer.angle) >= _NORMAL_MIN):
                    hopping += count * min(Fraction(2), abs(Fraction(theta) / 2 - Fraction(layer.angle)))

        def scaled(coefficient, total):
            # An upward angle sum reaches infinity only above 2**1000, where its bound exceeds the cap 2 below.
            return coefficient * Fraction(total) if math.isfinite(total) else Fraction(2)

        formation += scaled(3 * u / (1 - 6 * u), hopping_sum) + sum(
            (scaled(2 * u / (1 - 4 * u) * (1 - Fraction(1, 2**s)), total) for s, total in projector_sums.items()),
            Fraction(0))
        return OneHotSummary(
            hopping_blocks=hopping_blocks,
            projector_blocks=tuple(sorted(projector_blocks.items())),
            dense_blocks=dense,
            phase_assignments=assignments,
            angle_formation=min(Fraction(2), formation),
            rotation_pruning=min(Fraction(2), omitted_weight),
            gate_parameters=min(Fraction(2), hopping),
            physical_phase=totals.value("total"),
            phase_sources=tuple(sources.items()),
            dropped_count=dropped_count,
            dropped_units=dropped_units,
            omitted_projectors=omitted_count,
            projector_charge=projector_charge,
            identity_events=omitted_count + constant_events,
            identity_charge=identity_charge,
        )


class OneHotSummary(NamedTuple):
    """The planning totals of a one-hot schedule (``OneHotSchedule.summary``).

    ``phase_assignments`` is the upward ``sum_b |zeta_b| 2**-s_b``, infinite
    when that sum leaves the binary64 range. The three stage bounds are
    exact rationals capped at 2.
    """

    hopping_blocks: int
    projector_blocks: tuple[tuple[int, int], ...]
    dense_blocks: int
    phase_assignments: float
    angle_formation: Fraction
    rotation_pruning: Fraction
    gate_parameters: Fraction
    physical_phase: float
    phase_sources: tuple[tuple[str, float], ...]
    dropped_count: int
    dropped_units: int
    omitted_projectors: int
    projector_charge: Fraction
    identity_events: int
    identity_charge: Fraction


def derivation_bytes(num_grid_points, num_variables, support_sizes):
    """Return the transient bytes of deriving one step's blocks, or the planning summary, of a one-hot schedule.

    One step's hopping layers hold at most ``_LINK_BYTES`` per link
    occurrence, ``K + floor(K/2)`` per variable at most. The potential holds
    one tile of the largest table, at most ``ENTRY_TILE`` of its ``K**s``
    entries at ``_TILE_ENTRY_BYTES`` each. A schedule without a table has no
    tile. ``_DERIVATION_FIXED_BYTES`` covers the
    chunk lists, scalar objects and NumPy bookkeeping. The emitted blocks
    and the caller's records are counted by each caller.
    """
    k, sizes = num_grid_points, set(support_sizes)
    entries = min(ENTRY_TILE, k ** max(sizes)) if sizes else 0
    return _DERIVATION_FIXED_BYTES + _LINK_BYTES * num_variables * (k + k // 2) + _TILE_ENTRY_BYTES * entries


def projector_providers(tables):
    """Return the ``QHDProjectorProvider`` of each support size of ``tables``, in increasing size.

    The provider and its CX count are those that
    ``structured_number_projector_provider`` chooses for the size, so every
    projector of one size has the same construction.
    """
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import structured_number_projector_provider

    from .records import QHDProjectorProvider

    return tuple(QHDProjectorProvider(support_size=s, **structured_number_projector_provider(s))
                 for s in sorted({len(t.support) for t in tables}))


def step_blocks(plan, step):
    """Return the blocks of one step of a one-hot QHD Plan, in the order the circuit applies them.

    The blocks of a one-hot Plan follow from its support tables and step
    weights. This function forms those of step ``step`` with the arithmetic
    of the circuit construction, so their angles, supports, constructions
    and CX counts are those of the gates the circuit applies in that step.
    Blocks that ``rotation_threshold`` removes, or whose angle or phase
    would lie below the normal binary64 range, are not in the circuit and
    are not returned. The guide's
    [scientific model](../../algorithms/qhd.md#scientific-model-and-ordering)
    section describes the step structure.

    Args:
        plan (Plan): A Plan of [`QHD`][nwqlib.algorithms.qhd.method.QHD] with
            ``encoding="one_hot"`` and either ``execution="quantum"`` or
            ``theory_flavor="ir_product"``.
        step (int): Index of the step, from 0 to ``num_steps - 1``.

    Returns:
        blocks (tuple[QHDBlock, ...]): One [`QHDBlock`][nwqlib.algorithms.qhd.records.QHDBlock]
            per block in the step. First order lists the projector blocks of every
            support table, then the hopping blocks. Second order lists the first
            potential half, the odd-indexed, even-indexed and odd-indexed hopping
            layers, then the second potential half. Projector blocks follow the
            support tables in the Plan's order and their entries in grid order.

    Raises:
        ValueError: If the Plan is not a one-hot Plan with a compiled
            schedule, if ``step`` is out of range, or if the step's records,
            at 4096 bytes per block for every link and table entry the step
            can contain, with the derivation's transient storage, exceed the
            Method's ``max_bytes`` or ``max_work``.
    """
    from .method import _BLOCK_BYTES, _grid
    from .records import QHDBlock

    r, method = plan.reconstruction, plan.method
    if method.encoding != "one_hot" or r.onehot is None:
        raise ValueError("step_blocks reads a one-hot QHD Plan with a compiled schedule; plan with "
                         "execution='quantum' or theory_flavor='ir_product'")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < len(r.step_weights):
        raise ValueError(f"step must be an integer from 0 to {len(r.step_weights) - 1}, not {step!r}")
    grid = _grid(plan)
    factors = 2 if method.trotter_order == 2 else 1
    links = grid.num_links
    occurrences = grid.num_variables * (links + links // 2) + factors * sum(len(t.values) for t in r.support_values)
    workspace = derivation_bytes(grid.num_grid_points, grid.num_variables, (len(t.support) for t in r.support_values))
    method._admit(occurrences, workspace + _BLOCK_BYTES * occurrences, stage="one-hot step blocks")
    return tuple(QHDBlock(**block._asdict()) for block in OneHotSchedule.of(r, method, grid).blocks(step))


def _record(sources, totals, name, coefficient, duration):
    """Add the event ``fl(-t q)`` of one kinetic-diagonal or constant contribution to its source and the total."""
    phase = -duration * coefficient
    if not (_NORMAL_MIN < abs(phase) < _NORMAL_MAX or (phase == 0.0 and (coefficient == 0.0 or duration == 0.0))):
        phase = _normal_product(-duration, coefficient, f"the {name} phase contribution -t q")
    _record_partial(sources, totals, name, phase)


def _record_partial(sources, totals, name, value):
    """Add one partial sum to its source total, then to the overall total, the source first."""
    sources[name] = totals.add(name, value)
    totals.add("total", value)
