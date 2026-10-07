"""QHD initial states, their classical start vector and their native one-hot or binary preparation.

``QHD.initial_state`` selects one of the state records below and
``QHD.initial_state_preparation`` the recipe that prepares it natively. Every
state is a product of nonnegative per-variable amplitude vectors on the
Method's grid (``variable_amplitudes``). Planning evaluates them once, together
with the construction error of the classical start vector (``evaluate``),
and stores both in ``QHDReconstruction``. The classical routes start from
the tensor product of the stored vectors in lexicographic grid order, the
enumeration of ``(n_0, ..., n_(d-1))`` with each digit in ``0, ..., K-1`` and
the last variable advancing fastest, so that variable 0 is most significant:
``i = sum_(j=0)^(d-1) n_j * K**(d-1-j)`` (``restricted_state``), except for the
uniform state and the kinetic ground state on the periodic grid, whose
entries ``1/sqrt(K**d)`` are filled directly with their own construction
term of 2u. The construction error (``restricted_state_error``) is the
start term of the host state budgets
(``nwqlib._validation.expm_multiply_state_error``,
``split_step.state_error`` and the budget that ``split_step.evolve``
observes), which ``method._host_probability_window``,
``method._host_tie_window`` and the kernel's tie window use. The native routes prepare the stored
per-variable vectors with the structured preparation of the one-hot
encoding, the amplitude chain (``append_amplitude_chain``), or with one
Qiskit ``StatePreparation`` per register
(``append_register_state_preparation``). The binary encoding's structured
preparation is one H per qubit, which prepares the uniform state only, and
its Qiskit route one ``StatePreparation`` of K amplitudes per register
(``append_binary_initial_state``).

The error bounds below use the unit roundoff u = 2**-53. Those of the uniform
state, the kinetic ground state and the chain are first order in u. Those of
the Gaussian state and of the product composition in
``restricted_state_error`` are finite. They assume round-to-nearest binary64
with gradual underflow, correctly rounded basic operations and square root,
and at most one ulp (relative error 2u, or 2**-1074 below the normal range)
in ``math.sin``, ``math.exp``, ``math.expm1``, ``math.log1p``, ``math.atan2``
and ``math.hypot``. The math module takes sin, exp, expm1, log1p and atan2
from the platform C library, so these are platform assumptions. Python documents an error below one ulp for its own ``hypot``
since 3.10.
"""

from __future__ import annotations

import math
import operator
import sys
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

import numpy as np
from pydantic import Field, model_validator

from nwqlib._validation import UNIT_ROUNDOFF
from nwqlib.core.records import Real, Record

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from nwqlib.algorithms.qhd.grid import OneHotGrid

Positive = Annotated[Real, Field(gt=0)]


def _normalized(values):
    """Return ``values/||values||_2`` as a float64 array, for nonnegative values with a positive entry.

    Rounding of this function. With ``n = ||v||`` and ``n_hat = hypot(v)``
    within one ulp, ``|n_hat - n| <= 2u n``, and each quotient
    ``fl(v_i/n_hat)`` has relative error at most u, or absolute error at most
    2**-1075 when it is subnormal, charged as the representable 2**-1074
    (``_SUBNORMAL_ULP``). Hence
    ``||w_hat - v/n_hat|| <= u/(1 - 2u) + sqrt(K) 2**-1074`` and
    ``||v/n_hat - v/n|| = |n/n_hat - 1| <= 2u/(1 - 2u)``, so the result lies
    within ``3u/(1 - 2u) + sqrt(K) 2**-1074`` of ``v/||v||`` for K entries,
    and its norm within the same amount of one. This is a finite bound.

    Error of the supplied values. When they carry independent relative
    errors ``e_i u`` against exact values a, first-order perturbation of
    ``a/||a||`` removes the component along a, so the direction errs by at
    most ``u sqrt(sum_i a_i**2 e_i**2)/||a||`` to first order
    (``KineticGroundState``). For large errors this argument fails, and
    ``_gaussian_beta`` uses the finite bound ``2 ||t - v||/||v||`` instead.
    """
    norm = math.hypot(*values)
    return np.array([value / norm for value in values], dtype=float)


class UniformState(Record):
    """Uniform initial state: equal amplitude on every valid grid point.

    Pass it as `QHD(initial_state=UniformState())`. It takes no arguments. Each variable
    has amplitude `1/sqrt(K)` on each of its K grid points, so the state has
    `1/sqrt(K**d)` on every grid tuple. This is the uniform superposition of Leng et
    al., arXiv:2303.01471v1, Algorithm 1 step 3, and the initial state that Wu et al.,
    arXiv:2605.12066v1, Sec. VI, state for their simulations. On the Dirichlet grids it
    has a component outside the kinetic ground state
    ([`KineticGroundState`][nwqlib.algorithms.qhd.initial_state.KineticGroundState]),
    which the guide's
    [initial states](../../algorithms/qhd.md#initial-states-and-preparation) section
    discusses.

    Attributes:
        kind: Default `"uniform"`, the only accepted value. Initial-state identifier.
    """

    kind: Literal["uniform"] = "uniform"

    def variable_amplitudes(self, grid: OneHotGrid) -> tuple[np.ndarray, ...]:
        """Return `1/sqrt(K)` on the K points of each variable.

        Each entry is one correctly rounded square root and one division, so it errs by at
        most 2u, with `u = 2**-53`. The classical start vector does not use these vectors.
        It fills the equal product entry `1/sqrt(K**d)` directly.
        """
        k = grid.num_grid_points
        return tuple(np.full(k, 1.0 / math.sqrt(k)) for _ in range(grid.num_variables))


class KineticGroundState(Record):
    """Ground state of each variable's kinetic operator, the default initial state of QHD.

    Pass it as `QHD(initial_state=KineticGroundState())`, which is the default. It takes
    no arguments. The row "QHD Method defaults" of the
    [engineering constants](../../ENGINEERING_CONSTANTS.md#safety-factors-and-workflow-defaults)
    gives the reasons for this default and its limits.

    On both Dirichlet grids the ground state of one variable is
    `sin(pi (i + 1)/(K + 1))`, i = 0..K-1, with every entry positive and energy
    `2 sin(pi/(2 (K + 1)))**2/h**2` for the grid spacing h. The kinetic operator of d
    variables is the sum of the one-variable operators on separate tensor factors, so
    its ground state is the product of these vectors and its energy the sum of theirs.
    On the grid with boundary points the vector does not vanish at the box endpoints,
    because that grid's missing neighbors lie outside the box.

    On the periodic grid the ground state is the uniform state with energy zero, and
    this record then prepares exactly what
    [`UniformState`][nwqlib.algorithms.qhd.initial_state.UniformState] prepares. The
    uniform state is the ground state under both periodic kinetic models, the
    finite-difference stencil and the spectral model (`QHD.kinetic_model="spectral"`),
    which the Method accepts on the periodic grid only.

    Attributes:
        kind: Default `"kinetic_ground"`, the only accepted value. Initial-state
            identifier.
    """

    kind: Literal["kinetic_ground"] = "kinetic_ground"

    def variable_amplitudes(self, grid: OneHotGrid) -> tuple[np.ndarray, ...]:
        """Return the normalized ground-state vector of each variable's kinetic operator.

        On the periodic grid it is the uniform vector (``UniformState``). On the Dirichlet
        grids it is ``sin(pi (i + 1)/(K + 1))``. ``sin(pi - theta) = sin(theta)`` lets entry
        i use the angle ``theta_i = pi m_i/(K + 1)`` with ``m_i = min(i + 1, K - i)``, so
        ``0 < theta_i <= pi/2``. The computed angle has relative error at most 3u
        (``math.pi``, the product with the integer m_i and the division by the integer K +
        1). A relative angle error eta changes ``sin(theta)`` by the relative amount
        ``theta cot(theta) eta``, and ``0 <= theta cot(theta) < 1`` on ``(0, pi/2]``, so
        with the one-ulp sine each entry errs by at most 5u, with ``u = 2**-53``.
        ``variable_errors`` adds the 3u of the normalization.
        """
        # On both Dirichlet grids the kinetic operator of one variable, restricted to the
        # grid, is T/h**2 with the K-by-K tridiagonal matrix T of diagonal 1 and
        # off-diagonals -1/2. The interior grid and the grid with boundary points differ only
        # in the spacing h, so they share the eigenvectors of T. For
        # v_i = sin(r pi (i + 1)/(K + 1)), i = 0..K-1 and r = 1..K, the identity
        # sin(a - b) + sin(a + b) = 2 sin(a) cos(b) gives
        # (T v)_i = v_i - (v_(i-1) + v_(i+1))/2 = (1 - cos(r pi/(K + 1))) v_i, where the
        # missing neighbors v_(-1) = sin(0) and v_K = sin(r pi) vanish as the stencil
        # requires. The eigenvalues 2 sin(r pi/(2 (K + 1)))**2/h**2 increase with r, so r =
        # 1 is the ground state, sin(pi (i + 1)/(K + 1)) with every entry positive and
        # energy 2 sin(pi/(2 (K + 1)))**2/h**2.
        #
        # On the periodic grid the operator is C/h**2 with the circulant
        # C = I - (S + S^T)/2 and the cyclic shift S. The Fourier vectors
        # f_j = exp(2 pi i r j/K) satisfy (C f)_j = (1 - cos(2 pi r/K)) f_j, so the
        # eigenvalues 2 sin(pi r/K)**2/h**2, r = 0..K-1, are zero only for r = 0, whose
        # eigenvector is the constant vector. The ground state is therefore the uniform state
        # with energy zero. The spectral kinetic model has the same Fourier eigenvectors with
        # eigenvalues 2 pi**2 q**2/L**2 for the signed frequency index q and period L = K h.
        # They also vanish only for q = 0, so the uniform state is the ground state under both
        # periodic kinetic models.
        if grid.boundary == "periodic":
            return UniformState().variable_amplitudes(grid)
        k = grid.num_grid_points
        vector = _normalized([math.sin(math.pi * min(i + 1, k - i) / (k + 1)) for i in range(k)])
        return tuple(vector.copy() for _ in range(grid.num_variables))

    def variable_errors(self, grid: OneHotGrid) -> tuple[float, ...]:
        """Return the first-order 2-norm error of each vector in units of u.

        It is 8 on the Dirichlet grids (5 per entry, 3 normalizing) and 2 on the periodic
        grid, where each entry ``1/sqrt(K)`` takes one square root and one division.
        """
        return (2.0 if grid.boundary == "periodic" else 8.0,) * grid.num_variables


# Constants of the finite Gaussian and product-state error bounds, exact
# binary64 consequences rather than tuned values. Registered in
# docs/ENGINEERING_CONSTANTS.md. Revisit only with another floating-point
# format.
# - _EXPONENT_RELATIVE: a computed exponent z_hat carries five rounding
#   factors (GaussianState._variables), so z_hat = z (1 + theta) with
#   |theta| <= gamma_5 = 5u/(1 - 5u) (Higham, doi:10.1137/1.9780898718027,
#   Lemma 3.1), and |z - z_hat| <= z_hat gamma_5/(1 - gamma_5) = z_hat 5u/(1 - 10u).
# - _EXPONENT_ABSOLUTE: gradual underflow in q, q q or the halving of q q
#   changes an exponent below 2**-1021 by at most 2**-1074, covered by 2**-1073.
# - _SUBNORMAL_ULP: one ulp of a subnormal number, 2**-1074, the error of
#   exp there. It is also the representable allowance for every absolute
#   subnormal rounding below, including the half-ulp 2**-1075 of a correctly
#   rounded quotient or product, which binary64 cannot represent.
# - _OUTWARD: lifts each computed bound above its exact value. Every bound is
#   a monotone expression in nonnegative quantities. The arguments of expm1
#   are inflated by this factor before evaluation. The argument of exp in
#   _gaussian_beta has magnitude at most 746 wherever exp is nonzero and at
#   most three roundings, which change exp by less than 2.5e-13 relative.
#   The other operations, math.hypot among them, number fewer than 256, each
#   rounding by at most 2u (library functions to one ulp) with relative
#   sensitivity at most 2, which adds at most 1024u, about 1.1e-13. Both
#   together stay below 2**-40, about 9.1e-13. A result beyond the binary64
#   range becomes inf (_within_range), which the caps absorb.
_EXPONENT_RELATIVE = 5 * UNIT_ROUNDOFF / (1 - 10 * UNIT_ROUNDOFF)
_EXPONENT_ABSOLUTE = 2.0**-1073
_SUBNORMAL_ULP = 2.0**-1074
_OUTWARD = 1 + 2.0**-40


def _within_range(function, *arguments):
    """Return ``function(*arguments)``, or inf when the math module reports a result beyond the binary64 range.

    ``math.exp``, ``math.expm1``, ``math.hypot`` and the true division of
    integers (``restricted_state_error``) raise OverflowError there.
    An upper bound that is not representable is infinite, and the callers'
    caps then apply.
    """
    try:
        return function(*arguments)
    except OverflowError:
        return math.inf


def _gaussian_beta(values, shifted, enclosures):
    """Return an upper bound on ``beta = ||t - v||/||v||`` for one choice of common factor, or inf.

    ``values`` are the computed amplitudes ``v_i = fl(exp(-y_i))`` of one
    variable, with ``shifted`` the computed shifted exponents y_i and
    ``||v|| >= 1``. t is the exact Gaussian vector times the common factor
    that the caller chose, ``t_i = exp(-y_i - Delta_i)``. ``enclosures[i]``
    is a finite ``x_i >= |Delta_i|``, or None for an entry whose computed
    value is 0 and whose exact value is below 2**-1074
    (``GaussianState._variables``).

    Entry bound. With ``p_i = exp(-y_i)`` the exact exponential of the
    computed argument, ``|t_i - p_i| = p_i |exp(-Delta_i) - 1| <= p_i expm1(x_i)``,
    because ``|exp(d) - 1| <= exp(|d|) - 1``. For a normal ``v_i`` the
    one-ulp exponential gives ``|v_i - p_i| <= 2u p_i``, so
    ``p_i <= v_i/(1 - 2u)`` and
    ``|t_i - v_i| <= v_i (expm1(x_i) + 2u)/(1 - 2u)``. For a subnormal or zero
    ``v_i`` one ulp is ``2**-1074``, so ``p_i <= v_i + 2**-1074`` and
    ``|t_i - v_i| <= (v_i + 2**-1074) expm1(x_i) + 2**-1074``. Such an entry
    also satisfies ``|t_i - v_i| <= t_i + v_i`` with
    ``t_i <= exp(x_i - y_i)``, which stays small when the exponent error is
    large but far below the exponent itself, and the smaller of the two
    bounds is used, ``exp(x_i - y_i)`` rounded up by one subnormal ulp. No
    entry is treated as exactly zero. The bounds become infinite only where
    expm1 or exp exceeds the binary64 range.

    ``beta = ||(t_i - v_i)_i||/||v||``, evaluated with ``math.hypot`` for
    both norms, which scales its argument and so stays finite whenever the
    norm is representable (entries near 1e154 have squares whose sum
    overflows, yet their norm does not), and returns inf only when the norm
    itself exceeds the range. Its rounding, below one ulp, is counted in
    ``_OUTWARD``, which also lifts the result (each x is inflated by it
    before expm1). The caller turns
    beta into a direction bound with the exact inequality
    ``||t/||t|| - v/||v|| || <= 2 beta``: the left side is at most
    ``||t|| |1/||t|| - 1/||v||| + ||t - v||/||v||``, and the first term is
    ``| ||v|| - ||t|| |/||v|| <= beta``.
    """
    errors = []
    for value, y, x in zip(values, shifted, enclosures, strict=True):
        if x is None:
            # Computed 0 and exact value below 2**-1074.
            errors.append(_SUBNORMAL_ULP)
            continue
        x *= _OUTWARD
        growth = _within_range(math.expm1, x)
        if value >= sys.float_info.min:
            # |t - v| <= v (expm1(x) + 2u)/(1 - 2u) for a normal computed value.
            errors.append(value * (growth + 2 * UNIT_ROUNDOFF) / (1 - 2 * UNIT_ROUNDOFF))
            continue
        # Below the normal range: the smaller of (v + 2**-1074) expm1(x) + 2**-1074
        # and v + exp(x - y) + 2**-1074, where t <= exp(x - y).
        errors.append(min((value + _SUBNORMAL_ULP) * growth + _SUBNORMAL_ULP,
                          value + _within_range(math.exp, x - y) + _SUBNORMAL_ULP))
    # Scale-safe norms: hypot is finite whenever the norm is representable.
    return _OUTWARD * _within_range(math.hypot, *errors) / math.hypot(*values)


class GaussianState(Record):
    """Gaussian warm start `prod_j exp(-(x_j - c_j)**2/(2 sigma_j**2))`, normalized on the grid points.

    Build it with keyword arguments, for example
    `GaussianState(center=(0.5, -0.2), widths=(0.3, 0.3))`, and pass it as
    `QHD(initial_state=...)`. `center` and `widths` are both required, with one entry
    per Problem variable. The center c is a point in the coordinates of the Problem that
    the Plan solves, and the widths sigma_j are in the units of each of its variables.
    For `solve` these are the original coordinates. Box refinement plans each level's
    own problem with the QHD configuration's state, and its default search model solves
    in the unit coordinates `u = (x - a)/L` of each level box `[a, a + L]`, so there c
    and sigma are unit coordinates. The center may lie outside the box. The amplitude of
    a grid tuple is the product of the per-variable factors at its grid coordinates, and
    the state is that product normalized over the valid grid points.

    Leng et al., arXiv:2303.01471v1, Algorithm 1 step 3, name a Gaussian state as one
    choice of initial state, and the code behind the refinement results of Wu et al.,
    arXiv:2605.12066v1, starts every refinement level after the first from this
    amplitude, centered at the best point found so far
    (`BoxRefinement(level_initial_state="best_point_gaussian")`). The amplitude, not the
    probability, has width sigma, so the continuous probability density, proportional to
    `exp(-(x_j - c_j)**2/sigma_j**2)`, has standard deviation `sigma_j/sqrt(2)` along
    variable j. A Gaussian contains several momentum components of the kinetic operator,
    so it does not avoid the time-discretization dependence that the kinetic ground
    state avoids
    ([initial states](../../algorithms/qhd.md#initial-states-and-preparation)). On the
    periodic grid the distance is the chart difference `|x_ji - c_j|` of the coordinates
    in `[lower, upper)`, not the distance on the circle, so the amplitudes do not
    continue across the wrap link. For a center near one end of the period, the points
    near the other end, which are its neighbors through that link, get the amplitudes of
    their chart distance. A minimum-image variant is open work
    ([Limitations and open work](../../ROADMAP.md#qhd-core-models)).

    Every finite center and every finite positive width is accepted. Each variable's
    exponents `z_ji = (x_ji - c_j)**2/(2 sigma_j**2)` are shifted by their minimum
    before exponentiation, so the largest amplitude is exactly 1 before normalization
    and the state never underflows to zero. The construction error bound of the
    classical start vector is finite for every input. It grows with the exponents, about
    10u times the exponent per entry, with `u = 2**-53`, and it is capped by the
    distance of two nonnegative vectors of norm about one, about sqrt(2). The
    per-variable bound of `variable_errors` is infinite where the exponents give no
    bound.

    Attributes:
        kind: Default `"gaussian"`, the only accepted value. Initial-state identifier.
        center: Required. Finite center coordinate of each variable, in the Problem's
            variable order and the coordinates of the solved Problem, which are unit
            coordinates for a search-model refinement level.
        widths: Required. Positive finite width sigma of each variable, in the same
            coordinates.

    Raises:
        ValueError: If `center` is empty or `center` and `widths` differ in length.
            Planning also raises when they do not have one entry per Problem variable.
    """

    kind: Literal["gaussian"] = "gaussian"
    center: tuple[Real, ...]
    widths: tuple[Positive, ...]

    @model_validator(mode="after")
    def _one_entry_per_variable(self):
        if not self.center or len(self.center) != len(self.widths):
            raise ValueError("GaussianState needs one center coordinate and one width per variable")
        return self

    def _variables(self, grid):
        """Return per variable the unnormalized amplitudes, largest entry 1, and a bound on their direction error.

        Exponents. Entry i of a variable uses ``q_i = |x_i - c|/sigma`` and
        ``z_i = (q_i q_i)/2``. With ``|x_i - c|`` computed first, q carries two
        rounding factors. The difference ``x_i - c`` stays finite on every
        admitted grid with binary64 bounds, as the ``Real`` bounds of an
        ``Optimization`` are. It overflows only at
        ``|x_i - c| >= 2**1024 - 2**970``, and ``|c| <= 2**1024 - 2**971``, so
        it needs ``|x_i| >= 2**970``. An admitted grid has spacing h below
        2**511 (at most about 3.3e153, ``grid.OneHotGrid``) and coordinates
        ``x_i = lower + n_i h`` with ``n_i <= K``. So either K exceeds 2**457
        or ``|lower| >= 2**969``. In the second case the upper bound, a
        different binary64 number, differs from lower by at least 2**916, and
        h below 2**511 again needs more than 2**400 intervals. No such grid
        can be enumerated point by point, as this evaluation does. The
        division by sigma may give inf, which the branches below handle. The
        product q q and the exact halving add one factor, so the computed
        exponent satisfies ``|z - z_hat| <= _EXPONENT_RELATIVE z_hat +
        _EXPONENT_ABSOLUTE``, the absolute term covering gradual underflow. A
        computed ``z_hat = inf`` comes from q or q q above the range, so its
        exact exponent is at least 2**1022.

        Amplitudes. With ``z_min = min_i z_hat_i`` the amplitude is
        ``v_i = exp(-fl(z_hat_i - z_min))``, so an entry at the minimum is
        exactly 1 and the vector never underflows to zero. An overflowed
        ``z_hat_i = inf`` gives 0.

        Direction bound, finite branch. Normalization removes any factor common
        to all entries, so two common factors give two bounds on the same
        quantity, ``||t/||t|| - v/||v|| ||`` for the exact Gaussian vector t,
        and the smaller is returned (``_gaussian_beta``, twice beta). With
        ``y_i = fl(z_hat_i - z_min)``, whose subtraction errs by at most
        ``u y_i/(1 - u)``, the enclosures of the exponent error are the
        following.

        - Common factor ``exp(z_min)`` of the computed minimum:
          ``x_i = _EXPONENT_RELATIVE z_hat_i + _EXPONENT_ABSOLUTE + u y_i/(1 - u)``.
        - Common factor ``exp(z_m)`` of the exact exponent of the first
          minimizing entry m, which is then exact (``x_m = 0``), and for the
          others ``x_i = _EXPONENT_RELATIVE (z_hat_i + z_hat_m) +
          2 _EXPONENT_ABSOLUTE + u y_i/(1 - u)``. This one stays small for a
          far center with steep decay, where the first grows with z_min.

        An overflowed entry has exact exponent difference at least
        ``2**1022 - 2**1020 (1 + 6u)``, above 2**1021, when ``z_min <= 2**1020``,
        so its exact amplitude lies below 2**-1074 (enclosure None). With a
        larger z_min the bound is infinite.

        Direction bound, saturated branch. If ``z_min`` itself overflows, the
        exact exponent of the nearest entry m is at least 2**1022. The
        distances are compared as ``|x_i/2 - c/2|``, which never overflows and
        has relative error at most 2u here. The entries at the smallest
        computed distance get 1 and all others 0. When that distance is
        attained once and every other one is at least its rounded product with
        ``1 + 16u``, the key and product roundings leave the exact distances
        at ``d_i >= d_m (1 + 8u)``, so ``z_i - z_m >= 16u z_m >= 16u 2**1022``
        and every exact entry other than m, relative to ``exp(-z_m)``, lies
        below 2**-1074. Against the exact vector with ``t_m = 1``, beta is
        then at most ``sqrt(K - 1) 2**-1074``, and the direction bound is
        twice that. Tied or nearly tied distances are not resolved, the vector
        is uniform over the tie, and the bound is infinite.

        Raises:
            ValueError: The center and widths do not have one entry per
                variable.
        """
        d, k = grid.num_variables, grid.num_grid_points
        if len(self.center) != d:
            raise ValueError(
                f"GaussianState center and widths need one entry per Problem variable, {d} here"
            )
        u = UNIT_ROUNDOFF
        rows = []
        for j, (center, width) in enumerate(zip(self.center, self.widths, strict=True)):
            coordinates = [grid.grid_value(j, i) for i in range(k)]
            # q = |x - c|/sigma, where x - c is finite on an admitted grid.
            ratios = [abs(x - center) / width for x in coordinates]
            # z = q**2/2 = (x - c)**2/(2 sigma**2), inf when q q overflows.
            exponents = [0.5 * (q * q) for q in ratios]
            lowest = min(exponents)
            if math.isinf(lowest):
                # Halved distances never overflow and only order the entries.
                distances = [abs(0.5 * x - 0.5 * center) for x in coordinates]
                nearest = min(distances)
                values = [1.0 if distance == nearest else 0.0 for distance in distances]
                separated = values.count(1.0) == 1 and all(
                    distance == nearest or distance >= nearest * (1 + 16 * u) for distance in distances)
                bound = 2 * _OUTWARD * math.sqrt(k - 1) * _SUBNORMAL_ULP if separated else math.inf
                rows.append((values, bound))
                continue
            # Shifted by the minimum: exp(-(z - z_min)) is exactly 1 at the minimum.
            shifted = [z - lowest for z in exponents]
            values = [math.exp(-y) for y in shifted]
            peak = exponents.index(lowest)
            if any(map(math.isinf, exponents)) and lowest > 2.0**1020:
                rows.append((values, math.inf))
                continue
            # Exponent-error enclosures against exp(z_min) and against exp(z_m), entry m exact.
            common = [None if math.isinf(z) else
                      _EXPONENT_RELATIVE * z + _EXPONENT_ABSOLUTE + u * y / (1 - u)
                      for z, y in zip(exponents, shifted, strict=True)]
            relative = [None if math.isinf(z) else 0.0 if i == peak else
                        _EXPONENT_RELATIVE * (z + lowest) + 2 * _EXPONENT_ABSOLUTE + u * y / (1 - u)
                        for i, (z, y) in enumerate(zip(exponents, shifted, strict=True))]
            # ||t/||t|| - v/||v|| || <= 2 beta for either common factor.
            bound = 2 * min(_gaussian_beta(values, shifted, common), _gaussian_beta(values, shifted, relative))
            rows.append((values, bound))
        return rows

    def variable_amplitudes(self, grid: OneHotGrid) -> tuple[np.ndarray, ...]:
        """Return the normalized Gaussian factor of each variable on its grid points.

        Each variable's exponents are shifted by their minimum before exponentiation, so the
        largest entry is exactly 1 before normalization.
        """
        return tuple(_normalized(values) for values, _ in self._variables(grid))

    def variable_errors(self, grid: OneHotGrid) -> tuple[float, ...]:
        """Return a 2-norm error bound, in units of u, of each normalized vector against its exact direction.

        It is the direction bound of the computed amplitudes against the exact Gaussian
        vector plus the rounding of the normalization, `3u/(1 - 2u) + sqrt(K) 2**-1074`,
        evaluated upward with the factor `1 + 2**-40`. Here `u = 2**-53`. The subnormal
        allowance is representable, and its addition to the far larger terms rounds like
        every other operation that the factor `1 + 2**-40` counts.
        The bound can be infinite when some exponents overflow and the smallest
        exponent exceeds ``2**1020``, or when the smallest exponent itself overflows
        and the grid points nearest the center are tied or nearly tied. Even with
        finite exponents, it can be infinite if both evaluations of ``_gaussian_beta``
        overflow, or if the smaller result overflows when doubled or converted to
        units of u.
        """
        return self._errors(self._variables(grid), grid)

    @staticmethod
    def _errors(rows, grid):
        """Return ``variable_errors`` from the rows of one ``_variables`` evaluation."""
        u, k = UNIT_ROUNDOFF, grid.num_grid_points
        normalization = 3 * u / (1 - 2 * u) + math.sqrt(k) * _SUBNORMAL_ULP
        return tuple(_OUTWARD * (bound + normalization) / u for _, bound in rows)


# The type of QHD.initial_state. The kind field selects the record when a
# saved Method is loaded.
InitialState = Annotated[
    UniformState | KineticGroundState | GaussianState, Field(discriminator="kind")
]


def restricted_state(state, grid: OneHotGrid, amplitudes=None) -> np.ndarray:
    """Return the initial state on the K**d valid grid points, in lexicographic grid-index order.

    The state is the tensor product of the per-variable vectors
    ``amplitudes``, by default ``state.variable_amplitudes(grid)``, in
    lexicographic grid order, the enumeration of ``(n_0, ..., n_(d-1))`` with
    each digit in ``0, ..., K-1`` and the last variable advancing fastest, so
    that variable 0 is most significant: ``i = sum_(j=0)^(d-1) n_j * K**(d-1-j)``,
    so ``np.kron`` of the per-variable vectors in variable order gives it.
    The classical kernel passes the vectors that planning evaluated and
    stored (``evaluate``, ``QHDReconstruction.initial_amplitudes``), so it
    evaluates no amplitudes itself. The uniform state's product has the
    single value ``1/sqrt(K**d)``, which is formed directly, and so has the
    kinetic ground state on the periodic grid, which is the uniform state
    (``KineticGroundState``). The result is complex, as the host evolution
    requires. Forming the product holds the d converted input vectors
    (``8 d K`` bytes), the real products of the earlier Kronecker stages, at
    most ``D/K`` entries before the last one, and the preallocated complex
    result, into which the last Kronecker stage writes its products
    directly. Each entry is the same single binary64 product that
    ``np.kron`` forms, with imaginary part zero, as a cast of the real
    product would give. Its explicit array payload is at most
    ``24 D + 8 d K`` bytes (``24 D`` for d = 1, whose input is the product).
    Python and NumPy object overhead is separate. The complex result becomes
    the evolved state, and the temporaries are released before the evolution
    allocates its buffers. ``start_vector_work`` counts its work, and
    ``restricted_state_error`` bounds its error.
    """
    if _is_uniform(state, grid):
        dimension = grid.num_grid_points**grid.num_variables
        return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=complex)
    if amplitudes is None:
        amplitudes = state.variable_amplitudes(grid)
    vectors = [np.asarray(vector, dtype=float) for vector in amplitudes]
    last = vectors.pop()
    prefix = reduce(np.kron, vectors) if vectors else np.ones(1)
    result = np.empty(prefix.size * last.size, dtype=complex)
    # The last Kronecker stage prefix[i] * last[j] is written into the complex result.
    np.multiply(prefix[:, None], last[None, :], out=result.reshape(prefix.size, last.size))
    return result


def evaluate(state, grid: OneHotGrid) -> tuple[tuple[np.ndarray, ...], float]:
    """Return ``(variable_amplitudes(grid), restricted_state_error(state, grid))`` from one amplitude evaluation.

    Planning calls this once and stores both results
    (``QHDReconstruction.initial_amplitudes`` and ``initial_state_error``),
    which the classical kernel, its windows and the native builder read, so
    no later stage evaluates the amplitudes again. The Gaussian's error
    bound comes from the same ``_variables`` rows as its amplitudes. The
    other states' error bounds evaluate no amplitudes.
    """
    if isinstance(state, GaussianState):
        rows = state._variables(grid)
        amplitudes = tuple(_normalized(values) for values, _ in rows)
        return amplitudes, restricted_state_error(state, grid, errors=GaussianState._errors(rows, grid))
    return state.variable_amplitudes(grid), restricted_state_error(state, grid)


def stored_amplitudes(amplitudes):
    """Return the per-variable vectors as the immutable float64 arrays that ``QHDReconstruction`` stores.

    Each vector is copied once into its own read-only float64 array
    (``core.records.FrozenArray``) with its entries and axis order
    unchanged, so the stored vectors are bitwise the evaluated ones and are
    not renormalized (``method._initial_state_bytes``).
    """
    from nwqlib.core.records import FrozenArray

    return tuple(FrozenArray(np.asarray(vector, dtype=np.float64)) for vector in amplitudes)


def start_vector_work(state, grid: OneHotGrid) -> int:
    """Return the work units of forming the classical start vector (``restricted_state``) from stored vectors.

    With ``D = K**d``, the direct uniform complex fill writes D entries, so
    it costs D. Otherwise reading the d stored float64 rows of K entries
    costs ``d K`` entry units, the left-associated Kronecker product
    stage j (j = 2..d) multiplies a product of ``K**(j - 1)`` entries by a
    vector of K entries and writes ``K**j`` entries, one unit each, the
    last stage writing into the complex128 result, and a further D is
    charged for that result, so the charge is
    ``d K + sum_(j=2)^d K**j + D``. The sum is empty for d = 1. These are
    logical entry and action units, like the kernel laws that add this
    charge once per evolution (``split_step.sizes``,
    ``method.restricted_sizes``), not CPU instructions. The amplitudes and
    their error bound belong to planning (``evaluate``) and are not charged
    again. K comes from the grid, not from a root of D.
    """
    d, k = grid.num_variables, grid.num_grid_points
    if _is_uniform(state, grid):
        return k**d
    return d * k + sum(k**j for j in range(2, d + 1)) + k**d


def _is_uniform(state, grid):
    """Return whether the record's state on this grid is the uniform state."""
    return type(state) is UniformState or (type(state) is KineticGroundState and grid.boundary == "periodic")


def restricted_state_error(state, grid: OneHotGrid, *, errors=None) -> float:
    """Return a bound, in units of u, on the 2-norm error of ``restricted_state``.

    The bound compares the computed vector psi_hat with the exact unit state
    psi of the record on the binary64 grid coordinates.

    Uniform state. The entry ``1/sqrt(D)`` is one correctly rounded square
    root and one division of the exactly representable D, 2u to first order.

    Product states. Let ``w_hat_j`` be the computed per-variable vectors,
    within ``e_j = variable_errors[j] u`` of their exact unit directions w_j
    (a first-order value for the kinetic ground state, a finite one for the
    Gaussian), and ``||w_hat_j|| <= N_j = 1 + 3u/(1 - 2u) + sqrt(K) 2**-1074``
    by the rounding of ``_normalized``. Telescoping over the factors gives
    the finite bound
    ``||kron(w_hat) - kron(w)|| <= sum_j e_j prod_(k<j) (1 + e_k) = prod_j (1 + e_j) - 1``,
    since ``||w_hat_k|| <= 1 + e_k`` and ``||w_k|| = 1``. Each product entry
    passes d - 1 roundings, which move the vector by at most
    ``gamma_(d-1) prod_j N_j`` with ``gamma_n = n u/(1 - n u)`` (Higham,
    doi:10.1137/1.9780898718027, Lemma 3.1), and gradual underflow of the
    products adds at most ``sqrt(D) (d - 1) 2**-1075``, charged as
    ``(d - 1) ceil(sqrt(D)) 2**-1074`` (``_SUBNORMAL_ULP``), one correctly
    rounded quotient of the exact integers ``(d - 1) ceil(sqrt(D))`` and
    ``2**1074``, with ``ceil(sqrt(D)) = isqrt(D - 1) + 1``. D is therefore
    never converted to a float, which fails once ``K**d`` exceeds the
    binary64 range, for example at d = 520 with K = 4. The term itself
    exceeds 1 only when ``(d - 1) ceil(sqrt(D))`` exceeds ``2**1074``.
    Range limit. Both the underflow allowance and the final bound in units
    of u must be finite. If ``n = (d - 1) ceil(sqrt(D))`` dominates the
    error estimate, the returned value is about ``n 2**-1021``. It leaves
    the binary64 range near ``n = 2**2045``, before the allowance itself
    overflows near ``n = 2**2098``. For K = 4, d = 2035 already exceeds
    the return-value range. A named ValueError recommends UniformState(),
    whose bound is independent of D, or fewer grid points or variables.
    In the small-roundoff regime where the underflow allowance is
    negligible, the sum is about ``(9d - 1) u`` for the kinetic ground
    state on the Dirichlet grids. Rounding of the integer quotient and
    its addition to other terms is covered by ``_OUTWARD``.

    Cap. The computed state has norm at most
    ``N = prod_j N_j (1 + gamma_(d-1)) + sqrt(D) (d - 1) 2**-1074``, and both
    it and psi are nonnegative, so
    ``||psi_hat - psi||**2 = ||psi_hat||**2 + 1 - 2 <psi_hat, psi> <= N**2 + 1``.
    The result is the smaller of the two bounds, about sqrt(2) when the
    Gaussian bound is infinite or large, evaluated upward with ``_OUTWARD``.
    ``errors`` supplies ``state.variable_errors(grid)`` when the caller has
    already evaluated them (``evaluate``). The value is the start term of the
    host state budgets, ``nwqlib._validation.expm_multiply_state_error`` and
    ``split_step.state_error``.

    Raises:
        ValueError: The underflow allowance or the final bound in units of u
            exceeds the binary64 range.
    """
    if _is_uniform(state, grid):
        return 2.0
    u, d, k = UNIT_ROUNDOFF, grid.num_variables, grid.num_grid_points
    gamma = (d - 1) * u / (1 - (d - 1) * u)
    # (d - 1) ceil(sqrt(D)) 2**-1074 >= sqrt(D) (d - 1) 2**-1074, one correctly rounded quotient of exact
    # integers, so D = K**d, which can exceed the binary64 range, is never converted to a float
    dimension = k**d
    root = math.isqrt(dimension - 1) + 1
    underflow = _within_range(operator.truediv, (d - 1) * root, 1 << 1074)
    if not math.isfinite(underflow):
        raise ValueError(
            f"the construction error bound of {type(state).__name__} on K**d = {k}**{d}, about "
            f"10**{d * math.log10(k):.0f} grid points, exceeds the binary64 range, because its underflow term "
            "grows with sqrt(K**d). Use UniformState(), whose bound does not depend on the grid size, or fewer "
            "grid points or variables")
    norm = (1 + 3 * u / (1 - 2 * u) + math.sqrt(k) * _SUBNORMAL_ULP) ** d
    # prod_j (1 + e_j) - 1, formed as expm1 of the summed log1p to keep small errors.
    errors = state.variable_errors(grid) if errors is None else errors
    total = math.fsum(math.log1p(error * u) for error in errors)
    growth = _within_range(math.expm1, total)
    error = growth + gamma * norm + underflow
    # Nonnegative vectors of norms N and 1 lie at most sqrt(N**2 + 1) apart.
    bound = norm * (1 + gamma) + underflow
    cap = math.sqrt(bound * bound + 1)
    result = _OUTWARD * min(error, cap) / u
    if not math.isfinite(result):
        raise ValueError(
            f"the construction error bound of {type(state).__name__} on K**d = {k}**{d}, "
            "expressed in units of u, exceeds the binary64 range. Use UniformState(), "
            "whose bound does not depend on the grid size, or fewer grid points or variables")
    return result


def chain_links(vector) -> int:
    """Return the number of links that ``append_amplitude_chain`` emits for one register (``chain_selection``).

    It is at most the index L of the last positive amplitude, since the
    chain stops once the remaining tail norm is zero, and smaller when the
    lower-range cutoff stops it earlier. Each emitted link costs one
    controlled RY, two CX in its standard lowering, and one CX, so a
    register costs 3 CX per emitted link and at most ``3 (K - 1)``.
    """
    return len(chain_selection(vector).angles)


class ChainSelection(NamedTuple):
    """The emitted prefix of one register's amplitude chain and its charge (``chain_selection``).

    Attributes:
        angles: The CRY angles of the emitted links, in link order.
        links: L, the index of the last positive amplitude, the links of
            the chain without the cutoff.
        charge: The exact charge Delta of the omitted links against the
            complete chain, zero when no link is omitted by the cutoff.
    """

    angles: tuple[float, ...]
    links: int
    charge: Fraction


def chain_selection(alpha) -> ChainSelection:
    """Return the links of one register's chain that the lower-range cutoff keeps, and the charge of the rest.

    The candidate angles are those of ``append_amplitude_chain``. Let m be
    the first link whose CRY parameter ``theta_m/2``, the exact half of the
    computed angle, lies below ``2**-1022`` (``validation._underflows``),
    including a computed angle of zero from a positive tail norm, since the
    exact angle is positive for every link up to L, the index of the last
    positive amplitude (``ChainSelection.links``). The chain then stops
    before link m and emits links 1 to m - 1, which prepare the vector with
    entries ``alpha_i`` for ``i < m - 1``, the tail norm ``r_(m-1)`` at
    ``m - 1`` and zero after it, for the stored vector alpha read exactly
    and normalized by ``N = ||alpha||``. With
    ``r_(m-1)**2 - alpha_(m-1)**2 = r_m**2``, its distance from the complete
    chain's target ``alpha/N`` is
    ``sqrt((r_(m-1) - alpha_(m-1))**2 + r_m**2)/N
    = (r_m/N) sqrt(2 r_(m-1)/(r_(m-1) + alpha_(m-1))) <= sqrt(2) r_m/N``.
    The exact sum ``L_m = sum_(i >= m) alpha_i`` is at least r_m, and
    ``N >= 1 - eta_K`` with ``eta_K = 3u/(1 - 2u) + ceil(sqrt(K)) 2**-1074``
    (``_normalized``), so ``Delta = min(sqrt2_up, sqrt2_up L_m/(1 - eta_K))``
    bounds it, where sqrt2_up is the upper rational of sqrt(2) of
    ``circuit_errors`` and also bounds the distance of two nonnegative unit
    vectors. The dropped tail norm r_m alone would undercharge, because the
    amplitude left at site ``m - 1`` changes from ``alpha_(m-1)`` to
    ``r_(m-1)`` as well. The count of omitted links is
    ``L - (m - 1)``. The native builder, the CX law, the rotation census and
    ``circuit_errors.preparation_error`` read the prefix and the charge
    here. The stored amplitudes, their construction error
    (``restricted_state_error``) and the classical start vector remain
    those of the complete vector. The work is the chain's own tail pass, and
    the exact tail sum is formed only when the cutoff applies.
    """
    from .circuit_errors import _SQRT2_UP, _TAU, _U
    from .validation import _underflows

    # The stored vector (QHDReconstruction.initial_amplitudes) as its K Python floats, read exactly.
    alpha = np.asarray(alpha, dtype=np.float64).tolist()
    tails = [0.0] * (len(alpha) + 1)
    for i in range(len(alpha) - 1, -1, -1):
        tails[i] = math.hypot(alpha[i], tails[i + 1])
    links = int(np.flatnonzero(alpha)[-1])
    angles = []
    for m in range(1, links + 1):
        theta = 2.0 * math.atan2(tails[m], alpha[m - 1])
        if _underflows(theta / 2.0, "a CRY parameter theta/2 of the structured preparation",
                       lambda theta=theta: Fraction(theta) / 2):
            k = len(alpha)
            root = math.isqrt(k) if math.isqrt(k) ** 2 == k else math.isqrt(k) + 1
            # eta_K = 3u/(1 - 2u) + ceil(sqrt(K)) * 2**-1074, with u = 2**-53.
            # The subnormal unit 2**-1074 is _TAU, and L_m = sum_(i >= m) alpha_i.
            eta = 3 * _U / (1 - 2 * _U) + root * _TAU
            tail = sum((Fraction(float(value)) for value in alpha[m:]), Fraction(0))
            # Delta = min(sqrt2_up, sqrt2_up L_m/(1 - eta_K))
            charge = min(_SQRT2_UP, _SQRT2_UP * tail / (1 - eta)) if eta < 1 else _SQRT2_UP
            return ChainSelection(tuple(angles), links, charge)
        angles.append(theta)
    return ChainSelection(tuple(angles), links, Fraction(0))


def append_amplitude_chain(circuit: QuantumCircuit, grid: OneHotGrid, amplitudes) -> None:
    """Append the linear one-hot chain that prepares ``sum_i alpha_i |e_i>`` in each register.

    ``amplitudes`` holds one nonnegative unit vector alpha per variable
    (``variable_amplitudes``), and ``|e_i>`` has register-local qubit i
    excited. The chain generalizes the standard linear W-state construction
    that prepares the uniform state of Leng et al., arXiv:2303.01471v1,
    Algorithm 1 step 3, to arbitrary nonnegative amplitudes.

    Derivation. Let ``r_m = sqrt(sum_(i >= m) alpha_i**2)`` be the tail norm,
    so r_0 = 1. X excites local qubit 0, giving ``r_0 |e_0>``. Before link m
    (m = 1..K-1) the register holds
    ``sum_(i < m-1) alpha_i |e_i> + r_(m-1) |e_(m-1)>``. The controlled RY
    with control m-1 and target m acts only on ``|e_(m-1)>``, the one
    component whose control is excited, and maps it to
    ``cos(theta_m/2) |e_(m-1)> + sin(theta_m/2) |1_(m-1) 1_m>``. The CX from
    local qubit m onto m-1 turns the second state into ``|e_m>`` and leaves
    every state with qubit m unexcited unchanged. With
    ``cos(theta_m/2) = alpha_(m-1)/r_(m-1)`` and
    ``sin(theta_m/2) = r_m/r_(m-1)`` the register then holds
    ``sum_(i < m) alpha_i |e_i> + r_m |e_m>``, and after link K-1 it holds
    the target, since ``r_(K-1) = alpha_(K-1)``. Both ratios are nonnegative,
    so ``theta_m = 2 atan2(r_m, alpha_(m-1))`` in ``[0, pi]``, which avoids
    the division and keeps full relative accuracy when r_m is small.

    Zero tail. If ``r_m = 0``, the register already holds the target before
    link m, because then ``r_(m-1) = alpha_(m-1)``. Link m has theta_m = 0
    and every later link has an unexcited control on all components, so
    they act as the identity for any angle, while their angles are the
    undefined 0/0. The chain therefore stops after link L, the index of the
    last positive amplitude (``ChainSelection.links``), and emits 3L CX. For
    the uniform state L = K - 1, which gives the ``3 d (K - 1)`` CX of the
    full chain.

    Accuracy. The tail norms come from the recurrence
    ``r_m = hypot(alpha_m, r_(m+1))``, so a normal r_m has relative error at
    most ``2 (K - m) u``, and atan2 adds one ulp of an angle in
    ``[0, pi/2]``. A relative error rho of r_m moves ``atan2(r_m, alpha)`` by
    at most rho/2, so ``|theta_m - theta_hat_m| <= (2 (K - m) + 2 pi) u``.
    The controlled RY of link m rotates the component of norm r_(m-1), and
    the derivative of the rotation with respect to its angle has norm 1/2,
    so link m moves the register by at most ``r_(m-1) |theta_m -
    theta_hat_m|/2``. With normal tail norms the prepared register therefore
    differs from the exact chain for the supplied vector by at most
    ``sum_m ((K - m) + pi) u = (K (K - 1)/2 + pi (K - 1)) u``. A subnormal
    tail norm errs by at most ``(K - m) 2**-1074`` absolutely, and since
    ``d atan2(r, alpha)/dr = alpha/r_(m-1)**2``, its link moves the register
    by at most that amount, below the neglected second-order terms. The
    chain prepares the supplied vector's direction, whose norm differs from
    one by at most 3u (``_normalized``). No 2**K amplitude vector is formed.

    Lower-range cutoff. A link whose parameter ``theta_m/2`` would fall
    below the normal binary64 range ends the chain before it, and the
    omitted links are charged to the preparation entry of the error ledger
    (``chain_selection``).
    """
    for var_index, alpha in enumerate(amplitudes):
        circuit.x(grid.qubit(var_index, 0))
        for m, theta in enumerate(chain_angles(alpha), start=1):
            control = grid.qubit(var_index, m - 1)
            target = grid.qubit(var_index, m)
            circuit.cry(theta, control, target)
            circuit.cx(target, control)


def chain_angles(alpha) -> list[float]:
    """Return the CRY angles that ``append_amplitude_chain`` emits for one register.

    ``theta_m = 2 atan2(r_m, alpha_(m-1))`` with the tail norms
    ``r_m = hypot(alpha_m, r_(m+1))``, so ``cos(theta_m/2) = alpha_(m-1)/r_(m-1)``
    and ``sin(theta_m/2) = r_m/r_(m-1)`` (``append_amplitude_chain`` derives
    them), for the links that ``chain_selection`` keeps. The chain and the
    rotation law of its circuit (``resources.rotation_population``) both
    read the angles here.
    """
    return list(chain_selection(alpha).angles)


def onehot_register_vector(vector) -> np.ndarray:
    """Return the one-hot embedding of one register's amplitudes, a vector of 2**K entries.

    Amplitude alpha_i sits at the basis index 2**i, register-local qubit i
    excited, and every other entry is zero.
    """
    state = np.zeros(1 << len(vector), dtype=complex)
    for grid_index, amplitude in enumerate(vector):
        state[1 << grid_index] = amplitude
    return state


def append_register_state_preparation(circuit: QuantumCircuit, grid: OneHotGrid, amplitudes) -> None:
    """Append one Qiskit ``StatePreparation`` per variable register with its one-hot embedding.

    Each block receives the ``2**K`` vector of ``onehot_register_vector``,
    so this recipe forms and synthesizes one dense register vector per
    variable. Qiskit's synthesis has no CX bound in the QHD record.
    """
    from qiskit.circuit.library import StatePreparation

    for var_index, vector in enumerate(amplitudes):
        qubits = [
            circuit.qubits[grid.qubit(var_index, grid_index)]
            for grid_index in range(grid.num_grid_points)
        ]
        circuit.append(StatePreparation(onehot_register_vector(vector)), qubits)


def append_initial_state(circuit: QuantumCircuit, grid: OneHotGrid, amplitudes, *, recipe: str) -> None:
    """Append the QHD initial state with the selected native recipe.

    ``recipe`` is the Method's ``initial_state_preparation``. On this one-hot
    register ``"structured"`` appends the amplitude chain
    (``append_amplitude_chain``) and ``"qiskit_state_preparation"`` one
    ``StatePreparation`` per register. ``"none"`` is resource-only and is
    rejected before circuit construction.
    """
    if recipe == "qiskit_state_preparation":
        append_register_state_preparation(circuit, grid, amplitudes)
    else:
        append_amplitude_chain(circuit, grid, amplitudes)


def append_binary_initial_state(circuit: QuantumCircuit, grid: OneHotGrid, bits: int, amplitudes, *,
                                recipe: str) -> None:
    """Append the QHD initial state on the binary registers with the selected native recipe.

    Variable j occupies qubits ``j b`` to ``j b + b - 1`` with grid index
    ``n_j = sum_l 2**l n_(j,l)`` (``binary`` module). ``"structured"``
    applies H to every one of the ``d b`` qubits. ``H|0> = (|0> + |1>)/sqrt(2)``,
    so the product is ``2**(-d b/2) = K**(-d/2)`` on every basis state, which
    is exactly the uniform state with no CX and no rotation. It prepares the
    uniform state and the kinetic ground state of the periodic grid, which
    is the same state, and ``method.QHD.plan`` rejects it for any other state.
    ``"qiskit_state_preparation"`` appends one Qiskit ``StatePreparation`` of
    the stored K amplitudes of each variable to its register. The basis index
    of the register's qubits in Qiskit's little-endian order is the grid
    index, so the vector needs no reordering. Qiskit's synthesis has no CX
    bound in the QHD record. ``"none"`` is resource-only and is rejected
    before circuit construction.
    """
    if recipe == "structured":
        circuit.h(range(grid.num_variables * bits))
        return
    from qiskit.circuit.library import StatePreparation

    for var_index, vector in enumerate(amplitudes):
        qubits = circuit.qubits[var_index * bits : (var_index + 1) * bits]
        circuit.append(StatePreparation(np.asarray(vector, dtype=complex)), qubits)
