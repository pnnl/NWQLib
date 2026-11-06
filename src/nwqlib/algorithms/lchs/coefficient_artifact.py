"""Fixed-node LCHS coefficient weights from one explicit, bounded linear program.

The finite LCHS sum sum_j w_j exp(-i t (kappa_j L_s + H)) approximates
exp(-t (L_s + i H)) for a PSD L_s and Hermitian H that need not commute. Ni
and Ying, arXiv:2606.29848v1, Section 6.1, Eqs. (72)-(74), bound its operator
error by the local Wiener norm on [0, b] of the scalar residual

    eta(x) = sum_j w_j exp(-i kappa_j x) - exp(-x),

for every t with t*||L_s|| <= b. The norm is not finite-dimensional, and a
finite cosine sum alone does not bound it. NWQLib closes the tail as
follows. With the cosine coefficients a_m = (2/b) int_0^b eta(x) cos(m pi x/b) dx
of the even extension, two integrations by parts give
|a_m| <= (2b/(pi^2 m^2)) (|eta'(0)| + |eta'(b)| + int_0^b |eta''|), and
sum_{m>M} m^-2 <= 1/M with int_0^b |eta''| <= b sum_j kappa_j^2 |w_j| + 1 - exp(-b)
gives, for M >= 1,

    ||eta||_W <= |a_0|/2 + sum_{m=1}^M |a_m|
                 + (2b/(pi^2 M)) (|eta'(0)| + |eta'(b)| + b sum_j kappa_j^2 |w_j| + 1 - exp(-b)).

Every a_m, eta'(0) and eta'(b) is affine in the complex weights. With the
dimensionless phases z_+ = m pi - kappa b and z_- = -m pi - kappa b,
(2/b) int_0^b exp(-i kappa x) cos(m pi x/b) dx = E(z_+) + E(z_-), where
E(z) = exp(i z/2) sin(z/2)/(z/2) is finite at the resonance z = 0. The
phases avoid the frequency m pi/b, which can overflow for a short support.
The target coefficients C_m = (2/b) int_0^b exp(-x) cos(m pi x/b) dx are

    C_0 = -2 expm1(-b)/b,    C_m = 2b (1 - (-1)^m exp(-b))/(b^2 + (m pi)^2) for m >= 1,

evaluated with -expm1(-b) for even m and 1 + exp(-b) for odd m. Forming
1 - exp(-b) instead cancels for small b, and then zero weights, whose error
at t = 0 is one, would receive a residual bound near zero.

`prepare_lchs_discrete_weights` keeps the nodes fixed and minimizes the
coefficient mass beta = sum_j |w_j| subject to this bound, a convex problem
in the weights. SciPy's HiGHS solves a linear program in which each complex
modulus |z| is replaced by its polygon bound. With D normal directions
theta_l = 2 pi l/D and kappa_D = cos(pi/D), h_D(z) = max_l Re(exp(-i theta_l) z)
satisfies kappa_D |z| <= h_D(z) <= |z|, so the constraints
Re(exp(-i theta_l) z) <= kappa_D t for all l imply |z| <= t. These constraints
are conservative and can reject weights that meet the exact bound. The
polygon values lie between the true values and the true values divided by
kappa_D, and every modulus enters with a nonnegative coefficient, so the
weights optimal for the exact problem at target kappa_D eps' are feasible
here. The returned weights therefore have
beta(w) <= B*(kappa_D eps')/kappa_D, with B*(e) the exact optimal mass at
target e and eps' the linear program's target. Neither the solver's success
nor this relation qualifies the result. After solving, the bound is
evaluated again from the weights in the returned artifact with exact complex
moduli, and the preparation fails unless that value plus its rounding
estimate meets the allowance.

Forming LCHS coefficients from an artifact reads its nodes, weights and
bound, imports no solver, and recomputes neither the optimization nor the
residual.
"""

from __future__ import annotations

from math import ceil, cos, exp, expm1, fsum, isfinite, pi
from time import perf_counter
from typing import Literal

import numpy as np

from nwqlib._validation import finite_real, integer
from nwqlib.core.records import StrictInteger, PositiveInt, Real, Record, Text
from nwqlib.operators.access import _check_bytes

# The linear program enforces its conservative residual bound at this fraction
# of the allowance. The rest absorbs HiGHS's primal feasibility tolerance,
# which a polygon vertex reaches with |z| = t, and the binary64 rounding of the
# bound evaluated afterwards from the weights.
_LP_TARGET_FRACTION = 15 / 16
# HiGHS's smallest admitted primal feasibility tolerance. The absolute row
# violations it permits bound the excess of the true residual over the
# linear program's target.
_HIGHS_FEASIBILITY_TOLERANCE = 1e-10
# Constant added to J + M in the standard-model rounding estimate of the
# residual bound. It covers the few roundings of each exponential, sinc,
# product and modulus that enter one term.
_ROUNDING_TERM_COUNT = 16
_UNIT_ROUNDOFF = 2.0**-53


class LCHSDiscreteWeightResidual(Record):
    """Operator-error bound of prepared LCHS weights, evaluated from those weights.

    Each term enters the local Wiener-norm bound on [0, b] described in
    [`prepare_lchs_discrete_weights`][nwqlib.algorithms.lchs.coefficient_artifact.prepare_lchs_discrete_weights].
    All values are binary64 evaluations with exact complex moduli, not
    polygon bounds or solver output.

    Attributes:
        support_endpoint: Right endpoint b > 0 of the scalar interval. The
            bound holds for exp(-t (L_s + i H)) whenever t*||L_s|| <= b.
        residual_modes: Number M >= 1 of cosine coefficients summed exactly.
        allowance: Operator-norm allowance that the preparation was asked
            to meet.
        constant_term: |a_0|/2.
        cosine_terms: Sum of |a_m| for m = 1..M.
        slope_at_zero: |eta'(0)|.
        slope_at_endpoint: |eta'(b)|.
        curvature_mass: b*sum_j kappa_j**2*|w_j| + 1 - exp(-b), the bound
            on the integral of |eta''| over [0, b].
        tail_bound: (2b/(pi**2 M))*(slope_at_zero + slope_at_endpoint +
            curvature_mass), the bound on the cosine coefficients beyond M.
        residual_bound: constant_term + cosine_terms + tail_bound, the
            operator-norm bound for a unit input.
        rounding_estimate: Standard-model estimate (J + M + 16)*u*S of the
            binary64 error in residual_bound, with u = 2**-53, J nodes and S
            the sum of the magnitudes of all terms entering it. It is an
            estimate, not an outward-rounded enclosure. The preparation
            requires residual_bound + rounding_estimate <= allowance.
    """

    support_endpoint: Real
    residual_modes: PositiveInt
    allowance: Real
    constant_term: Real
    cosine_terms: Real
    slope_at_zero: Real
    slope_at_endpoint: Real
    curvature_mass: Real
    tail_bound: Real
    residual_bound: Real
    rounding_estimate: Real


class LCHSDiscreteWeightPreparation(Record):
    """Solver settings, outcome and measured cost of one weight preparation.

    Attributes:
        solver: The solver call, `scipy.optimize.linprog` with method `highs`.
        scipy_version: SciPy version that ran it.
        arithmetic: Floating-point format of the program, the solver and the
            residual evaluation, `"binary64"`.
        polygon_directions: Number D of normal directions per complex modulus.
        polygon_factor: cos(pi/D), the factor kappa_D of the conservative
            constraints.
        lp_residual_target: Residual target eps' of the linear program,
            15/16 of the allowance.
        feasibility_tolerance: HiGHS primal feasibility tolerance, 1e-10.
        time_limit_seconds: Solver time limit given by the caller.
        status: HiGHS status code returned by linprog, 0 for success.
        message: Solver message.
        iterations: Solver iteration count.
        lp_objective: Optimal objective, the polygon bound on the
            coefficient mass. It is at least the true mass.
        rows: Inequality rows of the program.
        columns: Variables of the program.
        nonzeros: Stored entries of the constraint matrix.
        charged_bytes: Bytes of the largest set of NumPy arrays the
            preparation holds at once, charged before any of them was
            built. Python objects and the workspace inside `linprog` and
            HiGHS are not included.
        solve_seconds: Measured wall-clock time of the solver call.
    """

    solver: Text
    scipy_version: Text
    arithmetic: Literal["binary64"] = "binary64"
    polygon_directions: PositiveInt
    polygon_factor: Real
    lp_residual_target: Real
    feasibility_tolerance: Real
    time_limit_seconds: Real
    status: StrictInteger
    message: Text
    iterations: StrictInteger
    lp_objective: Real
    rows: PositiveInt
    columns: PositiveInt
    nonzeros: PositiveInt
    charged_bytes: PositiveInt
    solve_seconds: Real


class LCHSDiscreteWeightArtifact(Record):
    """Prepared complex LCHS weights at fixed real nodes, with their operator-error bound.

    [`prepare_lchs_discrete_weights`][nwqlib.algorithms.lchs.coefficient_artifact.prepare_lchs_discrete_weights]
    returns it. The finite sum sum_j w_j exp(-i t (kappa_j L_s + H)) over
    its `nodes` kappa_j and `weights` w_j approximates exp(-t (L_s + i H))
    within `residual.residual_bound` in operator norm whenever
    t*||L_s|| <= `residual.support_endpoint`, for PSD L_s and Hermitian H
    that need not commute. The record is immutable, its `content_id`
    identifies it, and `model_dump_json` and `model_validate_json` save and
    reload it without changing any number. Using it runs no solver.

    Attributes:
        nodes: Real nodes kappa_j (dimensionless), in address order.
        weights_real: Real parts of the weights w_j, in node order.
        weights_imag: Imaginary parts of the weights w_j, in node order.
        coefficient_mass: sum_j |w_j|, the normalization of the coefficient
            state and the beta of the recovery and sampling costs.
        residual: The operator-error bound and its terms.
        preparation: Solver settings, outcome and measured cost.
    """

    nodes: tuple[Real, ...]
    weights_real: tuple[Real, ...]
    weights_imag: tuple[Real, ...]
    coefficient_mass: Real
    residual: LCHSDiscreteWeightResidual
    preparation: LCHSDiscreteWeightPreparation

    @property
    def weights(self) -> tuple[complex, ...]:
        """Complex weights w_j in node order."""
        return tuple(complex(real, imag) for real, imag in zip(self.weights_real, self.weights_imag, strict=True))


def _residual_affine_maps(nodes: np.ndarray, support: float, modes: int):
    """Return the affine maps from the weights to a_0..a_M, eta'(0) and eta'(b).

    Row k of the returned complex matrix G and entry k of the vector g give
    z_k = G[k] @ w + g[k]: a_m for k = m <= M, eta'(0) for k = M+1 and
    eta'(b) for k = M+2. The moments and the targets C_m use the
    dimensionless phases and the expm1 numerators of the module description.
    """

    def interval_mean(phase):
        # (1/b) int_0^b exp(i nu x) dx = exp(i z/2) sin(z/2)/(z/2) with z = nu*b; np.sinc(x) = sin(pi x)/(pi x).
        return np.exp(0.5j * phase) * np.sinc(phase / (2 * pi))

    mode_phases = np.arange(modes + 1)[:, None] * pi
    node_phases = nodes * support
    matrix = np.empty((modes + 3, nodes.size), dtype=np.complex128)
    matrix[:modes + 1] = interval_mean(mode_phases - node_phases) + interval_mean(-mode_phases - node_phases)
    matrix[modes + 1] = -1j * nodes
    matrix[modes + 2] = -1j * nodes * np.exp(-1j * node_phases)
    mode_numbers = np.arange(1, modes + 1)
    numerators = np.where(mode_numbers % 2 == 0, -expm1(-support), 1.0 + exp(-support))
    offsets = np.empty(modes + 3, dtype=np.complex128)
    offsets[0] = 2.0 * expm1(-support) / support
    offsets[1:modes + 1] = -2.0 * support * numerators / (support * support + (mode_numbers * pi) ** 2)
    offsets[modes + 1] = 1.0
    offsets[modes + 2] = exp(-support)
    return matrix, offsets


def _residual_record(nodes, weights, support, modes, allowance):
    """Evaluate every term of the residual bound from the weights, with exact complex moduli."""
    matrix, offsets = _residual_affine_maps(nodes, support, modes)
    values = matrix @ weights + offsets
    magnitudes = np.abs(weights)
    coefficient = np.ones(modes + 1)
    coefficient[0] = 0.5
    tail_factor = 2.0 * support / (pi * pi * modes)
    curvature = support * fsum(nodes * nodes * magnitudes) + -expm1(-support)
    constant_term = 0.5 * float(abs(values[0]))
    cosine_terms = fsum(np.abs(values[1:modes + 1]))
    slope_zero, slope_end = float(abs(values[modes + 1])), float(abs(values[modes + 2]))
    # ponytail: the integration-by-parts tail is loose for large b or |kappa_j| and sets an allowance floor
    # near 2b(1-exp(-b))/(pi^2 M); a tighter tail with the same finite contract would admit smaller allowances.
    tail = tail_factor * (slope_zero + slope_end + curvature)
    bound = constant_term + cosine_terms + tail
    # Magnitudes of every term entering the bound, for the standard-model estimate.
    term_scale = (fsum(coefficient * (np.abs(matrix[:modes + 1]) @ magnitudes + np.abs(offsets[:modes + 1])))
                  + tail_factor * (np.abs(matrix[modes + 1:]) @ magnitudes).sum()
                  + tail_factor * (1.0 + exp(-support) + curvature))
    rounding = (nodes.size + modes + _ROUNDING_TERM_COUNT) * _UNIT_ROUNDOFF * term_scale
    return LCHSDiscreteWeightResidual(
        support_endpoint=support, residual_modes=modes, allowance=allowance, constant_term=constant_term,
        cosine_terms=cosine_terms, slope_at_zero=slope_zero, slope_at_endpoint=slope_end,
        curvature_mass=curvature, tail_bound=tail, residual_bound=bound, rounding_estimate=float(rounding))


def prepare_lchs_discrete_weights(*, nodes, support_endpoint, residual_allowance, residual_modes,
                                  polygon_directions, solver_time_limit,
                                  max_bytes=None) -> LCHSDiscreteWeightArtifact:
    """Optimize LCHS weights at fixed nodes under a qualified operator-error bound, as one explicit preparation.

    The weights minimize the coefficient mass sum_j |w_j| subject to the
    local Wiener-norm bound on [0, b] of the module description, which
    bounds the operator error of the finite LCHS sum for every elapsed time
    t with t*||L_s|| <= b, also when L_s and H do not commute. The bound
    uses the first `residual_modes` cosine coefficients exactly and an
    analytic bound for the rest, which is
    at least 2b(1 - exp(-b))/(pi**2 M) for any weights, so the allowance
    must exceed that constant. SciPy's HiGHS solves the linear program with
    conservative polygon moduli at 15/16 of the allowance. The bound is then
    evaluated again from the returned weights with exact moduli. No second
    attempt or other node set is tried.

    The byte charge is checked before any array of the program is built,
    and ``solver_time_limit`` bounds the solve. With J nodes, M modes, D directions and K = M + 3 moduli of
    affine maps, the program has R = D*(J + K) + 1 rows, 3J + K columns and
    N = D*(3J + K*(2J + 1)) + K + J matrix entries. The byte charge is the
    largest set of NumPy arrays alive at once, which occurs while SciPy
    converts the coordinate-format constraints to compressed rows:
    16*K*J + 24*S + 40*N + 8*(R + 1) + 80*D*J + 8*D*K + 32*(J + K) + 24*D
    bytes with S = D*K*(2J + 1). Python objects and the workspace inside
    `linprog` and HiGHS are not charged.

    Args:
        nodes (Sequence[float]): Fixed real nodes kappa_j, at least one,
            in the address order of the coefficient state. For example the
            nodes of an analytic LCHS rule, or a uniform grid.
        support_endpoint (float): b > 0. Use b >= T_e*B_Ls, the largest
            elapsed time times the bound on ||L_s|| of the problems that
            will use the weights.
        residual_allowance (float): Operator-norm allowance in (0, 1) for a
            unit input.
        residual_modes (int): Number M >= 1 of cosine coefficients summed
            exactly. The program grows linearly in M.
        polygon_directions (int): Number D >= 3 of normal directions of
            each polygon modulus. The constraints lose at most a factor
            cos(pi/D) of the feasible residual.
        solver_time_limit (float): Positive time limit of the solver in
            seconds.
        max_bytes (int | None): Default `None`, the physical memory of this computer.
            Upper limit on the charged known-array bytes.

    Returns:
        artifact (LCHSDiscreteWeightArtifact): The nodes, the weights, their
            coefficient mass, the bound with its terms and rounding estimate,
            and the solver record.

    Raises:
        ValueError: If an input is outside its domain, if the allowance is
            not above the tail constant at this M, if the byte charge exceeds
            ``max_bytes``, if HiGHS ends without an optimal point (its status and
            message are reported), or if the bound evaluated from the
            returned weights plus its rounding estimate exceeds the
            allowance. Each message gives the actual quantities.
    """
    from scipy import __version__ as scipy_version
    from scipy.optimize import linprog
    from scipy.sparse import csr_array

    kappa = np.asarray(nodes, dtype=float)
    if kappa.ndim != 1 or kappa.size == 0 or not np.isfinite(kappa).all():
        raise ValueError("nodes must be a nonempty one-dimensional sequence of finite reals")
    support = finite_real(support_endpoint, "support_endpoint")
    allowance = finite_real(residual_allowance, "residual_allowance")
    modes = integer(residual_modes, "residual_modes", 1)
    directions = integer(polygon_directions, "polygon_directions", 3)
    time_limit = finite_real(solver_time_limit, "solver_time_limit")
    if support <= 0 or not 0 < allowance < 1 or time_limit <= 0:
        raise ValueError("support_endpoint and solver_time_limit must be positive and residual_allowance "
                         "must lie in (0, 1)")
    count, terms = kappa.size, modes + 3
    tail_factor = 2.0 * support / (pi * pi * modes)
    tail_constant = tail_factor * -expm1(-support)
    target = _LP_TARGET_FRACTION * allowance
    if target <= tail_constant:
        raise ValueError(
            f"the tail bound alone is at least {tail_constant!r} at residual_modes={modes}, above the linear "
            f"program's target {target!r} for residual_allowance={allowance!r}. Raise residual_modes to at "
            f"least {ceil(2 * support * -expm1(-support) / (pi * pi * target))} or the allowance")
    rows = directions * (count + terms) + 1
    columns = 3 * count + terms
    term_entries = directions * terms * (2 * count + 1)
    nonzeros = 3 * directions * count + term_entries + terms + count
    # Peak known arrays, at the CSR conversion inside csr_array below, with 8-byte reals and integers and
    # 16-byte complex values. Live then are the nodes and j (16J), the affine maps (16KJ + 16K), the angles
    # and rotations (24D), weight_rows and weight_block (80DJ), term_rows (8DK), the three term arrays
    # (24S), the budget row (16(K + J)), the three coordinate arrays (24N) and SciPy's CSR data, indices
    # and row pointers (16N + 8(R + 1)). `rotated` (16DKJ) is released before this point. Every other stage
    # holds fewer bytes for J >= 1, K >= 4 and D >= 3. Forming term_values holds rotated and its negated
    # imaginary part (24DKJ) but none of the 40N >= 80DKJ coordinate and CSR bytes. The assembly arrays are
    # released before the solve, which then holds the CSR arrays, the right side and the objective, and the
    # residual evaluated afterwards adds O(KJ + R + C) bytes to those.
    charged_bytes = (16 * terms * count + 24 * term_entries + 40 * nonzeros + 8 * (rows + 1)
                     + 80 * directions * count + 8 * directions * terms + 32 * (count + terms) + 24 * directions)
    _check_bytes(charged_bytes, max_bytes, "LCHS discrete-weight linear program")

    matrix, offsets = _residual_affine_maps(kappa, support, modes)
    polygon_factor = cos(pi / directions)
    angles = 2 * pi * np.arange(directions) / directions
    rotation = np.exp(-1j * angles)
    # Variables: u = Re w (J), v = Im w (J), s_j >= |w_j| (J), t_k >= |z_k| (K).
    # Re(exp(-i theta) (G w + g)) = Re(exp(-i theta) G) u - Im(exp(-i theta) G) v + Re(exp(-i theta) g).
    j = np.arange(count)
    weight_rows = (np.arange(directions)[:, None] * count + j[None, :]).ravel()
    weight_block = (
        np.concatenate([weight_rows] * 3),
        np.concatenate([np.tile(j, directions), count + np.tile(j, directions), 2 * count + np.tile(j, directions)]),
        np.concatenate([np.repeat(np.cos(angles), count), np.repeat(np.sin(angles), count),
                        np.full(directions * count, -polygon_factor)]),
    )
    rotated = rotation[:, None, None] * matrix[None, :, :]
    term_rows = directions * count + (np.arange(directions)[:, None] * terms + np.arange(terms)[None, :])
    term_row_index = np.repeat(term_rows.ravel(), 2 * count + 1)
    term_columns = np.tile(np.concatenate([j, count + j, [0]]), directions * terms).reshape(directions, terms, -1)
    term_columns[:, :, -1] = 3 * count + np.arange(terms)[None, :]
    term_values = np.concatenate([rotated.real, -rotated.imag,
                                  np.full((directions, terms, 1), -polygon_factor)], axis=2)
    del rotated  # Not needed again, and the byte charge assumes it is gone before the CSR conversion.
    budget = np.concatenate([np.ones(terms), tail_factor * support * kappa * kappa]) / allowance
    budget[0] = 0.5 / allowance
    budget[modes + 1] = budget[modes + 2] = tail_factor / allowance
    budget_columns = np.concatenate([3 * count + np.arange(terms), 2 * count + j])
    constraint = csr_array(
        (np.concatenate([weight_block[2], term_values.ravel(), budget]),
         (np.concatenate([weight_block[0], term_row_index, np.full(terms + count, rows - 1)]),
          np.concatenate([weight_block[1], term_columns.ravel(), budget_columns]))),
        shape=(rows, columns))
    right_side = np.concatenate([
        np.zeros(directions * count),
        -(rotation[:, None] * offsets[None, :]).real.ravel(),
        [(target - tail_constant) / allowance],
    ])
    objective = np.concatenate([np.zeros(2 * count), np.ones(count), np.zeros(terms)])
    # The solve and the residual evaluation need none of the assembly arrays, and the byte charge assumes
    # they are released here.
    del (matrix, offsets, weight_rows, weight_block, term_rows, term_row_index, term_columns, term_values,
         budget, budget_columns)
    bounds = [(None, None)] * (2 * count) + [(0, None)] * (count + terms)
    started = perf_counter()
    result = linprog(objective, A_ub=constraint, b_ub=right_side, bounds=bounds, method="highs",
                     options={"time_limit": time_limit, "primal_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
                              "presolve": True, "disp": False})
    seconds = perf_counter() - started
    if result.status != 0 or result.x is None:
        raise ValueError(f"HiGHS ended with status {result.status} after {seconds:.3g} s and {result.nit} "
                         f"iterations: {result.message}")
    # The artifact holds these binary64 values (a -0.0 part becomes 0.0, the
    # same number), so the bound below is the bound of the returned weights.
    weights = result.x[:count] + 1j * result.x[count:2 * count]
    residual = _residual_record(kappa, weights, support, modes, allowance)
    qualified = residual.residual_bound + residual.rounding_estimate
    if not isfinite(qualified) or qualified > allowance:
        raise ValueError(
            f"the residual bound of the HiGHS weights is {residual.residual_bound!r} with rounding estimate "
            f"{residual.rounding_estimate!r}, above residual_allowance={allowance!r}. The polygon constraints "
            f"were met to HiGHS's tolerance, but the bound from the weights is the qualification")
    return LCHSDiscreteWeightArtifact(
        nodes=tuple(float(value) for value in kappa),
        weights_real=tuple(float(value) for value in weights.real),
        weights_imag=tuple(float(value) for value in weights.imag),
        coefficient_mass=fsum(np.abs(weights)),
        residual=residual,
        preparation=LCHSDiscreteWeightPreparation(
            solver="scipy.optimize.linprog(method='highs')", scipy_version=scipy_version,
            polygon_directions=directions, polygon_factor=polygon_factor, lp_residual_target=target,
            feasibility_tolerance=_HIGHS_FEASIBILITY_TOLERANCE, time_limit_seconds=time_limit,
            status=int(result.status), message=str(result.message), iterations=int(result.nit),
            lp_objective=float(result.fun), rows=rows, columns=columns, nonzeros=nonzeros,
            charged_bytes=charged_bytes, solve_seconds=seconds))


__all__ = [
    "LCHSDiscreteWeightArtifact",
    "LCHSDiscreteWeightPreparation",
    "LCHSDiscreteWeightResidual",
    "prepare_lchs_discrete_weights",
]
