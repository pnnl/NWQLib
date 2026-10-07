"""Portable records of the QHD augmented-Lagrangian layer.

The layer (``constrained.py``) solves a ``ConstrainedOptimization`` by a
sequence of box problems, each one a new QHD Plan. These records hold its
options, the constraint preprocessing, each round and the whole run. They are
typed, immutable and carry content identities. Their validators check
mathematical domains and the links between records. The rules that produce
the values live in ``constrained.py``.

Sources named here and in ``constrained.py``: Wu et al. arXiv:2605.12066v1,
Eqs. (6)-(8), Rockafellar doi:10.1007/BF01580138 (the PHR inequality term), and
Birgin and Martinez, *Practical Augmented Lagrangian Methods for Constrained
Optimization*, SIAM 2014, doi:10.1137/1.9781611973365 (Algorithm 4.1,
Eqs. (4.7)-(4.9), the effective objective of Eq. (10.3), Algencan's stopping
test, Eqs. (10.6)-(10.8), and the assumptions of the book's convergence
results in Chapters 5-7).
"""

from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from nwqlib._limits import DEFAULT_MAX_BYTES
from nwqlib.core.records import ContentID, Nonnegative, PositiveInt, Real, Record, Text
from nwqlib.execution import ExecutionLimits
from nwqlib.operators.access import Count
from nwqlib.problems.records import ConstrainedOptimization
from ._outer import refinement_stop
from .method import QHD
from .records import ModeStatus
from .refinement_records import BoxRefinement, BoxRefinementResult

PositiveReal = Annotated[Real, Field(gt=0)]

# Feasibility tolerance on the normalized residuals when a problem has only
# inequality constraints. The value is the constraint tolerance of Wu et al.
# arXiv:2605.12066v1, Sec. VI.B. It classifies the normalized residual of a
# returned point and claims no accuracy of the solution. A grid point
# strictly inside the feasible set has zero violation, and a grid point on
# the boundary can have a rounding residual that a positive tolerance
# accepts. Exact rational enumeration of x**2 + y**2 - 1 on the interior
# grids of [-1, 1]**2 with K = 16, 32 and 64 gives the smallest positive
# residuals 1/289, 1/1089 and 9/4225, far above both 1e-9 and 1e-6, so the
# two values classify these grids alike.
#
# Measured on 2026-09-26 (Python 3.12.14, NumPy 2.5.2,
# SciPy 1.18.1, SymPy 1.14.0, macOS arm64), refined runs with the
# ``most_probable`` point rule stopped at the same points and rounds under
# 1e-9 and 1e-6: the unit disk f = (x - 1)**2 + (y - 1)**2,
# g = x**2 + y**2 - 1 (K = 16 on [-1, 1]**2) at round 7, and constrained
# Rastrigin, Wu et al. Eqs. (17)-(18) (K = 32 on [-5, 5]**2), at round 10.
# Each ran with the Schrodinger flavor at 200 steps and the split-step
# flavor at 3,200 steps, classical execution on the Dirichlet interior grid,
# KineticGroundState, QuadraticSchedule(gamma=0.3), total time 10, and
# refinement in every round with search_model, gain 8, at most three levels
# and mass threshold 0.99. A larger explicit tolerance matters where a
# point can have a small positive violation, such as the off-grid mean of
# ``mode_or_mean``. In the same measurement, the refined disk run with
# ``mode_or_mean`` as both the inner point and the refinement point rule
# (midpoint coefficients, seed 11, max_iterations=15, both tolerances equal)
# reached the limit of 15 rounds with violation 1.04e-9 under 1e-9 and
# stopped after 11 rounds with violation 2.9e-7 under 1e-6, under both
# flavors. The stop under 1e-6 was read from the saved rounds of the run
# under 1e-9, since the tolerances enter only the stopping test
# (``constrained.solve_augmented_lagrangian``), and a direct Schrodinger run
# under 1e-6, on the same day in the same environment, stopped after the
# same 11 rounds with the same violation. The guide's example
# (docs/algorithms/qhd.md,
# "Constrained problems") is this run. With equality constraints there is
# no default, because the residual that a finite grid can reach depends on
# the grid and the constraint (docs/algorithms/qhd.md, "Constrained
# problems"). Registered in docs/ENGINEERING_CONSTANTS.md. Revisit if a
# problem class with the ``most_probable`` point rule stops at different
# points under 1e-9 and 1e-6.
DEFAULT_INEQUALITY_FEASIBILITY_TOLERANCE = 1e-9

# Limits of the explicit constrained_grid_minimum reference, the same
# defaults as QHDVerification. Registered in docs/ENGINEERING_CONSTANTS.md.
# Revisit with a deliberately larger reference grid.
GRID_MINIMUM_MAX_WORK = 1_000_000_000
GRID_MINIMUM_MAX_BYTES = DEFAULT_MAX_BYTES

# Run statuses that only a round's box refinement can cause: it completed no
# level for one of these reasons (_outer.refinement_stop).
_REFINEMENT_ONLY = ("flat_objective", "unresolved_objective", "width_floor")


class MultiplierBounds(Record):
    """Safeguard box for the multipliers, Algorithm 4.1 Step 4 of Birgin and Martinez, doi:10.1137/1.9781611973365.

    Build it with keyword arguments, for example
    `MultiplierBounds(equality_lower=-10.0, equality_upper=10.0, inequality_upper=10.0)`,
    and pass it as `AugmentedLagrangian(multiplier_bounds=...)`. All three arguments are
    required. The next round uses `clip(lambda+, equality_lower, equality_upper)` for
    every equality and `min(mu+, inequality_upper)` for every inequality, where lambda+
    and mu+ are the tentative multipliers of Eqs. (4.7)-(4.8). The bounds are in
    normalized units, the units of the multipliers of the scaled problem.

    Algorithm 4.1 requires `lambda_min < lambda_max` and `mu_max > 0`. Its Step 4 keeps
    the multipliers of every round bounded, so that the shifts lambda-bar/rho and
    mu-bar/rho tend to zero when rho grows (p. 35), and the book's convergence proofs
    use that boundedness, for example the proof of Theorem 5.1 (pp. 41-42). The book's
    analysis of the penalty in Chapter 7 also assumes that the true multipliers lie
    inside the bounds, away from lambda_min, lambda_max and mu_max,
    `lambda* in (lambda_min, lambda_max)` and `mu* in [0, mu_max)` (Assumption 7.7, Sec.
    7.5, p. 64), which no default can know. Without that assumption the global
    convergence results of Chapter 6 still hold, but the penalty may no longer stay
    bounded (p. 64).

    Attributes:
        equality_lower: Required. lambda_min, the lower bound of every equality
            multiplier.
        equality_upper: Required. lambda_max, greater than `equality_lower`.
        inequality_upper: Required. mu_max, the positive upper bound of every inequality
            multiplier. The lower bound is zero.

    Raises:
        ValueError: If `equality_lower` is not below `equality_upper`.
    """

    equality_lower: Real
    equality_upper: Real
    inequality_upper: PositiveReal

    @model_validator(mode="after")
    def _interval(self):
        if not self.equality_lower < self.equality_upper:
            raise ValueError("multiplier bounds need equality_lower < equality_upper (Algorithm 4.1)")
        return self


class AugmentedLagrangian(Record):
    """Options of `solve_augmented_lagrangian`: penalty schedule, scales, tolerances, point rule and stopping test.

    Build it with keyword arguments, for example
    `AugmentedLagrangian(max_iterations=20, objective_scale=10.0)`, and pass it as
    `solve_augmented_lagrangian(..., options=...)`. Every argument is optional, but a
    problem with equality constraints needs an explicit `feasibility_tolerance`. Every
    tolerance, the penalty measure and the complementarity test act on the normalized
    quantities `f/s_f`, `h_i/s_{h_i}` and `g_j/s_{g_j}`. The multipliers and their
    bounds are those of this scaled problem. The defaults and their sources are
    registered in the
    [engineering constants](../../ENGINEERING_CONSTANTS.md#qhd-augmented-lagrangian-defaults).

    Attributes:
        initial_penalty: Default `1.0`, from Wu et al., arXiv:2605.12066v1, Sec. VI.B.
            Positive rho of the first round.
        penalty_growth: Default `2.0`, from Wu et al., Sec. VI.B. The factor gamma > 1
            that increases rho.
        reduction_ratio: Default `0.25`, from the prototype code of the authors of Wu et
            al., arXiv:2605.12066v1, strictly between 0 and 1. The tau of the penalty
            test, Birgin and Martinez Eq. (4.9). The book only requires 0 < tau < 1.
        max_penalty: Default `1e9`, from Wu et al., Sec. VI.B. Upper bound of rho, at
            least `initial_penalty`.
        max_iterations: Default `15`, from Wu et al., Sec. VI.B. Largest positive number
            of rounds, each one inner QHD solve or one box refinement.
        objective_scale: Default `1.0`, which assumes a dimensionless objective.
            Positive s_f. The layer minimizes `f/s_f`.
        equality_scales: Default `None`, which selects all ones. Positive s_{h_i} for
            every equality, in the problem's order.
        inequality_scales: Default `None`, which selects all ones. Positive s_{g_j} for
            every inequality, in the problem's order.
        feasibility_tolerance: Default `None`, which selects `1e-9` when the problem has
            only inequalities and is an error when it has an equality. The nonnegative
            epsilon_f of the normalized feasibility test
            `max(||h||_inf, ||g_+||_inf) <= epsilon_f`. With equality constraints there
            is no default, because the residual that a finite grid can reach depends on
            the grid and the constraint.
        complementarity_tolerance: Default `None`, which selects the resolved
            `feasibility_tolerance`. The nonnegative epsilon_c of the normalized
            complementarity test. The book uses one tolerance for Eqs. (10.7) and
            (10.8).
        multiplier_bounds: Default `None`. Safeguard of Algorithm 4.1 Step 4, a
            [`MultiplierBounds`][nwqlib.algorithms.qhd.constrained_records.MultiplierBounds].
            With None the next round uses the tentative multipliers and the result
            states that no safeguard was used.
        equality_multipliers: Default `None`, which starts every equality multiplier at
            0. Initial normalized lambda for every equality. Algorithm 4.1 starts the
            multipliers inside `multiplier_bounds`, so `solve_augmented_lagrangian`
            refuses None for a problem with equalities when the bounds' equality
            interval excludes zero.
        inequality_multipliers: Default `None`, which starts every inequality multiplier
            at 0. Initial normalized mu >= 0 for every inequality.
        absorb_bounds: Default `False`. Whether a one-variable linear inequality
            `a x_j + b <= 0` tightens the box of x_j instead of entering the effective
            objective, when its cut `-b/a` is a binary64 number. A cut without one stays
            a constraint. Absorption changes the grid and the dynamics, so it is off by
            default.
        inner_point: Default `"most_probable"`. How each round reads its point from the
            inner QHD result: `"most_probable"`, a joint mode, `"best_observed"`, the
            least observed table value, or `"mode_or_mean"`, the joint mode or the
            conditional mean, whichever has the smaller inner objective. With box
            refinement it must equal `BoxRefinement.point_rule`. The Point rules note
            below gives the details.
        termination: Default `"feasibility_and_complementarity"`, which stops when both
            normalized tests of Algencan's Eqs. (10.7)-(10.8) hold, with status
            `feasible_complementary`. `"feasibility"` stops on feasibility alone, with
            status `feasible`, the rule of Wu et al.'s code, which stops when the
            violation is strictly below the tolerance while this test accepts equality.
            Neither stop certifies stationarity or optimality.
        penalty_update: Default `"on_insufficient_decrease"`, which applies the test of
            Algorithm 4.1 Step 3, Eq. (4.9). `"every_iteration"` multiplies rho by
            `penalty_growth` after every round, as Wu et al.'s code does.
        stationarity: Default `False`. Whether each round reports the projected-gradient
            diagnostic r_stat. It differentiates f, h and g symbolically and does not
            affect stopping.
        inequality_form: Default `"phr"`. How each round represents its kept
            inequalities in the inner problem
            ([Proposition 54](../../mathematics.md#r54)): `"phr"`, the PHR term,
            `"slack"`, a slack variable per inequality with a positive cap, or `"auto"`,
            chosen by planned cost on the quantum route. The outer update and the
            stopping tests are those of the PHR term in every case. The Inequality forms
            note below gives the rules.

    Point rules:
        `"most_probable"` chooses a joint mode, the point that Wu et al. Sec. V read.
        `"best_observed"` chooses the least observed table value. `"mode_or_mean"`
        compares the joint mode with the joint conditional mean using the inner
        objective, with a tie going to the grid point. This is the rule of the code
        behind Wu et al.'s published results, whose code takes the mean on an exact tie.
        The round compares its inner objective, L_k under the PHR policy and L(x, s)
        with an inequality representation. A physical level solves this objective, and a
        search-model level solves its positive normalization. The point rules act in the
        inner coordinates. Refinement compares recorded relative values of the inner
        objective across levels and takes the earlier level on a tie. The outer point is
        the projection onto the original variables, and effective_value is the PHR value
        L_k evaluated at that projection when a representation is present. With box
        refinement (`solve_augmented_lagrangian(refinement=...)`) the rule reads every
        level, so it must equal `BoxRefinement.point_rule`, which applies it, and a run
        with two different rules is rejected before any work.

    Inequality forms:
        `"phr"`, the default, keeps the PHR term of every inequality, whose table spans
        all of the inequality's variables. `"slack"` requests a slack variable for each
        kept inequality with a positive cap after eligible quadratic or constant branch
        tests. The normalized slack term is `mu_j (G_j + s_j) + (rho/2) (G_j + s_j)**2`.
        A zero cap adds no axis and keeps PHR when the quadratic branch is ineligible.
        Branch tests run only without refinement and with a grid-point rule. The
        expansion has supports of at most two variables when G_j is additively
        separable. In general, product supports are contained in unions of the original
        supports, and QHD checks the expanded supports against its limits before
        tabulating them.

        `"auto"` keeps PHR without branch tests for classical execution or a user
        GaussianState. Otherwise it applies the eligible branch tests and considers
        slack conversions in constraint order. It accepts a trial that planning accepted
        when planned classical work and CX count are both no larger and at least one is
        smaller, or when the current planning is refused and the trial is accepted with
        a CX count. A missing CX count prevents the comparison. The comparison study's
        quantum quality evidence for `"auto"` covers only the QHD guide's `n = 3`,
        `K = 4` cases with one affine inequality, `KineticGroundState`, joint-mode
        readout, no refinement and exact noiseless readout under the study's fixed
        schedule and outer settings. Classical execution uses a restricted state that
        grows by a factor K per slack.

        The outer update, the stopping tests and the stationarity diagnostic stay those
        of the PHR term at the round's point in the original coordinates.

    Raises:
        ValueError: If `max_penalty` is below `initial_penalty`, or the initial
            multipliers lie outside `multiplier_bounds`, where Birgin-Martinez Algorithm
            4.1 starts them.
    """

    # The defaults of initial_penalty through max_iterations come from Wu et al.
    # arXiv:2605.12066v1, Sec. VI.B, and that of reduction_ratio from the authors'
    # prototype, within the 0 < tau < 1 of Birgin and Martinez Algorithm 4.1. Registered
    # in docs/ENGINEERING_CONSTANTS.md ("QHD augmented-Lagrangian defaults"), which gives
    # the revisit conditions.
    initial_penalty: PositiveReal = 1.0
    penalty_growth: Annotated[Real, Field(gt=1)] = 2.0
    reduction_ratio: Annotated[Real, Field(gt=0, lt=1)] = 0.25
    max_penalty: PositiveReal = 1e9
    max_iterations: PositiveInt = 15
    objective_scale: PositiveReal = 1.0
    equality_scales: tuple[PositiveReal, ...] | None = None
    inequality_scales: tuple[PositiveReal, ...] | None = None
    feasibility_tolerance: Nonnegative | None = None
    complementarity_tolerance: Nonnegative | None = None
    multiplier_bounds: MultiplierBounds | None = None
    equality_multipliers: tuple[Real, ...] | None = None
    inequality_multipliers: tuple[Nonnegative, ...] | None = None
    absorb_bounds: StrictBool = False
    # "most_probable" for exact and sampled readout alike, the default of
    # BoxRefinement.point_rule, whose comment gives the comparison behind it
    # and its limits. Registered in docs/ENGINEERING_CONSTANTS.md ("QHD
    # augmented-Lagrangian defaults").
    inner_point: Literal["most_probable", "best_observed", "mode_or_mean"] = "most_probable"
    # Birgin and Martinez Eqs. (10.7)-(10.8). With the default kinetic
    # initial state and the most probable or best observed point, continuing
    # past this stop to a stop on the penalty measure of their Eq. (4.9)
    # found the same best feasible points on five exact refined problems and
    # six sampled runs. Where the two stops differed, continuing took more
    # rounds and a larger final rho, for example 11 rounds and rho 512
    # against 10 and 256 on constrained Rastrigin. With the uniform start or
    # mode_or_mean some runs did improve by continuing. Neither stop
    # certifies stationarity or optimality.
    #
    # Measured on 2026-09-26 (Python 3.12.14, NumPy 2.5.2,
    # SciPy 1.18.1, SymPy 1.14.0, macOS arm64), with classical Schrodinger
    # execution on the Dirichlet interior grid, QuadraticSchedule(gamma=0.3)
    # with midpoint coefficients, total time 10 and 200 steps, normalized
    # tolerance 1e-6, and refinement in every round with search_model, gain
    # 8, at most three levels, max_no_improve=2 and mass threshold 0.99. The
    # five exact problems are constrained Rastrigin, Wu et al. Eqs. (17)-(18)
    # (K = 32 on [-5, 5]**2), three problems with f = (x - 1)**2 +
    # (y - 1)**2 on [-1, 1]**2, namely the unit disk g = x**2 + y**2 - 1
    # (K = 16), the off-grid equality h = x + y - 0.3 (K = 7) and the mixed
    # scales h = 1000 (x - y) and g = (x + y - 1)/1000 with scales 1000 and
    # 0.001 (K = 7), and the equality problem of the combination test in
    # tests/test_qhd_augmented_lagrangian.py with that test's scales,
    # multiplier bounds, penalty range [1, 4], four rounds, tolerance 5/32,
    # K = 7, time 6, 20 steps and mass threshold 0.5. The six sampled runs
    # drew 256 samples per level from the computed distributions of the
    # combination and Rastrigin problems, with seeds 7, 19 and 43, by
    # composing one-level solves, since the measured code had no classical
    # sampled execution. The split-step flavor gave the same finding on exact
    # Rastrigin at 3,200 steps, exact combination at 320
    # steps and the three sampled Rastrigin seeds. Registered in
    # docs/ENGINEERING_CONSTANTS.md ("QHD augmented-Lagrangian defaults").
    termination: Literal["feasibility_and_complementarity", "feasibility"] = (
        "feasibility_and_complementarity"
    )
    penalty_update: Literal["on_insufficient_decrease", "every_iteration"] = (
        "on_insufficient_decrease"
    )
    stationarity: StrictBool = False
    # "phr" adds no slack variable, so the inner problem keeps the original variables.
    # An automatic default for the quantum route awaits a comparison study of
    # the three forms. Registered in docs/ENGINEERING_CONSTANTS.md ("QHD
    # augmented-Lagrangian defaults").
    inequality_form: Literal["phr", "slack", "auto"] = "phr"

    @model_validator(mode="after")
    def _domains(self):
        if self.max_penalty < self.initial_penalty:
            raise ValueError("max_penalty must be at least initial_penalty")
        bounds = self.multiplier_bounds
        # Algorithm 4.1 starts from multipliers inside the safeguard box.
        if bounds is not None and (
            any(not bounds.equality_lower <= value <= bounds.equality_upper
                for value in self.equality_multipliers or ())
            or any(value > bounds.inequality_upper for value in self.inequality_multipliers or ())
        ):
            raise ValueError("initial multipliers must lie inside multiplier_bounds, where Birgin-Martinez "
                             "Algorithm 4.1 starts them")
        return self


class AbsorbedBound(Record):
    """One inequality `a x_j + b <= 0` replaced by the box bound `x_j <= -b/a` or `x_j >= -b/a`.

    `AugmentedLagrangian(absorb_bounds=True)` produces it, and
    `ConstraintPreprocessing.absorbed` lists it. The fields below are read-only.

    Attributes:
        inequality: Position of the inequality in the problem.
        variable: Position of x_j among the variables.
        side: `upper` for a > 0 and `lower` for a < 0.
        value: `-b/a`, which bounds x_j and equals this binary64 number exactly.
    """

    inequality: Count
    variable: Count
    side: Literal["lower", "upper"]
    value: Real


class ConstraintPreprocessing(Record):
    """The box and constraints that the rounds use, derived from the problem once, in original coordinates.

    `AugmentedLagrangianRecord.preprocessing` holds it, and `plan_augmented_lagrangian`
    returns it first. The fields below are read-only.

    Attributes:
        original_bounds: The problem's box.
        bounds: The box after bound absorption, which every inner QHD Plan and the
            stationarity diagnostic use for the original variables.
        absorbed: Inequalities replaced by box bounds, in problem order.
        equalities: Problem positions of the equalities the rounds use (all).
        inequalities: Problem positions of the inequalities the rounds use, those not
            absorbed.
        inequality_lower: For each entry of `inequalities`, the lower bound ell =
            `(c0 + sum_A min T_A)/s_g` of the normalized inequality on the grid of
            `bounds` from its support tables, rounded downward, or None outside the
            binary64 range ([Proposition 54](../../mathematics.md#r54)). It bounds the
            tabulated representation on that grid, not the continuous box or a
            refinement level's grid.
        inequality_upper: The upper bound M = `(c0 + sum_A max T_A)/s_g` on the same
            grid, rounded upward, or None outside the binary64 range.
    """

    original_bounds: tuple[tuple[Real, Real], ...]
    bounds: tuple[tuple[Real, Real], ...]
    absorbed: tuple[AbsorbedBound, ...] = ()
    equalities: tuple[Count, ...]
    inequalities: tuple[Count, ...]
    inequality_lower: tuple[Real | None, ...]
    inequality_upper: tuple[Real | None, ...]

    @model_validator(mode="after")
    def _box(self):
        if len(self.bounds) != len(self.original_bounds) or any(
            not lower < upper for lower, upper in self.bounds
        ):
            raise ValueError("preprocessed bounds need one lower < upper interval per variable")
        if set(self.inequalities) & {item.inequality for item in self.absorbed}:
            raise ValueError("an absorbed inequality cannot also enter the rounds")
        if not len(self.inequality_lower) == len(self.inequality_upper) == len(self.inequalities) or any(
            low is not None and high is not None and not low <= high
            for low, high in zip(self.inequality_lower, self.inequality_upper)
        ):
            raise ValueError("each kept inequality needs one grid range with lower <= upper")
        return self


def grid_convention(qhd):
    """Return the name of a QHD Method's grid convention (``InnerRepresentation.grid``)."""
    if qhd.boundary == "periodic":
        return "periodic"
    return "dirichlet_endpoints" if qhd.include_boundary_points else "dirichlet_interior"


class SlackAxis(Record):
    """The slack variable of one converted inequality in a round's inner problem ([Proposition 54](../../mathematics.md#r54)).

    `InnerRepresentation.slacks` lists one per converted inequality. The fields below
    are read-only. The inequality `g_j <= 0` enters the round as
    `mu_j (G_j + s_j) + (rho/2) (G_j + s_j)**2` with `G_j = g_j/s_{g_j}` and
    `s_j in [0, upper]`, whose minimum over `s_j >= 0` is the PHR term of the effective
    objective L_k whenever the box contains the minimizer `[-G_j - mu_j/rho]_+`. The
    slack has the units of the normalized G_j.

    Attributes:
        inequality: Problem position of the converted inequality.
        variable: Name of the slack variable in the inner problem.
        cap: `U_0 = [-ell_j]_+` with ell_j the recorded lower bound of
            `ConstraintPreprocessing.inequality_lower`, positive. It contains
            `[-G_j - mu/rho]_+` for every nonnegative multiplier and positive penalty on
            the grid of the preprocessed box, for the tabulated representation of G_j
            (scope `grid`).
        upper: The upper end U of the slack box `[0, U]`. It is U_0 on a Dirichlet grid,
            and on a periodic grid `K/(K-1) U_0` rounded upward, so that the slack grid
            `0, h, ..., U - h` covers `[0, U_0]` with `U - h >= U_0`.
        spacing: The slack grid's spacing h.
        error_bound: Upper bound, rounded upward, on the slack-grid excess
            `min_s phi_j(x, s) - P_j(G_j(x))` over the grid points x of the preprocessed
            box, for the round's entering multiplier and penalty. The Slack-grid bound
            note below gives its formula and the conditions under which it is a bound.

    Slack-grid bound:
        `error_bound` is `r h + rho h**2/2` with `r = [mu + rho M_j]_+` on the Dirichlet
        interior grid, and `rho h**2/8` on the endpoint grid and on the periodic grid
        with its margin. It assumes the exact mesh of spacing h and the assumptions of
        the cap. Interpreting the formula as a slack-grid excess bound requires the cap
        and table-arithmetic assumptions and the node coverage used in Proposition 54.
        The interior formula requires a first node no larger than h and distance at most
        h from every required slack to the grid. The endpoint and covered periodic
        formulas require zero and distance at most h/2. Rounding the spacing and
        coordinates can affect these conditions. This field supplies no separate
        allowance for that effect.
    """

    inequality: Count
    variable: Text
    cap: PositiveReal
    upper: PositiveReal
    spacing: PositiveReal
    error_bound: Nonnegative


class FormTrial(Record):
    """One trial conversion of automatic selection, `AugmentedLagrangian(inequality_form="auto")`.

    `InnerRepresentation.trials` lists one per trial. The fields below are read-only.
    The current representation and the trial with one more slack variable are compared
    by their planned classical work, the work that QHD's symbolic and table check counts plus
    the chosen circuit construction's work, and by the CX count of that construction.

    Attributes:
        inequality: Problem position of the inequality the trial converts.
        current_work: Planned classical work of the current representation, or None when
            its planning was refused.
        current_cx: Its CX count per circuit, or None when its planning was refused or
            the construction has no CX count.
        trial_work: Planned classical work of the trial, or None when refused.
        trial_cx: CX count of the trial, or None as for `current_cx`.
        accepted: Whether the round keeps the trial. It does when both counts are no
            larger and one is strictly smaller, or when the current planning was refused
            and the trial was accepted with a CX count. A missing CX count keeps the
            current representation.
        reason: Why the trial was accepted or kept out.
    """

    inequality: Count
    current_work: Count | None
    current_cx: Count | None
    trial_work: Count | None
    trial_cx: Count | None
    accepted: StrictBool
    reason: Text

    @model_validator(mode="after")
    def _admitted(self):
        # A kept trial was admitted and has a CX law (constrained._accepts).
        if self.accepted and (self.trial_work is None or self.trial_cx is None):
            raise ValueError("an accepted trial needs its planned host work and CX count")
        return self


class InnerRepresentation(Record):
    """How one round represents its kept inequalities in the inner problem that its QHD Plans solve.

    `ALIteration.representation` holds it under `inequality_form="slack"` or `"auto"`,
    and `plan_augmented_lagrangian` returns round 0's. The fields below are read-only.
    The round's inner objective is `L(x, s) = A(x) + sum_j T_j` of
    [Proposition 54](../../mathematics.md#r54), with the equality part A of the
    effective objective L_k and one term T_j per kept inequality. T_j is the PHR term,
    the slack term `mu_j (G_j + s_j) + (rho/2) (G_j + s_j)**2`, the quadratic
    `mu_j G_j + (rho/2) G_j**2` when `ell_j + mu_j/rho >= 0`, or the constant
    `-mu_j**2/(2 rho)` when `M_j + mu_j/rho <= 0`. On the grid of the preprocessed box
    the last two equal the PHR term for the tabulated representation of G_j, and
    `min_s L(x, s)` is the PHR objective L_k. The outer update, the stopping tests and
    the stationarity diagnostic use the PHR term at the projected point x.

    Attributes:
        policy: `AugmentedLagrangian.inequality_form`.
        forms: One of `phr`, `slack`, `quadratic` and `constant` per kept inequality, in
            the order of `ConstraintPreprocessing.inequalities`.
        slacks: One [`SlackAxis`][nwqlib.algorithms.qhd.constrained_records.SlackAxis]
            per `slack` form, in the same order.
        variables: Names of the inner problem's variables, the problem's variables
            followed by the slack variables.
        grid: The slack grid's convention, from the Method's `boundary` and
            `include_boundary_points`: `dirichlet_interior`, `dirichlet_endpoints` or
            `periodic`.
        grid_points: Points of each slack grid, the Method's `num_grid_points`.
        cap_scope: `grid`. The caps and branch tests hold on the grid of the
            preprocessed box, not on the continuous box or on the nested grids of later
            refinement levels.
        error_bound: The sum of the slacks' error bounds, rounded upward, a bound on
            `E_max` of Proposition 54 for the round's multipliers and penalty. Zero
            without slacks.
        trials: The trial conversions of automatic selection, in constraint order, empty
            for the other policies.
        selection: Why the policy chose these forms.
        table_evaluations: Objective-table grid tuples that QHD planning evaluated for
            the candidates of automatic selection, each planned candidate once, the
            current representation and every trial, whether kept or discarded. Zero for
            the other policies and when selection planned nothing. The round adds the
            candidates that its Plans do not reuse to `ALResources.table_evaluations`.
            None when a candidate's planning raised after its tables were accepted, so
            that its partial evaluations are unknown.
        table_evaluations_unavailable: Why `table_evaluations` is None, or None.
        inner_objective: Content hash of the inner objective, the SHA-256 of its
            `srepr`, the inner variables' `srepr` and the inner box, which the inner
            Plans solve.
    """

    policy: Literal["phr", "slack", "auto"]
    forms: tuple[Literal["phr", "slack", "quadratic", "constant"], ...]
    slacks: tuple[SlackAxis, ...]
    variables: tuple[Text, ...]
    grid: Literal["dirichlet_interior", "dirichlet_endpoints", "periodic"]
    grid_points: PositiveInt
    cap_scope: Literal["grid"] = "grid"
    error_bound: Nonnegative
    trials: tuple[FormTrial, ...] = ()
    selection: Text
    table_evaluations: Count | None = 0
    table_evaluations_unavailable: Text | None = None
    inner_objective: Text

    @model_validator(mode="after")
    def _slacks(self):
        if len(self.slacks) != sum(form == "slack" for form in self.forms):
            raise ValueError("a representation needs one slack axis per slack form")
        positions = [item.inequality for item in self.slacks]
        if positions != sorted(set(positions)):
            raise ValueError("slack axes follow the constraint order, one per inequality")
        names = [item.variable for item in self.slacks]
        if tuple(self.variables[len(self.variables) - len(names):]) != tuple(names) or len(set(self.variables)) != len(
            self.variables
        ):
            raise ValueError("the inner variables end with the slack variables, in order and without repetition")
        if self.policy != "auto" and (self.trials or self.table_evaluations != 0):
            raise ValueError("only automatic selection records trials and plans candidates")
        if (self.table_evaluations is None) != (self.table_evaluations_unavailable is not None):
            raise ValueError("an unknown candidate table count needs exactly one stated reason")
        return self


class ALResources(Record):
    """Resources of one augmented-Lagrangian round, or of a whole run, with measurement kept apart from classical work.

    `ALIteration.resources` holds a round's, and `ConstrainedQHDResult.resources` the
    run's totals. The fields below are read-only. A count is None when it is unknown,
    and `unavailable` names each such field with its reason. Unknown is never replaced
    by zero. A total sums each count over the rounds, except `width` and
    `restricted_dimension`, which take the largest round, and is unknown when any
    round's count is unknown. Measurement counts are what the inner Runs set aside,
    including failed and uncertain attempts, as `ExecutionTrace` counts them. A round
    with box refinement counts every level of its refinement, completed or stopping
    (`BoxRefinementResult.resources`), and a count is unknown when the refinement's is.

    The counts are the work of the algorithm, each counted once as in an uninterrupted
    run, and not the wall time spent across the calls of a saved run
    (`resume_augmented_lagrangian`). A resumed run reads the counts of its completed
    rounds and levels from their records, and a Run that it continues keeps the counts
    of its own run log. Resume also reads the layer's saved constraint preprocessing and
    check counts, and the saved unscaled support tables and evaluation count of a
    search-model level whose table stage had completed, instead of evaluating them again.

    Attributes:
        width: Register width of the inner Plan, `D K` one-hot or `D log2 K` binary for
            its D variables, slack variables included, or of the refinement's completed
            levels.
        restricted_dimension: Valid grid size `K**D` of the inner Plan, or of the
            refinement's completed levels.
        cx: CX upper bound of the circuit that each round or refinement level prepared
            (`ResourceLaw` with metric `cx`), summed over rounds and levels and not
            multiplied by shots. The Circuit counts note below says when a round counts
            0 and when the value is unknown.
        arbitrary_rotations: Arbitrary rotations of the same circuits, counted as `cx`
            is, an upper bound for the binary encoding. For quantum execution it equals
            the body total `arbitrary_rotations` of
            [`run_resources`][nwqlib.algorithms.qhd.resources.run_resources].
        circuit_preparations: Circuit preparations.
        circuit_attempts: Circuit attempts of every status.
        completed_circuit_attempts: Those of them that completed with an observation.
            The rest are attempts that are reserved, uncertain or failed.
        shots: Raw shots set aside by every attempt.
        completed_shots: Raw shots of the completed attempts.
        data_bytes: Recorded Run data bytes.
        table_evaluations: Objective-table grid tuples evaluated by QHD planning,
            including discarded automatic-selection candidates and, with refinement,
            every level's planning. The Circuit counts note below gives what each round
            adds and when the count is unavailable.
        evolution_work: Work that the inner Run counted for classical evolution before
            running it, in QHD's work units.
        construction_work: Construction work that the inner Run counted, for a quantum
            circuit or for the setup of classical evolution.
        synthesis_work: Exact dense synthesis work that the inner Run set aside.
        layer_evaluations: Scalar expression evaluations by the augmented-Lagrangian
            layer itself (f, h, g, their gradients and the mean-position value), with
            refinement also the checks of f, h and g at every point that a level
            compares. A run's total also counts the support-table entries and any
            original-expression numerical scans of the grid check before the first
            round.
        layer_work: Their work, one unit per expression-tree node and per coordinate of
            each evaluation, and in a run's total one unit per monomial of the grid
            check's expansions.
        refinement_evaluations: Evaluations of the round's inner objective by box
            refinement at its level points
            (`RefinementResources.objective_evaluations`), zero without refinement. The
            refinement does not count their work.
        joint_mass_reads: Joint-observation read count of box refinement, summed over
            its levels, zero without refinement (`RefinementResources.joint_mass_reads`
            defines it).
        unavailable: `(field, reason)` for every count that is None.

    Circuit counts:
        A round or level that prepared no circuit, including every round of classical
        execution and one whose Run's creation raised before its run-log header,
        contributes 0 to `cx` and `arbitrary_rotations`, and one whose preparation
        raised after that header or whose preparation count is unknown makes the value
        unknown. A Run prepares at most one circuit, so for quantum execution
        `arbitrary_rotations` equals the body total of `run_resources`, which also
        multiplies each circuit by its shots and adds T estimates.

        `table_evaluations` includes automatic-selection candidates that were discarded.
        An unrefined round counts its reused Plan once. A refined round also counts
        every level's planning and each search-model level's unscaled table stage. The
        count is unavailable when a failed planning attempt's partial evaluations are
        unknown. The candidates' count is `InnerRepresentation.table_evaluations`, and
        the level counts are `RefinementResources` `table_evaluations` plus
        `support_evaluations`.
    """

    width: PositiveInt | None
    restricted_dimension: PositiveInt | None
    cx: Count | None
    arbitrary_rotations: Count | None
    circuit_preparations: Count | None
    circuit_attempts: Count | None
    completed_circuit_attempts: Count | None
    shots: Count | None
    completed_shots: Count | None
    data_bytes: Count | None
    table_evaluations: Count | None
    evolution_work: Count | None
    construction_work: Count | None
    synthesis_work: Count | None
    layer_evaluations: Count
    layer_work: Count
    refinement_evaluations: Count = 0
    joint_mass_reads: Count = 0
    unavailable: tuple[tuple[Text, Text], ...] = ()

    @model_validator(mode="after")
    def _availability(self):
        missing = {name for name in type(self).model_fields
                   if name not in ("unavailable", "schema_version", "parent_id")
                   and getattr(self, name) is None}
        named = [name for name, _reason in self.unavailable]
        if len(set(named)) != len(named) or set(named) != missing:
            raise ValueError("every unknown resource count needs exactly one stated reason")
        return self


class ALEvaluation(Record):
    """The point a round chose, its residuals and the multiplier and penalty update it implies.

    `ALIteration.evaluation` holds it, for example `result.best.evaluation` for the best
    point of a `ConstrainedQHDResult`. The fields below are read-only. `infeasibility`
    and `complementarity` are the left sides of the stopping tests, and `objective` is f
    at `point` in original units. Normalized values divide by the scales of the run.
    Multipliers are normalized, and `ConstrainedQHDResult.multipliers` converts the last
    round's multipliers to original units with `lambda_i = (s_f/s_{h_i}) lambda~_i` and
    `mu_j = (s_f/s_{g_j}) mu~_j`.

    The round reads its point by the rule of `AugmentedLagrangian.inner_point`, in the
    inner coordinates of its inner objective, L_k under the PHR policy and L(x, s) with
    an inequality representation. The Point rules note of
    [`AugmentedLagrangian`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian]
    describes the rules. The outer point is the projection onto the original variables,
    and effective_value is the PHR value L_k evaluated at that projection when a
    representation is present.

    With box refinement the point is that of the level with the least recorded relative
    value of the inner objective (`ALIteration.refinement`), and the grid and the
    probability are that level's. f, h and g are evaluated at its reported point in the
    original coordinates. For the search model that point is `a + D u` rounded once to
    binary64 while the level's objective is evaluated at the exact image `a + D u`, so
    the two can differ by the rounding of the point (`RefinementLevel.point`).

    With an `InnerRepresentation` the indices, probability, tie deficit, tie window and
    mode status belong to the joint inner grid point, not to the marginal of x summed
    over the slacks, `inner_value` holds L(x, s) and `effective_value` the PHR objective
    L_k evaluated at x ([Proposition 54](../../mathematics.md#r54)). A joint mode need
    not project to the mode of the marginal distribution of x. A joint probability table
    with rows indexed by x and entries `[[0.30, 0.29], [0.31, 0.10]]` has its unique
    joint maximum in the second row but its marginal maximum in the first row, and two
    tied joint points can have the same x. An unresolved joint status therefore does not
    imply competing projected x values, and a resolved joint status does not certify a
    unique projected marginal mode.

    Attributes:
        rule: The inner-point rule (`AugmentedLagrangian.inner_point`).
        kind: `grid_point` or `valid_mean`, the valid-mass mean position that
            `mode_or_mean` can choose.
        indices: Grid index per original variable, None for `valid_mean`. With
            refinement, on the grid of the best level.
        point: The point in the original coordinates, the projection of a joint inner
            point.
        probability: Unconditional probability of the inner grid point in the inner
            result, or in the best level's result with refinement, None for
            `valid_mean`. With slack variables it is the probability of the joint point.
        tie_deficit: For a most probable grid point (`most_probable`, and `mode_or_mean`
            when it takes the grid point), the computed maximum probability minus the
            chosen point's probability, positive when a point within the tie window was
            preferred by the lexicographic rule. None otherwise. With slack variables it
            is that of the joint point.
        tie_window: The inner readout's tie window for such a point, or None when there
            is none or the window was unavailable.
        mode_status: Mode-resolution status of the inner grid representative when this
            evaluation chooses it. With refinement it is copied from the best level's
            QHD Result. None for best_observed or a chosen valid-mass mean. For a slack
            representation it describes the joint inner point before projection onto the
            original variables.
        mean_unavailable: For `mode_or_mean`, why the valid-mass mean was not compared,
            so that the round took the grid point. That happens when f, h or g was not
            finite and real there, or with refinement the best level's inner objective
            (`RefinementLevel.mean_unavailable`). None otherwise.
        effective_value: L_k at the point in normalized units. Without a representation
            it is the inner value, with refinement the best level's recorded objective
            (`RefinementLevel.objective`). With a representation it is the PHR objective
            L_k evaluated at `point` from f, h and g, also at a grid point.
        effective_value_source: `table` when read from the inner Plan's support tables,
            or with refinement when the level read L_k at its grid point from its stored
            tables. `evaluated` when L_k was evaluated at an off-grid mean, from f, h
            and g by the layer or, with refinement, by the level, and in every round
            with a representation.
        inner_value: For a round with a representation, the inner objective L(x, s) at
            the chosen joint point in normalized units, None otherwise. In exact
            arithmetic the slack objective is at least the PHR objective for nonnegative
            slacks. The recorded values can have different evaluation and
            coordinate-transformation errors, which this field does not bound.
        inner_value_source: `table` or `evaluated` for `inner_value`, by the rules of
            `effective_value_source`, None without it.
        slack_indices: Grid index per slack variable of the joint grid point, None for
            `valid_mean` and without slack variables.
        slack_point: The slack coordinates of the joint point, in the order of
            `InnerRepresentation.slacks`, None without slack variables.
        objective: f at the point in original units.
        objective_normalized: f/s_f.
        equality_residuals: h_i at the point in original units.
        equality_residuals_normalized: h_i/s_{h_i}.
        inequality_residuals: g_j at the point in original units.
        inequality_residuals_normalized: g_j/s_{g_j}.
        infeasibility: `max(||h~||_inf, ||g~_+||_inf)`, the left side of Eq. (10.8) on
            normalized residuals.
        complementarity: `max(||h~||_inf, max_j |min(-g~_j, mu+_j)|)`, the left side of
            Eq. (10.7).
        measure: `max(||h~||_inf, ||V||_inf)` with `V_j = min(-g~_j, mu-bar_j/rho)`, Eq.
            (4.9).
        tentative_equality_multipliers: lambda+ of Eq. (4.7).
        tentative_inequality_multipliers: mu+ of Eq. (4.8), nonnegative.
        next_equality_multipliers: The equality multipliers a next round uses, lambda+
            clipped by `multiplier_bounds` when given.
        next_inequality_multipliers: The inequality multipliers a next round uses,
            nonnegative.
        equality_truncated: Whether the safeguard changed each lambda+.
        inequality_truncated: Whether the safeguard changed each mu+.
        next_penalty: The rho a next round uses. A round that stops the run still
            records it.
        stationarity: The projected-gradient diagnostic r_stat, or None.
        stationarity_unavailable: Why r_stat is None: not requested, a nondifferentiable
            function, or a nonfinite gradient at the point.
    """

    rule: Literal["most_probable", "best_observed", "mode_or_mean"]
    kind: Literal["grid_point", "valid_mean"]
    indices: tuple[Count, ...] | None
    point: tuple[Real, ...]
    probability: Nonnegative | None
    tie_deficit: Nonnegative | None = None
    tie_window: Nonnegative | None = None
    mode_status: ModeStatus | None = None
    mean_unavailable: Text | None = None
    effective_value: Real
    effective_value_source: Literal["table", "evaluated"]
    inner_value: Real | None = None
    inner_value_source: Literal["table", "evaluated"] | None = None
    slack_indices: tuple[Count, ...] | None = None
    slack_point: tuple[Real, ...] | None = None
    objective: Real
    objective_normalized: Real
    equality_residuals: tuple[Real, ...]
    equality_residuals_normalized: tuple[Real, ...]
    inequality_residuals: tuple[Real, ...]
    inequality_residuals_normalized: tuple[Real, ...]
    infeasibility: Nonnegative
    complementarity: Nonnegative
    measure: Nonnegative
    tentative_equality_multipliers: tuple[Real, ...]
    tentative_inequality_multipliers: tuple[Nonnegative, ...]
    next_equality_multipliers: tuple[Real, ...]
    next_inequality_multipliers: tuple[Nonnegative, ...]
    equality_truncated: tuple[StrictBool, ...]
    inequality_truncated: tuple[StrictBool, ...]
    next_penalty: PositiveReal
    stationarity: Nonnegative | None
    stationarity_unavailable: Text | None

    @model_validator(mode="after")
    def _shapes(self):
        grid = self.kind == "grid_point"
        if (self.indices is not None) != grid or (self.probability is not None) != grid:
            raise ValueError("a grid point carries its indices and probability, and a mean position has neither")
        if grid and len(self.indices) != len(self.point):
            raise ValueError("chosen indices and point need one entry per variable")
        if (self.inner_value is None) != (self.inner_value_source is None):
            raise ValueError("an inner value needs its source")
        # A table value belongs to a grid point. Without a representation the effective value is the
        # inner value, read from a table exactly at a grid point. With one it is L_k evaluated at x.
        source = self.effective_value_source if self.inner_value is None else self.inner_value_source
        if (source == "table") != grid or self.inner_value is not None and self.effective_value_source != "evaluated":
            raise ValueError("only a grid point has a table value, and L_k at a projected point is evaluated")
        if self.slack_point is None:
            if self.slack_indices is not None:
                raise ValueError("slack indices belong to a slack point")
        elif self.inner_value is None or (self.slack_indices is not None) != grid or (
            grid and len(self.slack_indices) != len(self.slack_point)
        ):
            raise ValueError("a slack point belongs to an inner value, with one index per slack at a grid point")
        if (self.tie_deficit is not None or self.tie_window is not None) and self.rule == "best_observed":
            raise ValueError("best_observed has no probability tie")
        # A mode selection, the most probable grid point of most_probable or of mode_or_mean when it keeps the
        # grid point, carries its readout's status, and best_observed or a selected mean carries none.
        if (self.mode_status is not None) != (grid and self.rule != "best_observed"):
            raise ValueError("the mode status belongs to a most probable grid point, which best_observed and a "
                             "valid-mass mean are not")
        equalities = {len(getattr(self, name)) for name in (
            "equality_residuals", "equality_residuals_normalized", "tentative_equality_multipliers",
            "next_equality_multipliers", "equality_truncated")}
        inequalities = {len(getattr(self, name)) for name in (
            "inequality_residuals", "inequality_residuals_normalized", "tentative_inequality_multipliers",
            "next_inequality_multipliers", "inequality_truncated")}
        if len(equalities) != 1 or len(inequalities) != 1:
            raise ValueError("residuals, multipliers and truncation flags need one entry per constraint")
        if self.mean_unavailable is not None and (self.rule != "mode_or_mean" or not grid):
            raise ValueError("only a mode_or_mean round that took the grid point can skip its mean")
        # A flag records whether the safeguard changed a tentative multiplier.
        if self.equality_truncated != tuple(
            a != b for a, b in zip(self.tentative_equality_multipliers, self.next_equality_multipliers)
        ) or self.inequality_truncated != tuple(
            a != b for a, b in zip(self.tentative_inequality_multipliers, self.next_inequality_multipliers)
        ):
            raise ValueError("truncation flags must mark exactly the multipliers the safeguard changed")
        if (self.stationarity is None) == (self.stationarity_unavailable is None):
            raise ValueError("stationarity needs exactly one of its value and the reason it is unavailable")
        return self


class ALIteration(Record):
    """One augmented-Lagrangian round: its multipliers and penalty, the inner QHD solve or box refinement, and the chosen point.

    `ConstrainedQHDResult.iterations` lists every started round, and `best` and `last`
    name two of them. The fields below are read-only. Without refinement the round's
    single inner QHD solve is named by `plan_id`, `result_id` and `run_id`. With
    refinement those fields, the masses and the range bound are None, and `refinement`
    holds the round's whole refinement: its levels with their boxes, masses, Plans,
    Results and Runs, its best point and level, its stopping reason and its resources.
    Each level's random stream is the child
    `SeedSequence(entropy, spawn_key=level.spawn_key)` with
    `level.spawn_key = spawn_key + (z - 1,)` for level z.

    Attributes:
        iteration: Round number k, from 0.
        penalty: rho_k of this round.
        equality_multipliers: lambda-bar^k, the normalized equality multipliers in L_k.
        inequality_multipliers: mu-bar^k, nonnegative.
        effective_objective: Content hash of L_k, the SHA-256 of its `srepr`, the
            variable order and the box.
        entropy: Entropy of the round's child `SeedSequence`.
        spawn_key: Its spawn key under the run's root seed.
        plan_id: Content hash of the inner QHD Plan, None when planning raised or with
            refinement.
        result_id: Content hash of the inner QHDAnalysis, None when no result was
            returned or with refinement.
        run_id: Identifier of the inner Run, None with refinement and when preparation
            raised without a saved Run whose run log names it.
        valid_mass: Valid mass of the inner result, None without one.
        invalid_mass: Invalid mass of the inner result, None without one.
        missing: The inner result's reasons for missing data.
        effective_range_bound: Upper bound on `max L_k - min L_k` over the grid, the sum
            of the ranges of the inner Plan's support tables. None without a Plan, with
            refinement (each level records its own `energy_scale`), with a
            representation, whose Plan solves the inner objective instead, or when that
            sum exceeds binary64.
        representation: How the round represents its kept inequalities and which inner
            objective its Plans solve, or None under `inequality_form="phr"`, where the
            inner objective is L_k.
        inner_range_bound: For a round with a representation, the same sum of table
            ranges for the inner objective L(x, s) over the inner Plan's grid, and None
            otherwise, as for `effective_range_bound`.
        refinement: The round's box refinement of its inner objective from the
            preprocessed box and the round's slack boxes, or None without refinement.
        evaluation: The chosen point and its update, an
            [`ALEvaluation`][nwqlib.algorithms.qhd.constrained_records.ALEvaluation], or
            None when the round produced no point. That happens with no valid point, an
            inner solve that raised, a refinement that completed no level, or f, h or g
            not finite and real at the round's point.
        resources: This round's resources, including work spent before a failure.
    """

    iteration: Count
    penalty: PositiveReal
    equality_multipliers: tuple[Real, ...]
    inequality_multipliers: tuple[Nonnegative, ...]
    effective_objective: Text
    entropy: int | tuple[int, ...]
    spawn_key: tuple[Count, ...]
    plan_id: ContentID | None = None
    result_id: ContentID | None = None
    run_id: Text | None = None
    valid_mass: Nonnegative | None = None
    invalid_mass: Nonnegative | None = None
    missing: tuple[Text, ...] = ()
    effective_range_bound: Nonnegative | None = None
    representation: InnerRepresentation | None = None
    inner_range_bound: Nonnegative | None = None
    refinement: BoxRefinementResult | None = None
    evaluation: ALEvaluation | None = None
    resources: ALResources

    @model_validator(mode="after")
    def _lineage(self):
        if self.result_id is not None and (self.plan_id is None or self.run_id is None):
            raise ValueError("an inner result needs its Plan and Run")
        if (self.result_id is None) != (self.valid_mass is None) or (self.valid_mass is None) != (
            self.invalid_mass is None
        ):
            raise ValueError("inner masses belong to an inner result")
        if self.plan_id is None and (self.effective_range_bound is not None or self.inner_range_bound is not None):
            raise ValueError("the range bound of L_k comes from the inner Plan")
        represented = self.representation is not None
        if represented and self.effective_range_bound is not None or not represented and (
            self.inner_range_bound is not None
        ):
            raise ValueError("a round with a representation bounds the range of its inner objective only")
        if represented and len(self.representation.forms) != len(self.inequality_multipliers):
            raise ValueError("a representation needs one form per kept inequality")
        if self.refinement is not None:
            self._refined()
        if self.evaluation is not None:
            if self.result_id is None and self.refinement is None:
                raise ValueError("a chosen point needs its inner result")
            if len(self.evaluation.next_equality_multipliers) != len(self.equality_multipliers) or len(
                self.evaluation.next_inequality_multipliers
            ) != len(self.inequality_multipliers):
                raise ValueError("a round's update needs one multiplier per constraint")
            slacks = 0 if not represented else len(self.representation.slacks)
            if (self.evaluation.inner_value is not None) != represented or len(
                self.evaluation.slack_point or ()
            ) != slacks:
                raise ValueError("a round with a representation records its inner value and one coordinate per "
                                 "slack variable")
        return self

    def _refined(self):
        """Tie a refined round's point and streams to its refinement record.

        The round's point is the projection onto the original variables of
        the point of the level with the least recorded relative value of the
        round's inner objective, the earlier level on ties
        (``BoxRefinementResult.best_level``), so a round has a point only
        when its refinement completed a level (``_outer.refinement_stop``).
        Each level checks f, h and g at the projection of every point it
        compares (``constrained._level_check``). A round whose refinement
        completed a level lacks a point when f, h or g was nevertheless not
        finite and real at the selected point, and
        ``AugmentedLagrangianRecord`` requires that round to end the run.
        """
        refined = self.refinement
        if self.plan_id is not None or self.run_id is not None or self.missing:
            raise ValueError("a refined round keeps its inner Plans, Runs and Results in its refinement record")
        if refined.seed_entropy != self.entropy or any(
            level.spawn_key != self.spawn_key + (z,) for z, level in enumerate(refined.levels)
        ):
            raise ValueError("each refinement level's random stream is a child of its round's stream")
        best = refined.best
        if self.evaluation is not None and best is None:
            raise ValueError("a refined round's point comes from a completed level of its refinement")
        e = self.evaluation
        if e is None:
            return
        # With a representation the level's point and indices are joint, the round's point is the
        # projection onto the original variables and the level's objective is the inner value.
        d = len(e.point)
        joint = best.point_indices
        read = (e.point + (e.slack_point or ()), e.indices, e.slack_indices, e.probability, e.mean_unavailable,
                e.effective_value if e.inner_value is None else e.inner_value)
        level = (best.point, None if joint is None else joint[:d],
                 None if joint is None or e.slack_point is None else joint[d:], best.point_probability,
                 best.mean_unavailable, best.objective)
        if read != level:
            raise ValueError("a refined round's point is the projection of that of the level with the least "
                             "recorded relative objective")


class AugmentedLagrangianRecord(Record):
    """Record of a whole augmented-Lagrangian run: problem, settings, preprocessing, rounds and stopping status.

    `ConstrainedQHDResult.record` holds it, and `ConstrainedQHDResult` reads
    `termination`, `best` and `last` from it. The fields below are read-only.

    Stopping statuses:

    - `feasible_complementary`: the last round meets the normalized complementarity and
      feasibility tests of Algencan's Eqs. (10.7)-(10.8). This is not convergence in
      Algencan's sense, because the projected stationarity of Eq. (10.6) is not a
      stopping condition, and it does not assess optimality.
    - `feasible`: the last round meets the feasibility test alone
      (`termination="feasibility"`).
    - `iteration_limit`: `max_iterations` rounds ran without stopping.
    - `no_valid_point`: the last inner result had no valid grid point.
    - `budget_exhausted`: the remaining cumulative `limits` could not fund another
      round, possibly after zero rounds.
    - `inner_failed`: the inner planning, preparation or execution of the last round
      raised, for example because QHD planning refused a larger penalty or a Run refused
      the remaining limits. Before anything has completed, in the first round without
      refinement or the first level of the first round with refinement,
      `solve_augmented_lagrangian` raises the original exception instead. The status
      also ends the run when f, h or g is not finite and real at a round's point, which
      in the first round raises, or at a point that a refinement level compares, its
      grid point or a point that a stall split scores, which raises only before anything
      has completed, as above. A refined round whose point then meets the stopping test
      ends with that test's status instead. A resume with `end_at_unfinishable=True`
      records it for a reopened Run that could not finish without new work, with that
      Run's error as the failure.
    - `flat_objective`, `unresolved_objective`, `width_floor`: reachable only with
      refinement. The last round's refinement stopped before its first level, because
      the inner objective's tables on the initial augmented box show no variation, no
      variation it can resolve, or a side at its width floor
      ([`BoxRefinementResult`][nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult]),
      so the round has no point and no update.

    These statuses update no multiplier in their last round and retry nothing, and the
    completed rounds stay in `iterations`. With refinement, a round whose refinement
    completed a level and then stopped with `budget_exhausted`, `inner_failed` or
    `no_valid_point` keeps its point and update, and the run ends with
    `feasible_complementary` or `feasible` when that point meets the stopping test and
    with the refinement's reason otherwise.

    Attributes:
        problem: The portable form of the constrained problem.
        qhd: The QHD configuration every round uses.
        options: The layer's options.
        refinement: The box-refinement options of every round, or None when each round
            is one QHD solve. With refinement, `options.inner_point` equals
            `refinement.point_rule`, the readout of every level.
        execution: `quantum` or `classical`, for every round.
        shots: Shots per round, or None for exact readout.
        entropy: Entropy of the run's root `SeedSequence`.
        limits: Cumulative execution limits of the whole run.
        preprocessing: The box and constraints the rounds use.
        equality_scales: s_{h_i} of the equalities the rounds use.
        inequality_scales: s_{g_j} of the inequalities the rounds use.
        feasibility_tolerance: Resolved epsilon_f on normalized residuals.
        complementarity_tolerance: Resolved epsilon_c.
        admitted_work_bound: Upper bound on the run's counted planning work,
            `(a * M * L + t * M + 1) * qhd.max_work`, fixed before the first round. The
            Work and byte bounds note below defines a, t, M and L and what the bound
            covers.
        admitted_bytes_bound: `(a * M * L + t * M) * qhd.max_bytes`, the sum of the byte
            allowances over the attempted Plans, levels and trials. It bounds neither
            process memory nor the outer records' content-hash JSON. The Work and byte
            bounds note below gives the details.
        iterations: Every started round, in order, each an
            [`ALIteration`][nwqlib.algorithms.qhd.constrained_records.ALIteration].
        termination: One of the statuses above.
        failure: Exception type and message for `inner_failed`, the reason for
            `budget_exhausted`, otherwise None. When the last round's refinement caused
            either status, its `failure` text.
        best: Round of the reported best point, the round with the least f among rounds
            whose normalized infeasibility is at most `feasibility_tolerance`, ties to
            the earliest. Without such a round, the round with the least infeasibility,
            then f, then round number. None when no round chose a point.
        last: The last round that chose a point, which holds the reported multipliers
            and penalty, or None.
        resources: Totals of the rounds plus the layer's preprocessing evaluations.


    Work and byte bounds:
        `admitted_work_bound` is fixed before the first round and bounds the counted QHD
        work categories and the layer's own counted work. Write M for
        `options.max_iterations`, L for `refinement.max_levels`, and `a = 6` for
        search-model or `a = 4` for physical refinement. The bound is
        `(a * M * L + t * M + 1) * qhd.max_work`. Without refinement, L = 1 and a = 4.
        An ordinary Plan has at most four category bounds: the symbolic expansion, the
        running total (the initial state's evaluation and the table, compiled-block
        and schedule-integral work), the optional kept state, and the classical evolution
        or circuit construction. Planning checks the first part of the running total,
        the initial state, alone before evaluating it. A search-model level adds the
        symbolic and running-total bounds of its unscaled table stage. Intermediate checks of one
        category are not counted separately. The trials of automatic inequality
        selection on the quantum route add `t = 6 (m + 1) - 4` bounds per round without
        refinement and `t = 6 (m + 1)` with it, for m kept inequalities, and t = 0
        otherwise. The layer counts the work of its support-table check of f, h and g,
        which also gives the inequality ranges, and of its evaluations once for the
        whole run. At most M L levels are attempted, an attempted level that stops or
        fails included. Refinement geometry, marginal and joint-mass processing, the
        refinement's repeated objective decomposition and point evaluation, the symbolic
        construction of the inner objectives, and library internals such as SymPy, SciPy
        and the simulators lie outside these limit checks.

        `admitted_bytes_bound` is `(a * M * L + t * M) * qhd.max_bytes`, with a, t, M
        and L as above. It sums the byte allowances of the QHD categories over the
        attempted Plans, levels and trials, each at most `qhd.max_bytes` while its
        operation runs. It bounds neither process memory, nor the content-hash JSON of
        the outer records, nor the further allocations of refinement itself.

    Record size:
        The bounds on the scalar JSON leaves of this record, with their derivation, are
        in [Engineering constants](../../ENGINEERING_CONSTANTS.md#augmented-lagrangian-record-size).
    """

    problem: ConstrainedOptimization
    qhd: QHD
    options: AugmentedLagrangian
    refinement: BoxRefinement | None = None
    execution: Literal["quantum", "classical"]
    shots: PositiveInt | None
    entropy: int | tuple[int, ...]
    limits: ExecutionLimits
    preprocessing: ConstraintPreprocessing
    equality_scales: tuple[PositiveReal, ...]
    inequality_scales: tuple[PositiveReal, ...]
    feasibility_tolerance: Nonnegative
    complementarity_tolerance: Nonnegative
    admitted_work_bound: PositiveInt
    admitted_bytes_bound: PositiveInt
    iterations: tuple[ALIteration, ...]
    termination: Literal[
        "feasible_complementary", "feasible", "iteration_limit", "no_valid_point",
        "budget_exhausted", "inner_failed", "flat_objective", "unresolved_objective", "width_floor",
    ]
    failure: Text | None = None
    best: Count | None
    last: Count | None
    resources: ALResources

    @model_validator(mode="after")
    def _run(self):
        rounds = self.iterations
        if tuple(item.iteration for item in rounds) != tuple(range(len(rounds))):
            raise ValueError("rounds must be numbered 0, 1, ... in order")
        if len(rounds) > self.options.max_iterations:
            raise ValueError("a run cannot exceed max_iterations rounds")
        if len(self.equality_scales) != len(self.preprocessing.equalities) or len(
            self.inequality_scales
        ) != len(self.preprocessing.inequalities):
            raise ValueError("resolved scales need one entry per constraint the rounds use")
        for item in rounds:
            if len(item.equality_multipliers) != len(self.equality_scales) or len(
                item.inequality_multipliers
            ) != len(self.inequality_scales):
                raise ValueError("round multipliers need one entry per constraint the rounds use")
        for earlier, later in zip(rounds, rounds[1:]):
            update = earlier.evaluation
            if update is None or (
                later.penalty,
                later.equality_multipliers,
                later.inequality_multipliers,
            ) != (update.next_penalty, update.next_equality_multipliers, update.next_inequality_multipliers):
                raise ValueError("each round must use the multipliers and penalty its predecessor computed")
        if self.refinement is not None:
            self._refined_run()
        else:
            failed = self.termination in ("no_valid_point", "inner_failed")
            if any(item.refinement is not None for item in rounds) or self.termination in _REFINEMENT_ONLY:
                raise ValueError("refinement records and statuses belong to a run with refinement options")
            if any(item.evaluation is None for item in rounds[:-1]) or (
                rounds and (rounds[-1].evaluation is None) != failed
            ) or (failed and not rounds):
                raise ValueError("only a final no_valid_point or inner_failed round lacks a chosen point")
            if self.termination == "inner_failed" and len(rounds) < 2:
                raise ValueError("inner_failed follows at least one completed round")
        if self.termination == "iteration_limit" and len(rounds) != self.options.max_iterations:
            raise ValueError("iteration_limit means max_iterations rounds ran")
        if (self.failure is not None) != (self.termination in ("inner_failed", "budget_exhausted")):
            raise ValueError("a failure reason belongs to inner_failed or budget_exhausted")
        chosen = [item.iteration for item in rounds if item.evaluation is not None]
        if self.last != (chosen[-1] if chosen else None) or (self.best is None) != (self.last is None) or (
            self.best is not None and self.best not in chosen
        ):
            raise ValueError("best and last must name rounds that chose a point")
        self._links()
        return self

    def _refined_run(self):
        """Tie every round's refinement to the run's settings and the status to the last refinement.

        ``_outer.refinement_stop`` states the rule: only the last round's
        refinement can end the run, a refinement that completed no level ends
        it with its own reason, and one that completed a level and then
        stopped with ``budget_exhausted``, ``inner_failed`` or
        ``no_valid_point`` ends it with the success status when the round's
        point met the stopping test and with that reason otherwise.
        """
        rounds = self.iterations
        if self.options.inner_point != self.refinement.point_rule:
            raise ValueError("with refinement, options.inner_point must equal refinement.point_rule")
        for item in rounds:
            refined = item.refinement
            if refined is None or (refined.options, refined.qhd, refined.execution, refined.shots) != (
                self.refinement, self.qhd, self.execution, self.shots
            ):
                raise ValueError("every round refines L_k with the run's refinement options, QHD configuration "
                                 "and execution")
        stops = [refinement_stop(item.refinement.termination, len(item.refinement.levels)) for item in rounds]
        if any(stop is not None for stop in stops[:-1]):
            raise ValueError("a refinement that ends the run belongs to its last round")
        if rounds and rounds[-1].evaluation is None and rounds[-1].refinement.levels:
            # f, h or g was not finite and real at the refinement's point, a later failure of the
            # round (_outer.inner_failure), which propagates in round 0.
            if self.termination != "inner_failed" or len(rounds) < 2:
                raise ValueError("a refined round without a point after a completed level ends the run with "
                                 "inner_failed after the first round")
            return
        stop = stops[-1] if rounds else None
        if stop is None:
            if self.termination in ("no_valid_point", "inner_failed", *_REFINEMENT_ONLY):
                raise ValueError(f"{self.termination} needs a last refinement that stopped with it")
            return
        refined, last = rounds[-1].refinement, rounds[-1].evaluation
        met = last is not None and last.infeasibility <= self.feasibility_tolerance and (
            self.options.termination == "feasibility" or last.complementarity <= self.complementarity_tolerance)
        success = "feasible" if self.options.termination == "feasibility" else "feasible_complementary"
        if self.termination != (success if met else stop):
            raise ValueError("a run that its last refinement stopped ends with the success status when the round "
                             "met the stopping test and with the refinement's reason otherwise")
        if self.termination == "inner_failed" and not refined.levels and len(rounds) < 2:
            raise ValueError("a first-level failure of the first round propagates instead of being recorded")
        if self.termination in ("inner_failed", "budget_exhausted") and self.failure != refined.failure:
            raise ValueError("the run's failure text is that of its last refinement")

    def _links(self):
        """Tie the rounds and the status to the options, problem and resolved settings they came from."""
        options, rounds, kept = self.options, self.iterations, self.preprocessing.inequalities
        if self.preprocessing.original_bounds != self.problem.bounds:
            raise ValueError("preprocessing must start from the problem's box")
        absorbed = {item.inequality for item in self.preprocessing.absorbed}
        if (self.preprocessing.equalities != tuple(range(len(self.problem.equalities)))
                or len(absorbed) != len(self.preprocessing.absorbed) or len(set(kept)) != len(kept)
                or set(kept) | absorbed != set(range(len(self.problem.inequalities)))):
            raise ValueError("preprocessing must keep or absorb each of the problem's constraints once")
        for name, count in (("equality_scales", len(self.problem.equalities)),
                            ("inequality_scales", len(self.problem.inequalities)),
                            ("equality_multipliers", len(self.problem.equalities)),
                            ("inequality_multipliers", len(self.problem.inequalities))):
            given = getattr(options, name)
            if given is not None and len(given) != count:
                raise ValueError(f"options.{name} needs one value per constraint of the problem")
        for given, resolved in ((options.equality_scales, self.equality_scales),
                                (None if options.inequality_scales is None
                                 else tuple(options.inequality_scales[j] for j in kept), self.inequality_scales)):
            if given is not None and given != resolved:
                raise ValueError("resolved scales differ from the options")
        given = options.feasibility_tolerance
        if given is not None and given != self.feasibility_tolerance:
            raise ValueError("resolved feasibility tolerance differs from the options")
        given = options.complementarity_tolerance
        if self.complementarity_tolerance != (self.feasibility_tolerance if given is None else given):
            raise ValueError("resolved complementarity tolerance differs from the options")
        if rounds:
            first = rounds[0]
            initial = options.inequality_multipliers
            if (first.penalty != options.initial_penalty
                    or first.equality_multipliers != (options.equality_multipliers
                                                      or (0.0,) * len(self.equality_scales))
                    or first.inequality_multipliers != (tuple(initial[j] for j in kept) if initial is not None
                                                        else (0.0,) * len(kept))):
                raise ValueError("the first round must use the options' initial penalty and multipliers")
        if any(item.penalty > options.max_penalty for item in rounds):
            raise ValueError("a round's penalty exceeds max_penalty")
        names = tuple(str(name) for name in self.problem.variable_names)
        lower = dict(zip(kept, self.preprocessing.inequality_lower, strict=True))
        for item in rounds:
            form = item.representation
            if (form is None) != (options.inequality_form == "phr") or form is not None and (
                form.policy != options.inequality_form or form.variables[:len(names)] != names
                or len(form.variables) != len(names) + len(form.slacks)
                or tuple(axis.inequality for axis in form.slacks)
                != tuple(j for j, kind in zip(kept, form.forms, strict=True) if kind == "slack")
            ):
                raise ValueError("every round of a non-PHR policy records its representation of the kept "
                                 "inequalities after the problem's variables")
            if form is None:
                continue
            kinds = dict(zip(kept, form.forms, strict=True))
            bound = sum((Fraction(axis.error_bound) for axis in form.slacks), Fraction(0))
            if (form.grid, form.grid_points) != (grid_convention(self.qhd), self.qhd.num_grid_points) or any(
                lower[axis.inequality] is None or axis.cap != -lower[axis.inequality] or axis.upper < axis.cap
                for axis in form.slacks
            ) or Fraction(form.error_bound) < bound or any(
                trial.accepted and kinds.get(trial.inequality) != "slack" for trial in form.trials
            ):
                raise ValueError("a round's representation uses the run's grid, the recorded caps with their error "
                                 "bounds, and every trial conversion it accepted")
        if options.multiplier_bounds is None and any(
            item.evaluation is not None and (
                item.evaluation.next_equality_multipliers != item.evaluation.tentative_equality_multipliers
                or item.evaluation.next_inequality_multipliers != item.evaluation.tentative_inequality_multipliers)
            for item in rounds
        ):
            raise ValueError("without multiplier_bounds the next round uses the tentative multipliers")
        stops = {"feasible_complementary": "feasibility_and_complementarity", "feasible": "feasibility"}
        if self.termination in stops:
            last = rounds[-1].evaluation if rounds else None
            if (options.termination != stops[self.termination] or last is None
                    or last.infeasibility > self.feasibility_tolerance
                    or self.termination == "feasible_complementary"
                    and last.complementarity > self.complementarity_tolerance):
                raise ValueError("a stopping status needs its test option and a last round that meets it")

    @property
    def safeguard_used(self):
        """Whether `options.multiplier_bounds` applied Algorithm 4.1 Step 4.

        The truncation flags of each round show which values changed.
        """
        return self.options.multiplier_bounds is not None


class ConstrainedGridMinimum(Record):
    """Evaluated finite-grid reference of `constrained_grid_minimum`.

    [`constrained_grid_minimum`][nwqlib.algorithms.qhd.constrained.constrained_grid_minimum]
    returns it. The fields below are read-only. The answer is `gap`, the run's stored
    best objective minus the least evaluated feasible objective `objective`, in original
    units. Computed feasibility uses the run's scales and `feasibility_tolerance` on the
    normalized residuals. The reference covers the preprocessed box's grid. It bounds
    neither objective nor constraint evaluation error and does not establish equality of
    the computed and mathematical feasible sets. A large objective constant can make
    distinct exact values tie in binary64.

    Attributes:
        record_id: Content hash of the AugmentedLagrangianRecord of the run.
        grid_points: `K**d` points evaluated.
        feasible_points: Points whose computed normalized infeasibility is at most the
            tolerance.
        indices: Grid index per variable of the point passing the computed feasibility
            test with the least evaluated binary64 objective, ties to the
            lexicographically smallest index, or None when no grid point passes.
        point: That point in the original coordinates, or None.
        objective: Its evaluated f in original units, including the constant, or None.
        best_objective: Stored evaluated f at the run's reported best point, or None
            when the run chose no point.
        best_feasible: Whether that best point passes the same computed tolerance test,
            or None. This does not state objective optimality.
        gap: `best_objective - objective` in original units, or None when either is
            None. It compares a stored value with a fresh grid minimum and can be
            negative, including when the point is infeasible or is an off-grid mean
            chosen by `mode_or_mean`. The Gap note below bounds the exact gap.
        feasibility_tolerance: The tolerance used.
        work: Work checked against `max_work` before the first evaluation.

    Gap:
        Let G be the nonempty grid subset passing this check's computed feasibility
        test, and let the reported point a lie in G. Suppose a finite E >= 0 bounds the
        absolute error of the stored objective at a and every fresh objective on G,
        relative to exact values at the same coordinates. For finite reported gap r and
        u = 2**-53, the exact gap `F(a) - min_G F` is at most `r/(1-u) + 2*E` if r >= 0,
        and `r/(1+u) + 2*E` if r < 0. These are real-arithmetic inequalities under
        round-to-nearest binary64 with gradual underflow. The check establishes neither
        E nor equality of G with the mathematically feasible grid set.
    """

    record_id: ContentID
    grid_points: PositiveInt
    feasible_points: Count
    indices: tuple[Count, ...] | None
    point: tuple[Real, ...] | None
    objective: Real | None
    best_objective: Real | None
    best_feasible: StrictBool | None
    gap: Real | None
    feasibility_tolerance: Nonnegative
    work: Count
