"""Options and records of QHD box refinement (``refinement.refine_box``)."""

from typing import Annotated, Literal

from pydantic import Field, PrivateAttr, model_validator

from nwqlib.core.records import ContentID, Nonnegative, PositiveInt, Real, Record, Text
from nwqlib.operators.access import Count
from ._outer import total_counts
from .initial_state import GaussianState
from .method import QHD
from .records import ModeStatus

# kappa of the search model when BoxRefinement.potential_gain is None. No
# value is best for every problem (refinement._level_problem shows why kappa
# has no monotone effect). In a comparison with the settings of the scaling
# comparison in BoxRefinement and the kinetic initial state, kappa = 8 gave
# a smaller infinity-norm distance of the best point from the minimizer than
# kappa = 1 after ten levels on all three standalone problems (double well
# 5.0e-6 against 2.2e-4, anisotropic quadratic 3.0e-6 against 3.9e-4,
# Ackley 3.0e-8 against 1.4e-5), at 8 to 13 percent more counted evolution
# work (RefinementResources.evolution_work) with either initial state. With
# the best observed point and the uniform initial state, kappa = 1 did
# better on the exact-readout double well (0.00112 against 0.00360). These
# were measured on 2026-09-26 at revision
# aede9fd119563d63562c9afed4b00021920560c6, in the environment that the
# refinement module docstring states. At revision
# 461d7dce706eed4724a4e850915746f8683831b2 on the same day and in the same
# environment, the best observed point with the uniform initial state,
# max_no_improve=2, max_work = max_bytes = 10**10 per solve and 256 outcomes
# per level drawn from each computed distribution (seeds 7, 19 and 43, by
# composing one-level classical solves, since classical execution takes no
# shots) favored kappa = 1 on one of three Ackley seeds. So the value stays
# configurable. Only 1 and 8 were compared. Registered in
# docs/ENGINEERING_CONSTANTS.md ("Box refinement scaling and
# potential_gain").
DEFAULT_POTENTIAL_GAIN = 8.0

# Width sigma of the best-point Gaussian of
# BoxRefinement.level_initial_state="best_point_gaussian", as a fraction of
# each side of the level box, when BoxRefinement.level_gaussian_width is
# None. It is the only width that has been compared, a tested value and
# not an optimum. The two-variable refinement comparison and the
# one-variable split-step check of refinement._best_point_gaussian both
# used amplitude width 1/6 in unit coordinates. That docstring gives their
# settings and results, and the refinement module docstring the revision
# and environment of the measurements. Registered in
# docs/ENGINEERING_CONSTANTS.md ("Box refinement level initial state").
DEFAULT_LEVEL_GAUSSIAN_WIDTH = 1 / 6


class BoxRefinement(Record):
    """Options of one box refinement, the finite-grid search of Wu et al. arXiv:2605.12066v1, Sec. V.

    Every level solves QHD on its box, keeps on each axis the index interval
    that holds the conditional marginal mass ``mass_threshold``, and solves
    the next level on the box those intervals represent. The defaults are
    registered in docs/ENGINEERING_CONSTANTS.md ("Box refinement defaults",
    "Box refinement scaling and potential_gain" and "Box refinement stall
    split").

    Attributes:
        scaling: How a level box becomes a QHD problem. ``"search_model"``
            solves on the unit box ``u in [0, 1]^d`` the normalized objective
            ``(F(a + D u) - c)/E`` of ``refinement._level_problem``, the
            same dimensionless model at every level. ``"physical"`` solves the
            original objective on the level box in original coordinates, the
            same physical Hamiltonian on a smaller box. The default is
            ``"search_model"``. Every level plans its problem with the QHD
            configuration, so under the search model the center and widths
            of a ``GaussianState`` given as ``QHD.initial_state`` are unit
            coordinates of each level box, not original ones.
        potential_gain: kappa > 0, the dimensionless strength of the search
            model's normalized potential. Each level solves
            ``a(t) T_u + kappa b(t) V_z`` with ``V_z = (F - c_z)/E_z``, so
            kappa multiplies V_z after E_z is formed
            (``refinement._level_problem``). None selects
            ``DEFAULT_POTENTIAL_GAIN`` (8) for the search model. The physical
            model solves the original objective and has no gain, so it
            rejects any explicit value. ``gain`` resolves the value each level
            solves.
        max_levels: Largest number of levels solved.
        mass_threshold: Conditional marginal mass eta that each axis
            interval must reach, in (0, 1]. The reported union-bound formula
            is ``max(0, 1 - sum_j(1 - m_j))`` for the observed conditional
            marginal masses m_j (``RefinementLevel.joint_mass_bound``).
            In exact arithmetic, marginal masses at least eta imply joint
            mass at least ``max(0, 1 - d(1 - eta))``. With counts these are
            empirical masses conditional on a valid outcome, not a confidence
            bound for the underlying population.
        box_rule: ``"centered"``: each kept grid point represents the cell
            between the midpoints to its neighbors, of width h centered on it
            on a uniform grid, and an interval that reaches an end of its
            axis keeps that face of the box (``refinement._next_box``).
        max_no_improve: Number of consecutive levels without a strict
            decrease of the best relative objective
            (``RefinementLevel.relative_objective``) after which refinement
            stops.
        point_rule: Point each level reports. ``"most_probable"``, the
            default, is the most probable valid grid point of the level's QHD
            result, ``"best_observed"`` the observed valid point with positive
            weight and the least evaluated binary64 value of the objective
            solved by that level (``QHDAnalysis.candidate``), and ``"mode_or_mean"`` the
            conditional mean position when its relative objective is strictly
            smaller than the most probable point's, and that point otherwise.
            In the augmented-Lagrangian layer
            (``constrained.solve_augmented_lagrangian(refinement=...)``) the
            objective is the round's L_k, this rule reads every level of the
            round, and it must equal ``AugmentedLagrangian.inner_point``, so
            that one rule describes every point of the run.
        stall_split: What a level does when its ordinary next box equals
            its box. ``"none"`` stops the refinement with ``box_unchanged``.
            ``"best_region"`` finds the resolved valley with the largest
            smaller-peak/valley ratio of an axis marginal, scores each
            strict side of it by the objective at the side's most
            probable joint grid point, and continues on the lower-scoring
            side only, together with the valley cell
            (``refinement._stall_split``, which derives the rule, its cost
            and its risk). A level without a resolved valley
            (``refinement._valley_admission``) still stops with
            ``box_unchanged``. A periodic grid with a positive split budget
            is refused. The other side's probability
            is given up, so the kept box no longer holds ``mass_threshold``
            of that level's distribution. Classical execution needs
            ``QHD(keep_state=True)`` for it, since the scores read the joint
            distribution.
        max_splits: Largest number of stall splits in one refinement. A
            stall after the last one stops with ``split_limit``, whether or
            not its level has a valley, and with ``max_splits=0`` the first
            stall does. It keeps its default with ``stall_split="none"``.
            In the augmented-Lagrangian layer the budget holds for each
            round's refinement.
        level_initial_state: The state each level starts from.
            ``"configured"``, the default, plans every level with
            ``QHD.initial_state``. ``"best_point_gaussian"`` plans the first
            level with ``QHD.initial_state`` and every later level with a
            ``GaussianState`` centered at the refinement's best point so far,
            with width ``gaussian_width`` times each side of the level box
            (``refinement._best_point_gaussian``, which states when it helped
            and when it hurt). Each such level records its Gaussian
            (``RefinementLevel.initial_state``). In the augmented-Lagrangian
            layer the first level of every round uses ``QHD.initial_state``.
        level_gaussian_width: Width sigma of the best-point Gaussian as a
            fraction of each side of the level box, which is its width in the
            search model's unit coordinates. None selects
            ``DEFAULT_LEVEL_GAUSSIAN_WIDTH`` (1/6) with
            ``level_initial_state="best_point_gaussian"``. ``"configured"``
            rejects any value. ``gaussian_width`` resolves it.
    """

    # The search model solves the same dimensionless problem at every level,
    # invariant under translations, positive coordinate rescalings and
    # additive constants of F (refinement._level_problem). In a comparison
    # on two-variable Dirichlet interior grids with the classical Schrodinger
    # model, T = 10, 200 steps, the quadratic schedule with gamma = 0.3, the
    # uniform initial state, eta = 0.99, the most probable point,
    # max_levels=10, max_no_improve=10 and max_work = max_bytes = 10**10,
    # the physical model stopped with an unchanged box after 7 levels on the
    # double well of refinement._level_problem with K = 12 and after 3 on the
    # anisotropic quadratic (x - 3/10)**2 + (y - 2/5)**2 + x y/2 on
    # [0, 1] x [-1/2, 3/2] with K = 12, where the search model kept
    # shrinking for ten levels. On the two-variable Ackley function of Wu et
    # al. arXiv:2605.12066v1, Eq. (16) (``ACKLEY`` in
    # examples/generators/qhd_scientific.py), on [-5, 5]**2 with K = 32, both
    # reached ten levels, and the physical model ended closer to the
    # minimizer than the search model with gain 1 and farther than with
    # gain 8 (infinity-norm distances 0.00441, 0.0320 and 0.00151). That
    # minimizer is the binary64 point (0.961528396769231, 0.3696594771697266)
    # (``SHIFT`` in the same file). The paper's Sec. VI.A reports this
    # seed-123 shift as x* ~ (0.962, 0.370), and the point is the fourth of
    # the successive uniform(-1, 1, size=2) draws from
    # numpy.random.RandomState(123) that the authors' experiment script,
    # which the paper does not link, makes one per test function, the Ackley
    # draw after those of the quadratic, Rosenbrock and Rastrigin functions.
    # Measured
    # on 2026-09-26 at revision aede9fd119563d63562c9afed4b00021920560c6 with
    # seed 7, in the environment that the refinement module docstring
    # states. The physical model stays an explicit option. Registered in
    # docs/ENGINEERING_CONSTANTS.md ("Box refinement scaling and
    # potential_gain").
    scaling: Literal["search_model", "physical"] = "search_model"
    # max_levels=3 is the default of the refinement prototype written by the
    # authors of Wu et al. arXiv:2605.12066v1, mass_threshold=0.99 the eta of
    # that paper's Sec. V and max_no_improve=2 the setting of its experiment
    # scripts. Registered in docs/ENGINEERING_CONSTANTS.md ("Box refinement
    # defaults").
    max_levels: PositiveInt = 3
    mass_threshold: Annotated[Real, Field(gt=0, le=1)] = 0.99
    box_rule: Literal["centered"] = "centered"
    max_no_improve: PositiveInt = 2
    # "most_probable" is the candidate rule of Wu et al. Sec. V, and it keeps
    # a point that the computed distribution defines. With the default
    # kinetic initial state (QHD.initial_state) it matched "best_observed" on
    # exact refined constrained Rastrigin and on the standalone double-well
    # and Ackley comparisons, and on sampled refined constrained Rastrigin it
    # tied twice and gave a better point once in three seeds. With the
    # uniform initial state "best_observed" gave better points on refined
    # constrained Rastrigin and on sampled standalone Ackley, so it stays an
    # explicit option. These comparisons were made on 2026-09-26 at revision
    # 461d7dce706eed4724a4e850915746f8683831b2 (standalone, uniform start) and
    # 659fa80702e79192a27568ea4935531004d14e51 (augmented Lagrangian, and
    # standalone with the kinetic start), in the environment that the
    # refinement module docstring states, with classical schrodinger
    # evolution over T = 10 in 200 steps and max_work = max_bytes = 10**10
    # per solve. The standalone runs used the problems and grids of the
    # scaling comparison above, gain 8, at most ten levels and
    # max_no_improve=2. Constrained Rastrigin was
    # 20 + 9x**2 + 9y**2 - 10 cos(6 pi x) - 10 cos(6 pi y) on [-5, 5]**2 with
    # K = 32 and the inactive constraints 0.5 + 0.02 (y - 0.5)**2 - x <= 0 and
    # 0.5 + 0.02 (x - 0.5)**2 - y <= 0, with at most 15 rounds from penalty 1,
    # normalized tolerances 1e-6 and a refinement of at most three levels at
    # gain 8 and max_no_improve=2 in each round. Sampled runs drew 256
    # outcomes per level from each computed distribution (seeds 7, 19 and 43)
    # by composing one-level classical solves, since classical execution
    # takes no shots. Under exact readout with every grid point supported,
    # "best_observed" takes the least evaluated binary64 value of the
    # level's solved objective over that grid, independently of probability
    # magnitudes. This evaluated-table statement does not certify a
    # mathematical objective minimum. At a fixed level its point is no
    # evidence of a sampling advantage. The refinement boxes
    # still depend on the distribution. The same default holds for
    # AugmentedLagrangian.inner_point, since one rule reads every point of a
    # refined augmented-Lagrangian run. Registered in
    # docs/ENGINEERING_CONSTANTS.md ("Box refinement defaults").
    point_rule: Literal["most_probable", "best_observed", "mode_or_mean"] = "most_probable"
    # None resolves to DEFAULT_POTENTIAL_GAIN for the search model (gain).
    potential_gain: Annotated[Real, Field(gt=0)] | None = None
    # "none" keeps the mass_threshold condition of every box, since a split
    # discards probability on the evidence of two objective values
    # (refinement._stall_split). max_splits=1 allows one such discard per
    # refinement, and the double-well example in that docstring made one
    # split, at level 5, under budgets 1 and 3, measured at the revision and
    # in the environment that the refinement module docstring states.
    # Registered in
    # docs/ENGINEERING_CONSTANTS.md ("Box refinement stall split").
    stall_split: Literal["none", "best_region"] = "none"
    max_splits: Count = 1
    # "configured" keeps QHD.initial_state, the default KineticGroundState,
    # at every level. A best-point Gaussian helped in a one-variable case, and
    # in all six cases of a two-variable comparison it did worse than the
    # kinetic ground state at every level (refinement._best_point_gaussian),
    # so it stays an explicit choice. Registered in docs/ENGINEERING_CONSTANTS.md ("Box
    # refinement level initial state").
    level_initial_state: Literal["configured", "best_point_gaussian"] = "configured"
    # None resolves to DEFAULT_LEVEL_GAUSSIAN_WIDTH under best_point_gaussian (gaussian_width).
    level_gaussian_width: Annotated[Real, Field(gt=0)] | None = None

    @property
    def gaussian_width(self):
        """Return the width fraction w of the best-point Gaussian, or None with ``level_initial_state="configured"``.

        It is ``level_gaussian_width``, or ``DEFAULT_LEVEL_GAUSSIAN_WIDTH``
        when that is None (``refinement._best_point_gaussian``).
        """
        if self.level_initial_state == "configured":
            return None
        return DEFAULT_LEVEL_GAUSSIAN_WIDTH if self.level_gaussian_width is None else self.level_gaussian_width

    @property
    def gain(self):
        """Return kappa, the gain that each level solves, or None for the physical model.

        It is ``potential_gain``, or ``DEFAULT_POTENTIAL_GAIN`` when that is
        None. ``refinement.refine_box`` solves and records this value
        (``RefinementLevel.potential_gain``).
        """
        if self.scaling == "physical":
            return None
        return DEFAULT_POTENTIAL_GAIN if self.potential_gain is None else self.potential_gain

    @model_validator(mode="after")
    def _where_options_act(self):
        """Allow a gain only for the search model, a split budget other than 1 only with splits, and a width only with the Gaussian."""
        if self.scaling == "physical" and self.potential_gain is not None:
            raise ValueError("potential_gain scales the search model's normalized potential. The physical "
                             "model solves the original objective and has no gain, so leave potential_gain "
                             "unset with scaling='physical'")
        if self.stall_split == "none" and self.max_splits != 1:
            raise ValueError("max_splits budgets the stall split, so stall_split='none' needs max_splits=1")
        if self.level_initial_state == "configured" and self.level_gaussian_width is not None:
            raise ValueError("level_gaussian_width sets the width of the best-point Gaussian, so it needs "
                             "level_initial_state='best_point_gaussian'")
        return self


class RefinementResources(Record):
    """Counted work of one refinement level, or of a whole refinement.

    Acquisition (circuit preparations and attempts, shots, stored Run data)
    and host work (planning tables, classical kernel invocations,
    construction, exact synthesis and the refinement's own evaluations) are
    separate populations and are never added to each other. Each work field
    has its own unit, so the host-work fields are not summed either. A
    search-model level plans twice: once for the unscaled level objective,
    whose support tables give E and c (``table_evaluations``), and once for
    the normalized objective it solves (``support_evaluations``). A physical
    level plans once. A count is None when it is unknown, and
    ``unavailable`` names each such field with its reason. Unknown is never
    replaced by zero, so a total (``total_resources``) is None when any
    level's count is. The Run counts are those of ``_outer.run_counts``.
    The counts are the work of the algorithm, each counted once as in an
    uninterrupted refinement, and not the host time spent across the calls
    of a durable refinement (``refinement.resume_box_refinement``), whose
    completed levels keep their recorded counts and whose continued Run
    keeps the counts of its own journal. The unscaled table stage of a
    reopened search-model level, which resume repeats, is not counted again.

    Attributes:
        circuit_preparations: Native circuit preparations of the level Runs.
        circuit_attempts: Native circuit acquisition attempts, of any status.
        completed_circuit_attempts: Those of them that completed with an
            observation.
        shots: Raw shots reserved by the attempts of every status.
        completed_shots: Raw shots of the completed attempts.
        data_bytes: Run data bytes recorded by the level Runs.
        evolution_work: Work units that the level Runs charged for classical
            kernel invocations (restricted evolution and readout summary),
            zero for native execution.
        construction_work: Construction work that the level Runs charged,
            for a native circuit or for the classical kernel's setup.
        synthesis_work: Exact dense synthesis work reserved by the level Runs.
        table_evaluations: Grid tuples evaluated for the unscaled level
            objective's support tables, the extra planning of the search
            model. Zero for the physical model.
        support_evaluations: Grid tuples evaluated for the support tables of
            the solved level objective.
        objective_evaluations: Evaluations of the original objective by the
            refinement itself, at the level's own coordinates
            (``RefinementLevel.objective``), including the two scores of a
            stall split.
        joint_mass_reads: Joint-observation read charge. Count and exact-bin
            entries are charged when the observations are decoded and their
            decoded data are reused. A kept-state dense grid is charged once
            by the kept-state length. A streamed kept-state readout charges
            that length for each started probability pass, one per joint
            request and two for the paired split modes, even when only valid
            one-hot entries are gathered or the final pass stops early. These
            kept-state units are full-state-equivalent charges, not measured
            memory traffic (``refinement._LevelReadout``).
        cx: Sum of the selected native blocks' CX upper bounds over the
            circuits that the level Runs prepared, one circuit per level Run
            at most, not multiplied by shots (``_outer.law_count``). A level
            that prepared no circuit, such as one that stopped before its
            preparation, whose Run's creation raised before its journal
            header (``_outer.HeaderlessRun``) or that ran classically,
            contributes 0, and one whose preparation raised after that
            header or whose preparation count is unknown makes the value
            unknown.
        arbitrary_rotations: Sum of the selected native blocks' arbitrary
            rotations (``resources.rotation_law``, or for the binary encoding
            the upper bound of ``method.QHD._select_binary_native``), counted
            as ``cx`` is, so for quantum execution it equals the body total
            ``arbitrary_rotations`` of ``resources.run_resources``, which also
            multiplies each level's circuit by its shots and adds T
            estimates.
        unavailable: ``(field, reason)`` for every count that is None.
    """

    circuit_preparations: Count | None = 0
    circuit_attempts: Count | None = 0
    completed_circuit_attempts: Count | None = 0
    shots: Count | None = 0
    completed_shots: Count | None = 0
    data_bytes: Count | None = 0
    evolution_work: Count | None = 0
    construction_work: Count | None = 0
    synthesis_work: Count | None = 0
    table_evaluations: Count | None = 0
    support_evaluations: Count | None = 0
    objective_evaluations: Count = 0
    joint_mass_reads: Count = 0
    cx: Count | None = 0
    arbitrary_rotations: Count | None = 0
    unavailable: tuple[tuple[Text, Text], ...] = ()

    @model_validator(mode="after")
    def _availability(self):
        missing = {name for name in _RESOURCE_COUNTS if getattr(self, name) is None}
        named = [name for name, _reason in self.unavailable]
        if len(set(named)) != len(named) or set(named) != missing:
            raise ValueError("every unknown resource count needs exactly one stated reason")
        return self


_RESOURCE_COUNTS = tuple(name for name in RefinementResources.model_fields
                         if name not in ("unavailable", "schema_version", "parent_id"))


def total_resources(levels, stopped):
    """Sum the resources of the completed ``levels`` and of the ``stopped`` level, which may be None.

    Each count is summed (``_outer.total_counts``), and a count is unknown
    when any level's is, with the reasons labelled by level number.
    """
    entries = [(f"level {level.level}", level.resources) for level in levels]
    if stopped is not None:
        entries.append((f"level {len(levels) + 1}", stopped))
    counts, unavailable = total_counts(entries, _RESOURCE_COUNTS)
    return RefinementResources(**counts, unavailable=unavailable)



Box = tuple[tuple[Real, Real], ...]


class StallSplit(Record):
    """One stall split of ``BoxRefinement.stall_split="best_region"`` (``refinement._stall_split``).

    Masses are conditional on a valid one-hot outcome of the stalled level's
    distribution. Index 0 of each pair is the lower region, whose side on
    ``axis`` ends at ``coordinate``, and index 1 the upper region, whose side
    starts there. Every other side of both regions is the level box's side.

    Attributes:
        axis: Variable index of the split axis.
        valley: Grid index of the valley minimum on that axis.
        coordinate: Original coordinate of the split, the face of the
            valley's centered cell toward the discarded side, so that the
            valley cell stays in the chosen region.
        regions: The lower and the upper region.
        point_indices: Grid indices of the most probable joint grid point
            of each strict side of the valley, the indices below it and those
            above it on ``axis``, in the level's distribution. The chosen
            region also holds the valley column, so its most probable point
            can lie there instead.
        points: Those grid points in original coordinates, given as
            ``RefinementLevel.point`` gives a grid point, so for the search
            model the rounded image ``a + D u`` of the unit point, a display
            value.
        scores: Original objective F at each grid point, evaluated in the
            level's own coordinates as the level's reported grid point is
            (``RefinementLevel.objective``).
        relative_scores: The relative objective F - C at each grid point
            (``RefinementLevel.relative_objective``), the values the split
            compares.
        region_masses: Conditional probability each region held.
        chosen: 0 or 1, the region the refinement continues in, the one
            whose side has the lower relative score, on equal relative scores
            the one whose side holds more probability, then the lower one.
        discarded_mass: Conditional probability of the other region, which
            the refinement gives up.
    """

    axis: Count
    valley: Count
    coordinate: Real
    regions: tuple[Box, Box]
    point_indices: tuple[tuple[Count, ...], tuple[Count, ...]]
    points: tuple[tuple[Real, ...], tuple[Real, ...]]
    scores: tuple[Real, Real]
    relative_scores: tuple[Real, Real]
    region_masses: tuple[Nonnegative, Nonnegative]
    chosen: Literal[0, 1]
    discarded_mass: Nonnegative

    @model_validator(mode="after")
    def _regions(self):
        """Require regions that meet at the coordinate, a point inside each, and the lower-scoring choice."""
        (lower, upper), d = self.regions, len(self.regions[0])
        if not d or len(upper) != d or not self.axis < d or any(
            len(field) != d for pair in (self.points, self.point_indices) for field in pair
        ):
            raise ValueError("a stall split needs one entry per variable and an axis among them")
        if any(lower[j] != upper[j] for j in range(d) if j != self.axis) or not (
            lower[self.axis][0] < lower[self.axis][1] == self.coordinate == upper[self.axis][0] < upper[self.axis][1]
        ):
            raise ValueError("the split regions share every side but the split axis, which they divide at the "
                             "split coordinate")
        # Each scored point lies on its side of the cut. The level grid resolves every face strictly
        # between adjacent coordinates (refinement._resolved), while an end coordinate of a grid with
        # boundary points can round outside the box, so only the split axis is compared.
        if not self.points[0][self.axis] < self.coordinate < self.points[1][self.axis]:
            raise ValueError("the scored points lie on either side of the split coordinate")
        other = 1 - self.chosen
        if (self.relative_scores[self.chosen] > self.relative_scores[other]
                or self.discarded_mass != self.region_masses[other]):
            raise ValueError("the refinement continues in the lower-scoring region and discards the other")
        return self


class RefinementLevel(Record):
    """One completed refinement level: its box, QHD solve, reported point and next box.

    Masses are conditional on a valid outcome, which is every outcome of the
    binary encoding and a one-hot outcome with one excitation per variable.
    Coordinates are in the original variables, taken from the level's grid
    (``refinement._coordinates``), except ``unit_point``, which is in the
    search model's unit coordinates. ``refinement._axis_interval`` defines the
    intervals and axis masses, ``refinement._joint_mass_bound`` the joint
    bound and ``refinement._next_box`` the next box.

    Attributes:
        level: Level number, starting at 1.
        box: The level's box, one ``(lower, upper)`` pair per variable.
        spacing: Grid spacing h of each variable in original coordinates,
            the level grid's spacing for the physical model and the side
            length times the unit grid's spacing, rounded once, for the
            search model.
        energy_scale: E, the sum of the ranges of the unscaled level
            objective's stored support tables, rounded up
            (``_outer.range_bound``), an upper bound on the objective's
            range over the level grid. The search model divides by it, and
            the physical model records it for ``conditioning``.
        potential_gain: kappa, the factor of the normalized potential that the
            level solved (``BoxRefinement.gain``). None for the
            physical model.
        energy_shift: c, the constant term of the unscaled level objective
            plus the minima of its stored support tables, summed exactly and
            rounded once to binary64. The search model subtracts its exact
            value, so that ``(F - c)/E`` lies in [0, 1] on the grid and the
            solved potential ``kappa (F - c)/E`` in ``[0, kappa]``, up to the
            rounding that ``refinement._level_problem`` describes. None for
            the physical model.
        table_magnitude: Sum over those support tables of their largest
            absolute value (``refinement._table_magnitude``).
        spawn_key: Spawn key of the level's child of the refinement's
            ``numpy.random.SeedSequence``.
        plan_id: Content identity of the level's QHD Plan.
        result_id: Content identity of the level's QHD result.
        run_id: Identity of the Run that produced it.
        logical_width: Register width of the level's Plan, ``d*K`` for the
            one-hot encoding and ``d*b`` with ``K = 2**b`` for the binary one.
        restricted_dimension: Valid grid size ``K**d``.
        valid_mass: Unconditional valid mass of the level's result.
        valid_count: The result's ``valid_count`` for counts, the integer
            number S_z of valid decoded outcomes of the finite-shot statement
            of Proposition 49 in docs/mathematics.md, or None for exact
            readout.
        returned_shots: The result's ``returned_shots`` for counts, the raw
            number of returned draws among which ``valid_count`` decoded to a
            grid point, or None for exact readout.
        mode_status: Mode-resolution status of this level's QHD readout among its
            positive observed valid points. Copied from the inner Result regardless
            of point_rule. It describes the readout's tie representative, including
            when this level reports a best-observed point or a mean instead. It does
            not certify the choice of marginal intervals or a stall-split region.
        point_indices: Grid indices of the reported point, or None when
            ``mode_or_mean`` reported the mean position.
        point: Reported point in original coordinates. For the physical
            model it is the grid coordinate or mean at which ``objective``
            was evaluated. For the search model it is ``a + D u`` for
            ``u = unit_point``, rounded once to binary64, a display value. F
            at this rounded point can differ from ``objective`` or be
            undefined.
        unit_point: Search model only: the unit coordinates u of the
            reported grid point or mean position, at which the level
            objective ``F(a + D u)`` was evaluated and whose grid point
            carries ``point_probability``. None for the physical model.
        point_probability: Unconditional probability of the reported grid
            point, or None for the mean position.
        objective: Original objective F at the reported location, which is
            ``point`` for the physical model and the exact affine image
            ``a + D u`` of ``unit_point`` for the search model, up to the
            evaluation's rounding and, for a ``Float`` coefficient, SymPy's
            rounding in the substitution (``refinement._unit_objective``).
            At a grid point both are read from the level's stored tables,
            and at an off-grid mean both are evaluated term by term
            (``refinement._tabulated_objective``).
        relative_objective: F - C at the same location, the value by which
            the refinement compares points. C is one exact constant for the
            whole refinement, the constant term of the first level's
            decomposition of the objective. Each level forms its own constant
            term minus C exactly and adds it to the term values once, so a
            large constant of F, which can round the objective's variation
            out of ``objective``, does not reach this value when every
            coefficient of F is an exact SymPy number. A binary64 ``Float``
            coefficient makes SymPy round each constant term at the size of
            that constant first (``refinement._tabulated_objective``).
        mean_unavailable: Why ``mode_or_mean`` kept the grid point because
            the objective at the mean position was not finite and real
            (``_outer.mean_point``), or None.
        intervals: Kept index interval ``(first, last)`` of each variable.
        axis_masses: Conditional marginal mass m_j of each interval.
        joint_mass_bound: ``max(0, 1 - sum_j (1 - m_j))``, a lower bound on
            the conditional mass of the joint box that holds for every joint
            distribution with these marginals.
        joint_mass: Conditional mass of the joint box computed from the
            joint observations, or None when the result keeps only
            marginals. The joint box and both masses are those of the
            intervals, so at a split level they describe the level box, and
            ``split.region_masses`` gives the chosen region's probability.
        joint_mass_kind: ``"exact"`` for exact probabilities or amplitudes,
            ``"empirical"`` for counts, or None with ``joint_mass``.
        next_box: Box of the next level under the box rule, or the region
            that ``split`` chose.
        split: The stall split of this level, or None. A level splits only
            when its box rule returned the level box itself.
        split_declined: With ``stall_split="best_region"``, why a level
            whose box rule returned the level box itself did not split: the
            level is the last of ``max_levels``, the split budget is spent,
            the next level cannot run, the readout has no tie window, or no
            valley passes the resolution screen
            (``refinement._valley_admission``). None otherwise, and always
            None with ``stall_split="none"``. It explains the stop and does
            not replace ``BoxRefinementResult.termination``.
        initial_state: With ``level_initial_state="best_point_gaussian"``,
            the ``GaussianState`` that this level's Plan started from, in the
            coordinates of the level problem, the unit coordinates for the
            search model (``refinement._best_point_gaussian``). None for the
            first level and with ``"configured"``, where the level started
            from ``QHD.initial_state``.
        resources: Counted work of this level, including the two objective
            evaluations of a split.
    """

    level: PositiveInt
    box: Box
    spacing: tuple[Annotated[Real, Field(gt=0)], ...]
    energy_scale: Annotated[Real, Field(gt=0)]
    potential_gain: Annotated[Real, Field(gt=0)] | None
    energy_shift: Real | None
    table_magnitude: Nonnegative
    spawn_key: tuple[Count, ...]
    plan_id: ContentID
    result_id: ContentID
    run_id: Text
    logical_width: PositiveInt
    restricted_dimension: PositiveInt
    valid_mass: Nonnegative
    valid_count: Count | None
    returned_shots: Count | None
    mode_status: ModeStatus
    point_indices: tuple[Count, ...] | None
    point: tuple[Real, ...]
    unit_point: tuple[Real, ...] | None = None
    point_probability: Nonnegative | None
    objective: Real
    relative_objective: Real
    mean_unavailable: Text | None = None
    intervals: tuple[tuple[Count, Count], ...]
    axis_masses: tuple[Nonnegative, ...]
    joint_mass_bound: Nonnegative
    joint_mass: Nonnegative | None
    joint_mass_kind: Literal["exact", "empirical"] | None
    next_box: Box
    split: StallSplit | None = None
    split_declined: Text | None = None
    initial_state: GaussianState | None = None
    resources: RefinementResources

    @model_validator(mode="after")
    def _domain(self):
        """Require one entry per variable, ordered intervals, a nonempty next box inside the box, a split of it,
        and the integer counts of a counts readout.
        """
        d = len(self.box)
        if not d or any(
            len(field) != d
            for field in (self.spacing, self.point, self.intervals, self.axis_masses, self.next_box)
        ):
            raise ValueError("refinement level fields need one entry per variable")
        if self.point_indices is not None and len(self.point_indices) != d:
            raise ValueError("refinement level point indices need one entry per variable")
        if self.unit_point is not None and len(self.unit_point) != d:
            raise ValueError("refinement level unit point needs one entry per variable")
        if (self.point_indices is None) != (self.point_probability is None):
            raise ValueError("a grid point has indices and a probability, the mean position neither")
        if (self.joint_mass is None) != (self.joint_mass_kind is None):
            raise ValueError("a direct joint mass needs its kind")
        if any(first > last for first, last in self.intervals):
            raise ValueError("a kept interval runs from its first to its last index")
        if any(
            not lower <= new_lower < new_upper <= upper
            for (lower, upper), (new_lower, new_upper) in zip(self.box, self.next_box, strict=True)
        ):
            raise ValueError("the next box must be a nonempty box inside the level box")
        if self.split is not None and (
            len(self.split.regions[0]) != d
            or self.split.regions[0][self.split.axis][0] != self.box[self.split.axis][0]
            or self.split.regions[1][self.split.axis][1] != self.box[self.split.axis][1]
            or any(side != self.box[j] for j, side in enumerate(self.split.regions[0]) if j != self.split.axis)
            or self.next_box != self.split.regions[self.split.chosen]
        ):
            raise ValueError("a stall split divides the level box, and the next box is its chosen region")
        if self.split_declined is not None and (self.split is not None or self.next_box != self.box):
            raise ValueError("only a stalled level without a split records why it did not split")
        # A completed counts level copies the integer totals of its Result, which
        # has a positive valid point, and valid_mass is their quotient rounded once
        # (QHDAnalysis.valid_count). Exact readout has neither count, and the
        # direct joint mass of counts is empirical.
        if (self.valid_count is None) != (self.returned_shots is None) or self.valid_count is not None and not (
            0 < self.valid_count <= self.returned_shots and self.valid_mass == self.valid_count / self.returned_shots
        ):
            raise ValueError("a counts level records 0 < valid_count <= returned_shots with valid_mass their "
                             "quotient, and an exact-readout level neither count")
        if self.joint_mass_kind is not None and (self.joint_mass_kind == "empirical") != (self.valid_count is not None):
            raise ValueError("a direct joint mass is empirical exactly for counts")
        return self

    @property
    def conditioning(self):
        """Return ``table_magnitude / energy_scale``, the factor by which table rounding reaches the model.

        The search model divides tables of size up to ``table_magnitude`` by
        ``energy_scale``, so an evaluation error of a few ``u`` relative to
        the table values becomes an error of about ``4 u`` times this factor
        in its normalized potential, with ``u = 2**-53``
        (``refinement._level_problem``). That estimate illustrates simple
        expressions and is not a bound. The resolution stop
        (``refinement._resolution_stop``) does not bound this factor either,
        so a level can pass it with a factor near ``1/u``. For the physical
        model it measures how much of the level's potential variation can be
        rounding.
        """
        return self.table_magnitude / self.energy_scale


Termination = Literal[
    "level_limit",
    "box_unchanged",
    "width_floor",
    "no_improvement",
    "budget_exhausted",
    "flat_objective",
    "unresolved_objective",
    "inner_failed",
    "no_valid_point",
    "split_limit",
]

# What each stopping reason means, read from the record alone. A stopping
# reason is an operational stop, not a convergence certificate.
_TERMINATION_TEXT = {
    "level_limit": "The configured maximum number of levels completed.",
    "box_unchanged": "The next box equals the last solved box.",
    "split_limit": "The box stalled after the allowed number of stall splits.",
    "width_floor": "The next level's geometry failed its resolution rule.",
    "no_improvement": ("The configured number of consecutive levels gave no strict improvement in the recorded "
                       "relative objective."),
    "budget_exhausted": "The remaining cumulative limits could not fund a level.",
    "flat_objective": "A level's evaluated support tables had zero range.",
    "unresolved_objective": "A level's positive table range did not exceed its evaluation-resolution rule.",
    "inner_failed": "An inner planning or execution failed after completed work.",
    "no_valid_point": "An inner Result supplied no positive valid point.",
}

# How each RefinementResources count may be read. An integer count is not by
# itself an exact count of executed operations.
_RESOURCE_TEXT = {
    "circuit_preparations": "exact count of recorded circuit preparations, not of shots or executed gates",
    "circuit_attempts": "exact count of recorded acquisition attempts of every status, not of shots or executed "
                        "gates",
    "completed_circuit_attempts": "exact count of recorded attempts that completed, not of shots or executed gates",
    "shots": "raw-shot reservations over attempts of every status, requested and reserved budget accounting",
    "completed_shots": "requested and reserved raw shots of the completed attempts, not the returned shots",
    "data_bytes": "recorded Run data accounting, not the archive size or peak memory",
    "evolution_work": "reserved work units of the classical kernel's model, not measured operations or wall time",
    "construction_work": "reserved construction work units, not measured operations or wall time",
    "synthesis_work": "reserved exact-synthesis work units, not measured operations or wall time",
    "table_evaluations": "recorded grid tuples evaluated for the unscaled level objectives' tables",
    "support_evaluations": "recorded grid tuples evaluated for the solved level objectives' tables",
    "objective_evaluations": "recorded evaluations of the objective by the refinement itself",
    "joint_mass_reads": ("joint-observation read charge: decoded entries once, a kept dense grid once and one "
                         "kept-state length per streamed pass"),
    "cx": "upper bound summed over prepared circuit bodies, not multiplied by shots",
}


def _number(value):
    """Display a stored scalar with eight significant digits, or ``unavailable`` for None."""
    return "unavailable" if value is None else format(value, ".8g")


def _point(values):
    """Display a stored point as a bracketed list of ``_number`` entries."""
    return "[" + ", ".join(_number(value) for value in values) + "]"


class BoxRefinementResult(Record):
    """Completed levels, best point and stopping reason of one box refinement.

    ``best_level`` is the completed level with the least recorded relative
    objective (``RefinementLevel.relative_objective``), the earlier level on
    ties. ``objective`` reports its original objective. Levels completed before a
    stop stay in ``levels``. ``resources`` adds the work of every completed
    level and of the level that stopped the refinement (``stopped_resources``),
    so what a failed level's tables and Run charged is counted. A planning
    that raises reports no work, so its partial work is not counted.
    ``results`` gives the QHD results of the completed levels. They are live
    objects, outside the record's identity.

    The reported points are finite-grid search results. Neither the joint
    mass bound nor a small box certifies that the global minimizer lies in
    the box or that the best point is a continuous or global optimum.

    ``str(result)`` and ``report`` read the stored fields only. They plan,
    evaluate, acquire and load nothing, reconstruct no state and call no
    resource estimate, so a record validated from JSON alone can be
    summarized too.

    Record size. Count the scalar JSON leaves of the exported record, each
    number, Boolean, string or null once, including every nested record's
    schema, parent and content identities and every allowed unknown-count
    reason pair, and excluding object keys and containers. An unsplit
    level has at most ``13 d + 74 + s`` leaves for d variables and s entries
    in its spawn key. Its dimension-dependent fields contribute ``13 d``
    from ``box`` and ``next_box`` (2d each), ``intervals`` (2d), the center
    and widths of a best-point ``initial_state`` (d each), and ``spacing``,
    ``point_indices``, ``point``, ``unit_point`` and ``axis_masses`` (d
    each). Its 23 other scalar fields include ``mode_status``, the null
    ``split`` and scalar ``split_declined``, and ``initial_state`` adds at
    most its kind and 3 identities. There are 3 identity leaves and at most
    44 resource leaves, from 15 counts, 3 resource identities and at most
    13 reason pairs. A ``StallSplit`` has ``8 d + 14`` leaves, replacing one
    null and adding ``8 d + 13``. Standalone refinement has s = 1 and the
    augmented-Lagrangian layer s = 2, so the unsplit bounds are
    ``13 d + 75`` and ``13 d + 76``, and the split bounds are ``21 d + 88``
    and ``21 d + 89``, respectively.

    The fields outside ``levels`` add at most ``2 g + 147`` leaves. This
    counts 11 for the eight scalar result fields and three identities,
    14 for the options, 88 for total and stopped resources, and
    ``34 + 2 g`` for QHD. The QHD term includes 15 direct scalar settings,
    including ``encoding`` and ``kinetic_model``, 3 identities, 5 schedule
    leaves, ``4 + 2 g`` initial-state leaves and 7 ``BinarySynthesis``
    leaves. The binary-synthesis record has 4 settings and 3 identities
    and is exported for both encodings. Here g is the common length of a
    ``GaussianState``'s center and widths, or zero for other initial states.
    With n completed levels, b split levels and spawn keys of length at
    most two, the bound is
    ``2 g + 147 + n (13 d + 76) + b (8 d + 13)``. Use 75 in place of 76
    for standalone refinement. ``options.max_levels`` bounds n, and
    ``b <= min(n, options.max_splits)``, with b = 0 when splits are disabled.
    Each completed level admits at least d K running-total work
    (``QHD._admit_symbolic_work``), where K is ``qhd.num_grid_points``.
    For d >= 1 and K >= 2, ``13 d + 76 <= 45 d K`` without a split and
    ``21 d + 89 <= 55 d K`` with a split, since ``45 d K - 13 d >= 77 d >= 76``
    and ``55 d K - 21 d >= 89 d >= 89``. The fixed fields remain when no
    level completes. These counts give no byte allowance because text
    lengths and integer digit counts vary. Identity JSON bytes are
    unbudgeted, and live QHD results are outside the record's identity.

    Attributes:
        problem_id: Content identity of the refined Optimization.
        qhd: QHD configuration used at every level.
        options: Refinement options.
        execution: ``"quantum"`` or ``"classical"``, as for ``nwqlib.solve``.
        shots: Shots per level, or None for exact readout.
        seed_entropy: Entropy of the ``numpy.random.SeedSequence`` whose
            children seed the levels.
        levels: Completed levels in order.
        best_level: Level number of the best point, or None without a
            completed level.
        termination: Why refinement stopped. Before each level it checks, in
            this order, ``level_limit`` (``max_levels`` levels completed),
            ``box_unchanged`` (the next box equals the last box, which with
            ``stall_split="best_region"`` means that no axis of the last
            level had a resolved valley), ``split_limit`` (the next box
            equals the last box after ``max_splits`` stall splits),
            ``width_floor`` (the box to solve has a side at or below its
            width floor, ``refinement._resolved`` and, for the physical model,
            the grid owner's admission, ``refinement._level_geometry``),
            ``no_improvement``
            (``max_no_improve`` consecutive levels without a strict decrease
            of the best relative objective) and ``budget_exhausted`` (the remaining
            cumulative limits cannot fund a level), and reports the first
            that holds. With ``stall_split="best_region"`` a stalled last
            level whose next level could not run records no split and
            reports ``no_improvement`` or ``budget_exhausted``, or
            ``width_floor`` when neither region of its resolved valley
            passes the width floor (``refinement.refine_box``). Within a level, ``flat_objective`` means that the level
            objective's tables show no variation (E = 0),
            ``unresolved_objective`` that their ranges sum to a positive value
            at most the units in the last place of the tables' largest
            values, too small to tell from evaluation error (``refinement._resolution_stop``, a
            resolution rule), ``inner_failed`` that
            planning or solving a level after the first raised (the first
            level's error propagates, ``_outer.inner_failure``), and ``no_valid_point`` that
            the level's QHD result has no valid grid point.
        failure: Exception type and message for ``inner_failed``, the
            result's missing-data reasons for ``no_valid_point``, the limits
            that fell short for ``budget_exhausted`` (``_outer.round_limits``),
            otherwise None.
        stopped_resources: Work spent on the level that stopped the
            refinement, or None when it stopped before that level began.
        resources: Work of all levels, completed and stopped.
        max_plannings: Largest number of QHD plannings that ``options``
            allows, two per search-model level (the table stage and the
            solved objective) and one per physical level, known before the
            first level. Each operation that one planning admits, namely the
            symbolic expansion, the running total of the initial state and
            tables (``QHD._admit_symbolic_work``), a kept state, a classical
            kernel and a native construction, is at most ``qhd.max_work`` units and
            ``qhd.max_bytes`` bytes, so
            the table, support, kernel and construction work fields of
            ``resources`` are each at most ``max_plannings * qhd.max_work``.
            The refinement's own objective evaluations and joint-mass reads,
            and the second expansion of each level objective that those
            evaluations use (``refinement._level_decomposition``), are
            outside this bound.
    """

    problem_id: ContentID
    qhd: QHD
    options: BoxRefinement
    execution: Literal["quantum", "classical"]
    shots: PositiveInt | None
    seed_entropy: Count
    levels: tuple[RefinementLevel, ...]
    best_level: PositiveInt | None
    termination: Termination
    failure: Text | None
    stopped_resources: RefinementResources | None
    resources: RefinementResources
    max_plannings: PositiveInt
    _results: tuple = PrivateAttr(default=())
    _problem: object = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _levels(self):
        """Check level numbering and chaining, the best level, the resource total, the readout kind of every
        level and the rule that reports a mean position.
        """
        for number, level in enumerate(self.levels, 1):
            if level.level != number:
                raise ValueError("refinement levels are numbered from 1 in order")
            if number > 1 and level.box != self.levels[number - 2].next_box:
                raise ValueError("each refinement level solves the previous level's next box")
        best = None
        for level in self.levels:
            if best is None or level.relative_objective < best.relative_objective:
                best = level
        if (None if best is None else best.level) != self.best_level:
            raise ValueError("best level is the least relative objective, earliest on ties")
        if total_resources(self.levels, self.stopped_resources) != self.resources:
            raise ValueError("refinement resources are the sum over its levels")
        if len(self.levels) > self.options.max_levels:
            raise ValueError("refinement completed more levels than max_levels")
        search = self.options.scaling == "search_model"
        if any(level.potential_gain != self.options.gain for level in self.levels):
            raise ValueError("each search-model level solves with the options' potential gain, a physical "
                             "level with none")
        if any((level.unit_point is None) == search for level in self.levels):
            raise ValueError("each search-model level records the unit point it evaluated, a physical level "
                             "none")
        gaussian = self.options.level_initial_state == "best_point_gaussian"
        if any((level.initial_state is not None) != (gaussian and level.level > 1)
               or level.initial_state is not None and len(level.initial_state.center) != len(level.box)
               for level in self.levels):
            raise ValueError("with level_initial_state='best_point_gaussian' every level after the first records "
                             "its Gaussian of one entry per variable, and otherwise no level records one")
        if (self.failure is None) != (self.termination not in {"inner_failed", "no_valid_point", "budget_exhausted"}):
            raise ValueError("a failure text belongs to inner_failed, no_valid_point and budget_exhausted")
        if any((level.valid_count is None) != (self.shots is None) for level in self.levels):
            raise ValueError("every level of a refinement with shots records counts, and with exact readout none")
        if any(level.point_indices is None for level in self.levels) and self.options.point_rule != "mode_or_mean":
            raise ValueError("only mode_or_mean reports a mean position instead of a grid point")
        splits = sum(level.split is not None for level in self.levels)
        splitting = self.options.stall_split == "best_region"
        if any((level.split_declined is not None) != (splitting and level.split is None and level.next_box == level.box)
               for level in self.levels):
            raise ValueError("with stall_split='best_region' every stalled level without a split records why, and "
                             "with 'none' no level does")
        if splits > (self.options.max_splits if splitting else 0) or (self.termination == "split_limit") != (
            splitting and splits == self.options.max_splits and self.termination in {"box_unchanged", "split_limit"}
        ):
            raise ValueError("stall splits need stall_split='best_region', at most max_splits of them, and a stall "
                             "after the last one stops with split_limit")
        return self

    @property
    def results(self):
        """QHD results of the completed levels, in level order."""
        return self._results

    @property
    def problem(self):
        """The refined Optimization with its live SymPy objects, or None for a record validated from JSON alone.

        ``refine_box``, both paths of ``resume_box_refinement`` and
        ``load_box_refinement`` attach it. A refinement nested in an
        augmented-Lagrangian record has neither this problem nor its level
        Results, which ``ConstrainedQHDResult`` holds and saves with its run.
        A search-model level Plan solves a transformed objective on the unit
        box and cannot stand in for the original problem.
        """
        return self._problem

    @property
    def best(self):
        """The best completed level, or None."""
        return None if self.best_level is None else self.levels[self.best_level - 1]

    @property
    def candidate(self):
        """Best point in original coordinates, or None.

        For the search model it is a display value (``RefinementLevel.point``).
        """
        return None if self.best is None else self.best.point

    @property
    def objective(self):
        """Original objective of the best level, or None, evaluated as ``RefinementLevel.objective`` states."""
        return None if self.best is None else self.best.objective

    def __str__(self):
        """Summarize the refinement from its stored fields, without planning, evaluating or acquiring anything.

        The lines state the stop with its meaning and stored detail, the
        selected point and how it was selected, each level's interval mass
        with its readout kind, its chosen split-region mass and its readout
        mode status, and the CX and shot accounting (``report`` gives the
        meaning of every count).
        """
        return "\n".join(self._summary_lines())

    def _summary_lines(self, failure_probability=None):
        """Return the display lines of ``__str__`` and, with a failure probability, the radius of each level.

        Every line reads stored fields. The radius is ``_radius`` of the
        level's valid count.
        """
        options = self.options
        lines = [f"Box refinement stopped: {self.termination}. Completed levels: {len(self.levels)} of at most "
                 f"{options.max_levels}.", _TERMINATION_TEXT[self.termination]]
        details = [text.rstrip(".") for text in (
            self.failure, self.levels[-1].split_declined if self.levels else None) if text is not None]
        if details:
            lines.append(f"Stop detail: {'; '.join(details)}.")
        best = self.best
        if best is None:
            lines.append("Selected point: unavailable (no completed level).")
        else:
            lines.append(f"Selected point from level {best.level}: {_point(best.point)}. Recorded objective: "
                         f"{_number(best.objective)}.")
            lines.append("Selection compares recorded relative objectives and chooses the earliest level on ties.")
            kind = "a level grid point" if best.point_indices is not None else "a conditional mean"
            lines.append(f"The selected point is {kind}. Its objective gives no proof of a continuous or global "
                         "optimum.")
            if options.scaling == "search_model":
                lines.append("Search-model objective evaluation uses the stored unit point. The original-coordinate "
                             "point is a rounded display value.")
        for level in self.levels:
            readout = "exact-readout" if level.returned_shots is None else "empirical"
            lines.append(f"Level {level.level} interval mass: {_number(level.joint_mass)} ({readout}). Marginal "
                         f"union-bound formula: {_number(level.joint_mass_bound)}.")
            if level.split is not None:
                lines.append(f"Level {level.level} chosen split-region {readout} mass: "
                             f"{_number(level.split.region_masses[level.split.chosen])}.")
            lines.append(f"Level {level.level} readout mode status: {level.mode_status}.")
            if failure_probability is not None and level.valid_count:
                lines.append(f"Level {level.level} Proposition 49 interval radius: approximately "
                             f"{_number(self._radius(level, failure_probability))}, using {level.valid_count} "
                             f"valid draws, horizon {options.max_levels}, and failure probability "
                             f"{failure_probability!r}, conditional on its sampling premises.")
        resources, reasons = self.resources, dict(self.resources.unavailable)
        cx = resources.cx if resources.cx is not None else f"unavailable ({reasons['cx']})"
        shots = resources.shots if resources.shots is not None else f"unavailable ({reasons['shots']})"
        if self.shots is None:
            returned = "unavailable (exact readout)"
        elif not self.levels:
            returned = "unavailable (no completed level)"
        else:
            returned = sum(level.returned_shots for level in self.levels)
        lines.append(f"CX upper bound across prepared circuit bodies: {cx}.")
        lines.append(f"Raw-shot reservations: {shots}. Returned shots at completed levels: {returned}.")
        return lines

    def _radius(self, level, failure_probability):
        """Return epsilon_z of Proposition 49 of docs/mathematics.md for ``level``, evaluated in binary64.

        Conditional on the earlier history and the validity indicators, the
        level's S_z = ``valid_count`` >= 1 valid positions are independent
        draws from one fixed level population, with the sample size chosen
        without inspecting them, and alpha and the horizon H =
        ``options.max_levels`` are fixed before the counts are observed. The
        nonwrapping interval family has K (K + 1)/2 members per axis, so the
        d axes give J = d K (K + 1)/2 marginal events. For each fixed interval
        Hoeffding's inequality gives ``Pr(abs(m_hat - m) > e) <=
        2 exp(-2 S_z e**2)``, and a union bound over the level's J events
        gives at most ``2 J exp(-2 S_z e**2)``, which is at most alpha/H for
        ``e**2 >= log(2 H J/alpha)/(2 S_z)``. Conditional averaging and a union
        bound over at most H levels give failure probability at most alpha,
        and an early stop does not change the configured H. With the positive
        integer ``A = H d K (K + 1) = 2 H J``, the exact identity
        ``log(A/alpha) = log(A) - log(alpha)`` for A > 0 and alpha > 0 gives
        ``epsilon_z = sqrt((log(A) - log(alpha))/(2 S_z))`` without the
        quotient A/alpha, which overflows binary64 for a small allowed alpha.
        Ordinary ``log`` and ``sqrt`` give an approximate evaluation, not a
        directed-rounding enclosure.
        """
        from math import log, sqrt

        d, k = len(level.box), self.qhd.num_grid_points
        numerator = log(self.options.max_levels * d * k * (k + 1)) - log(failure_probability)
        return sqrt(numerator / (2 * level.valid_count))

    def _statements(self, failure_probability):
        """Interpretation sentences that follow from the stored record only (``report``)."""
        lines = [_TERMINATION_TEXT[self.termination] + " A stopping reason is an operational stop, not a "
                 "convergence certificate."]
        if self.best is not None:
            lines.append("The selected level has the least recorded relative objective among the completed levels, "
                         "the earlier level on ties. This is not necessarily the smallest displayed objective, "
                         "whose rounding can conceal the comparison.")
            lines.append("A selected point with grid indices is a grid point of its level. Otherwise mode_or_mean "
                         "selected the level's conditional mean position, which can lie off the grid.")
            if self.options.scaling == "search_model":
                lines.append("The displayed point is the rounded affine image of the stored unit point, and the "
                             "stored objective was evaluated in the level's own coordinates, so evaluating the "
                             "objective again at the displayed point can give another value or fail.")
        if self.levels:
            lines.append("joint_mass_bound is max(0, 1 - sum_j (1 - m_j)) evaluated in binary64 on the stored axis "
                         "masses, the lower bound of Proposition 49 of docs/mathematics.md on the conditional mass "
                         "of the product of the kept intervals for every joint distribution with those marginals. "
                         "joint_mass is the observed mass of that product event, available when the level Result "
                         "kept joint observations. Under exact readout, joint_mass_kind='exact' identifies the "
                         "nonsampled readout, not exact real arithmetic. Under counts both fields describe the "
                         "empirical distribution of the returned sample. Neither uncorrected binary64 field is a "
                         "directed-rounding population certificate.")
            lines.append("Each mass concerns its own level's population and old-grid event. Levels change the "
                         "Hamiltonian, box and initial state, so their masses cannot be multiplied to obtain the "
                         "final box's mass, and a high mass does not imply that a global minimizer lies in the box.")
            lines.append("Each level's mode status is a diagnostic of the tie representative of its QHD Result, "
                         "whatever the point rule. It gives no proof about the marginal intervals or a split region.")
        if any(level.split is not None for level in self.levels):
            lines.append("At a stall split every kept interval covers its axis, so that level's interval masses "
                         "describe the whole level box. The chosen region's stored mass is reported separately, "
                         "and discarded_mass is the other region's.")
        if any(level.valid_count is not None for level in self.levels):
            if failure_probability is None:
                lines.append("No confidence level is stored. report(failure_probability=alpha), with alpha fixed "
                             "before the counts are inspected, evaluates the interval radius of Proposition 49.")
            else:
                lines.append("The radius epsilon_z = sqrt(log(H d K (K + 1)/alpha)/(2 S_z)) uses the configured "
                             "horizon H = max_levels, J = d K (K + 1)/2 marginal interval events and the S_z valid "
                             "draws of level z, not its raw or returned shots. It is an approximate binary64 "
                             "evaluation of the conditional confidence formula of Proposition 49 of "
                             "docs/mathematics.md under its sampling premises, which this report does not verify. "
                             "On that event each kept interval's population mass is at least its exact empirical "
                             "mass minus epsilon_z, and a chosen split region's likewise. The stored masses are "
                             "rounded, so a strict bound first divides each by beta_K = (1 + u)**(K + 1)/(1 - u). "
                             "Subtracting epsilon_z from joint_mass has no justification from this event, and a "
                             "refinement nested in an augmented-Lagrangian claim needs the whole run's horizon. "
                             "Levels without valid draws have no radius.")
        lines.append("T cost is unavailable in this record-only report. run_resources(..., synthesis_epsilon=...) "
                     "gives an estimate with its model and budget.")
        return lines

    def _resource_report(self):
        """Return each stored count with its unavailable reason and its meaning (``_RESOURCE_TEXT``).

        A rotation count is the selected blocks' count only for the one-hot
        encoding with no positive stopped-level count, and an upper bound
        otherwise.
        """
        stopped = self.stopped_resources
        exact = self.qhd.encoding == "one_hot" and not (stopped is not None and stopped.arbitrary_rotations)
        meaning = dict(_RESOURCE_TEXT, arbitrary_rotations=(
            "arbitrary rotations of the selected one-hot blocks (resources.rotation_law), not multiplied by shots"
            if exact else "upper bound on arbitrary rotations, not multiplied by shots"))

        def counts(resources):
            if resources is None:
                return None
            reasons = dict(resources.unavailable)
            return {name: dict(value=getattr(resources, name), unavailable=reasons.get(name), meaning=text)
                    for name, text in meaning.items()}

        return dict(
            total=counts(self.resources), stopped=counts(stopped),
            max_plannings=dict(value=self.max_plannings, meaning="configured upper bound on QHD plannings, not the "
                               "number that completed"),
            shots_per_level=dict(value=self.shots, meaning="requested raw shots per level, None for exact readout"),
            limits=dict(max_work=self.qhd.max_work, max_bytes=self.qhd.max_bytes, meaning="requested limits of "
                        "each QHD planning's declared categories, not measured usage or whole-process bounds"),
            t_cost=dict(value=None, meaning="unavailable in this record-only report; run_resources(..., "
                        "synthesis_epsilon=...) gives an estimate with its model and budget"),
        )

    def report(self, *, failure_probability=None):
        """Return a JSON-ready report of the refinement built only from its stored fields.

        It plans, evaluates, acquires and loads nothing, reconstructs no
        state and calls no resource estimate. ``levels`` gives each level's
        readout kind, counts, intervals, masses and the event they describe,
        its chosen split-region mass and its readout mode status.
        ``resources`` gives each count with its unavailable reason and its
        meaning. Without ``failure_probability`` the report states the
        stored empirical quantities and their meaning, and ``confidence`` is
        None. With a failure probability alpha, 0 < alpha < 1, fixed before
        the counts are inspected, ``confidence`` gives the horizon H, the
        number J of marginal interval events and, for each level with valid
        draws S_z, the radius of ``_radius``, an approximate evaluation of
        the conditional confidence formula of Proposition 49 of
        docs/mathematics.md.
        """
        if failure_probability is not None and (
            isinstance(failure_probability, bool) or not isinstance(failure_probability, (int, float))
            or not 0 < failure_probability < 1
        ):
            raise ValueError("failure_probability must be a number strictly between 0 and 1, or None")
        alpha = None if failure_probability is None else float(failure_probability)
        k = self.qhd.num_grid_points
        levels = [
            dict(level=level.level, mode_status=level.mode_status,
                 readout="exact" if level.returned_shots is None else "counts",
                 valid_count=level.valid_count, returned_shots=level.returned_shots,
                 intervals=[list(interval) for interval in level.intervals], axis_masses=list(level.axis_masses),
                 joint_mass=level.joint_mass, joint_mass_kind=level.joint_mass_kind,
                 joint_mass_bound=level.joint_mass_bound,
                 interval_event=("product of the kept intervals, the whole level box at a stall split"
                                 if level.split is not None else "product of the kept intervals"),
                 split_region_mass=None if level.split is None else level.split.region_masses[level.split.chosen])
            for level in self.levels
        ]
        confidence = None if alpha is None else dict(
            failure_probability=alpha, horizon=self.options.max_levels,
            meaning="approximate evaluation of the conditional confidence formula of Proposition 49 of "
                    "docs/mathematics.md under its sampling premises",
            levels=[dict(level=level.level, valid_count=level.valid_count,
                         marginal_events=len(level.box) * k * (k + 1) // 2, radius=self._radius(level, alpha))
                    for level in self.levels if level.valid_count],
        )
        return dict(summary="\n".join(self._summary_lines(alpha)), statements=self._statements(alpha),
                    record=self.model_dump(mode="json"), levels=levels, confidence=confidence,
                    resources=self._resource_report(), inner_results=[level.result_id for level in self.levels])

    def save(self, path):
        """Write this refinement and its completed level Results to a new directory.

        The archive stores the original Optimization, the portable
        refinement record and each level Result with its own Plan. Saving
        requires the live problem and the level Results named by the record.
        It performs no planning, objective evaluation or acquisition.
        ``refinement.save_archive`` gives the layout and its checks, and
        ``load_box_refinement`` reads the archive back.

        Args:
            path (str | Path): A directory that does not exist yet.

        Returns:
            path (Path): The written directory.
        """
        from .refinement import save_archive

        return save_archive(self, path)
