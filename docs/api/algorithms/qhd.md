# QHD API

Start with the [QHD guide](../../algorithms/qhd.md) for the original ordered box, selected model and returned candidate, and its section on [constrained problems](../../algorithms/qhd.md#constrained-problems) for the augmented-Lagrangian layer, with or without [box refinement](../../algorithms/qhd.md#box-refinement) in each round. Its [scientific model](../../algorithms/qhd.md#scientific-model-and-ordering) section gives the points, spacing and kinetic links of the Dirichlet interior, Dirichlet endpoint and periodic grids that `boundary` and `include_boundary_points` select, and its [source map](../../algorithms/qhd.md#source-map) names the paper, version and equation behind each step of the Method or marks the step as standard or NWQLib's own.

::: nwqlib.algorithms.qhd.method.QHD

The [split-step flavor](../../algorithms/qhd.md#split-step-classical-flavor) describes the kinetic models that `kinetic_model` selects, their eigenvalues and the transforms that apply them.

The [binary encoding](../../algorithms/qhd.md#binary-encoding) section explains the register, the kinetic models, the circuit choices of `BinarySynthesis` and their CX counts.

::: nwqlib.algorithms.qhd.binary.BinarySynthesis

The [schedule records](../../algorithms/qhd.md#schedules-and-coefficient-rule) give the weights of `H(t) = a(t) K + b(t) V`, their point values and their interval integrals. `QHD.coefficient_rule` selects whether a step uses the point values at its midpoint or the step averages of the integrals (`schedules.step_weights`).

::: nwqlib.algorithms.qhd.schedules.QuadraticSchedule

::: nwqlib.algorithms.qhd.schedules.CubicSchedule

::: nwqlib.algorithms.qhd.schedules.ShiftedCubicSchedule

The [initial states](../../algorithms/qhd.md#initial-states-and-preparation) are products of nonnegative per-variable amplitude vectors, and `QHD.initial_state_preparation` selects their native recipe. `KineticGroundState` is the default, and `GaussianState` is a warm start, which box refinement can center at the best point found so far.

::: nwqlib.algorithms.qhd.initial_state.UniformState

::: nwqlib.algorithms.qhd.initial_state.KineticGroundState

::: nwqlib.algorithms.qhd.initial_state.GaussianState

The guide's [readout](../../algorithms/qhd.md#readout) section explains the best observed candidate, the most probable point and the tie window that `QHDAnalysis` reports.

::: nwqlib.algorithms.qhd.records.QHDAnalysis

[Explicit verification](../../algorithms/qhd.md#explicit-verification) describes what each `QHDVerification` comparison checks and what it does not establish.

::: nwqlib.algorithms.qhd.records.QHDVerification

## Fault-tolerant resources

The guide's section on [fault-tolerant resources](../../algorithms/qhd.md#fault-tolerant-resources) states what the rotation law counts, how the synthesis budget is split and which error sources are bounded.

::: nwqlib.algorithms.qhd.resources.circuit_resources

::: nwqlib.algorithms.qhd.resources.run_resources

::: nwqlib.algorithms.qhd.evolution_bounds.evolution_bound

::: nwqlib.algorithms.qhd.resources.QHDCircuitResources

::: nwqlib.algorithms.qhd.resources.QHDSynthesisProjection

::: nwqlib.algorithms.qhd.resources.QHDErrorSource

::: nwqlib.algorithms.qhd.resources.QHDRunResources

::: nwqlib.algorithms.qhd.resources.QHDRunEntry

::: nwqlib.algorithms.qhd.evolution_bounds.QHDEvolutionBound

## Constrained problems

The guide's section on [constrained problems](../../algorithms/qhd.md#constrained-problems) gives the effective objective, the multiplier and penalty updates and the stopping test with their sources, Wu et al., arXiv:2605.12066v1, Eqs. (6)–(8), Rockafellar, doi:10.1007/BF01580138, and Birgin and Martínez, doi:10.1137/1.9781611973365, Algorithm 4.1 and Eqs. (4.7)–(4.9) and (10.6)–(10.8). `solve_augmented_lagrangian` returns a `ConstrainedQHDResult`. Its `AugmentedLagrangianRecord` keeps one `ALIteration` per round, with the round's point and update in `ALEvaluation` and its counts in `ALResources`.

The guide's [slack-variable subsection](../../algorithms/qhd.md#slack-variables-for-inequality-constraints) explains `inequality_form`, the augmented inner problem, projected readout and the cap and error scope. `plan_augmented_lagrangian` returns round 0's preprocessing, inequality representation and inner Plan without creating a Run. `InnerRepresentation`, `SlackAxis` and `FormTrial` expose the resolved choice.

::: nwqlib.algorithms.qhd.constrained.solve_augmented_lagrangian

::: nwqlib.algorithms.qhd.constrained.plan_augmented_lagrangian

::: nwqlib.algorithms.qhd.constrained.resume_augmented_lagrangian

::: nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult

::: nwqlib.algorithms.qhd.constrained.load_augmented_lagrangian

::: nwqlib.algorithms.qhd.constrained.constrained_grid_minimum

::: nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian

::: nwqlib.algorithms.qhd.constrained_records.MultiplierBounds

::: nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangianRecord

::: nwqlib.algorithms.qhd.constrained_records.ConstraintPreprocessing

::: nwqlib.algorithms.qhd.constrained_records.InnerRepresentation

::: nwqlib.algorithms.qhd.constrained_records.SlackAxis

::: nwqlib.algorithms.qhd.constrained_records.FormTrial

::: nwqlib.algorithms.qhd.constrained_records.AbsorbedBound

::: nwqlib.algorithms.qhd.constrained_records.ALIteration

::: nwqlib.algorithms.qhd.constrained_records.ALEvaluation

::: nwqlib.algorithms.qhd.constrained_records.ALResources

::: nwqlib.algorithms.qhd.constrained_records.ConstrainedGridMinimum

## Box refinement

The guide's section on [box refinement](../../algorithms/qhd.md#box-refinement) explains how each level keeps its marginal intervals and forms the next box, what the search model solves and why its gain has no universal value, when the optional stall split applies and what it gives up, and the order of the stops.

::: nwqlib.algorithms.qhd.refinement.refine_box

::: nwqlib.algorithms.qhd.refinement.resume_box_refinement

::: nwqlib.algorithms.qhd.refinement.load_box_refinement

::: nwqlib.algorithms.qhd.refinement_records.BoxRefinement

::: nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult

::: nwqlib.algorithms.qhd.refinement_records.RefinementLevel

::: nwqlib.algorithms.qhd.refinement_records.RefinementResources

::: nwqlib.algorithms.qhd.refinement_records.StallSplit
