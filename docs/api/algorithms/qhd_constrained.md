# QHD constrained problems and box refinement

Reference for the augmented-Lagrangian layer, which solves a `ConstrainedOptimization` by a sequence of QHD solves, and for box refinement, which repeats QHD on shrinking boxes, alone or inside each augmented-Lagrangian round. Import every name on this page from `nwqlib.algorithms.qhd`, for example `from nwqlib.algorithms.qhd import solve_augmented_lagrangian, refine_box`.

The guide's section on [constrained problems](../../algorithms/qhd.md#constrained-problems) gives the effective objective, the multiplier and penalty updates and the stopping test with their sources, Wu et al., arXiv:2605.12066v1, Eqs. (6)–(8), Rockafellar, doi:10.1007/BF01580138, and Birgin and Martínez, doi:10.1137/1.9781611973365, Algorithm 4.1 and Eqs. (4.7)–(4.9) and (10.6)–(10.8). Its [slack-variable subsection](../../algorithms/qhd.md#slack-variables-for-inequality-constraints) explains `inequality_form`, the augmented inner problem, projected readout and the cap and error scope. The guide's section on [box refinement](../../algorithms/qhd.md#box-refinement) explains how each level keeps its marginal intervals and forms the next box, what the search model solves and why its gain has no universal value, when the optional stall split applies and what it gives up, and the order of the stops. The [QHD](qhd.md) page documents the `QHD` configuration that every round and level uses. `solve_augmented_lagrangian` and `refine_box` each have a complete example below.

| Task | Entry |
| --- | --- |
| Solve a constrained problem | [`solve_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.solve_augmented_lagrangian] with [`AugmentedLagrangian`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian] options |
| Inspect round 0 and its cost before running | [`plan_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.plan_augmented_lagrangian] |
| Continue an interrupted run, or reopen a saved one | [`resume_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.resume_augmented_lagrangian], [`load_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.load_augmented_lagrangian] |
| Read the best point, its feasibility and the multipliers | [`ConstrainedQHDResult`][nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult], [`ALEvaluation`][nwqlib.algorithms.qhd.constrained_records.ALEvaluation] |
| Compare the best point with the evaluated grid minimum | [`constrained_grid_minimum`][nwqlib.algorithms.qhd.constrained.constrained_grid_minimum] |
| Refine the box of an unconstrained problem | [`refine_box`][nwqlib.algorithms.qhd.refinement.refine_box] with [`BoxRefinement`][nwqlib.algorithms.qhd.refinement_records.BoxRefinement] options |
| Continue or reopen a refinement | [`resume_box_refinement`][nwqlib.algorithms.qhd.refinement.resume_box_refinement], [`load_box_refinement`][nwqlib.algorithms.qhd.refinement.load_box_refinement] |
| Read the levels of a refinement | [`BoxRefinementResult`][nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult], [`RefinementLevel`][nwqlib.algorithms.qhd.refinement_records.RefinementLevel] |

## Solve a constrained problem

::: nwqlib.algorithms.qhd.constrained.solve_augmented_lagrangian
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.MultiplierBounds
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained.plan_augmented_lagrangian
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained.resume_augmented_lagrangian
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained.load_augmented_lagrangian
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained.constrained_grid_minimum
    options:
      heading_level: 3

## Read a constrained run

`ConstrainedQHDResult.record` is an `AugmentedLagrangianRecord`, which keeps one `ALIteration` per round, with the round's point and update in `ALEvaluation` and its counts in `ALResources`. Under `inequality_form="slack"` or `"auto"`, `InnerRepresentation`, `SlackAxis` and `FormTrial` record each round's resolved choice. Feasibility, complementarity and the stopping status describe the chosen point and its multipliers. A stopping status does not assess optimality.

::: nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangianRecord
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.ALIteration
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.ALEvaluation
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.ALResources
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.ConstraintPreprocessing
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.AbsorbedBound
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.InnerRepresentation
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.SlackAxis
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.FormTrial
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.constrained_records.ConstrainedGridMinimum
    options:
      heading_level: 3

## Refine the box

::: nwqlib.algorithms.qhd.refinement.refine_box
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.refinement_records.BoxRefinement
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.refinement.resume_box_refinement
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.refinement.load_box_refinement
    options:
      heading_level: 3

## Read a refinement

::: nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.refinement_records.RefinementLevel
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.refinement_records.RefinementResources
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.refinement_records.StallSplit
    options:
      heading_level: 3

## Limits

- Neither layer is a global optimizer. Their points are points of finite grids, except an off-grid mean under `mode_or_mean`, and no stopping status is a statement about the continuous problem.
- A stopping status does not assess optimality. `feasible_complementary` is not convergence in Algencan's sense, because projected stationarity is not a stopping condition.
- Finite-grid multipliers are dual iterates of the grid problem, not the continuous KKT multipliers, and the reported multipliers belong to the last round, not to the best point.
- Neither the joint mass bound of a refinement level nor a small box certifies that the global minimizer lies in the box.
- `constrained_grid_minimum` covers the preprocessed box's grid only and refuses a run with box refinement. Its gap is an evaluated-value diagnostic, not a bound for the mathematical objective.
- The work and byte bounds of the record sum per-category limits. They are not runtime or peak-memory bounds.
