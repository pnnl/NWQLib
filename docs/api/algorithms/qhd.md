# QHD

<a id="qhd-api"></a>Use an `Optimization` with `QHD` to find a candidate minimizer of an objective over a box, and use the entries below to choose its schedule and initial state, read the observed points and estimate fault-tolerant resources. Import every name on this page from `nwqlib.algorithms.qhd`, for example `from nwqlib.algorithms.qhd import QHD, circuit_resources`.

The [QHD guide](../../algorithms/qhd.md) explains the model, the readout and the measured evidence, and its [source map](../../algorithms/qhd.md#source-map) gives the paper, version and equation behind each step. <a id="constrained-problems"></a><a id="box-refinement"></a>The augmented-Lagrangian layer for constrained problems and box refinement are on [QHD constrained problems and box refinement](qhd_constrained.md). The `QHD` entry below has a complete example.

| Task | Entry |
| --- | --- |
| Configure and run QHD on an `Optimization` | [`QHD`][nwqlib.algorithms.qhd.method.QHD] with `solve` or `plan` |
| Choose the binary circuit's synthesis | [`BinarySynthesis`][nwqlib.algorithms.qhd.binary.BinarySynthesis] |
| Choose the schedule a(t), b(t) | [`QuadraticSchedule`][nwqlib.algorithms.qhd.schedules.QuadraticSchedule], [`CubicSchedule`][nwqlib.algorithms.qhd.schedules.CubicSchedule], [`ShiftedCubicSchedule`][nwqlib.algorithms.qhd.schedules.ShiftedCubicSchedule] |
| Choose the initial state | [`KineticGroundState`][nwqlib.algorithms.qhd.initial_state.KineticGroundState], [`UniformState`][nwqlib.algorithms.qhd.initial_state.UniformState], [`GaussianState`][nwqlib.algorithms.qhd.initial_state.GaussianState] |
| Read the best observed and most probable points | [`QHDAnalysis`][nwqlib.algorithms.qhd.records.QHDAnalysis] |
| Compare the result with a grid minimum or a reference evolution | [`QHDVerification`][nwqlib.algorithms.qhd.records.QHDVerification] with `result.verify(checks=...)` |
| Count rotations and T gates and bound the product-formula error | [`circuit_resources`][nwqlib.algorithms.qhd.resources.circuit_resources], [`evolution_bound`][nwqlib.algorithms.qhd.evolution_bounds.evolution_bound], [`run_resources`][nwqlib.algorithms.qhd.resources.run_resources] |

## Configure QHD

::: nwqlib.algorithms.qhd.method.QHD
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qhd.binary.BinarySynthesis
    options:
      heading_level: 3

## Schedules

The guide's [schedules](../../algorithms/qhd.md#schedules-and-coefficient-rule) section compares the three schedules. `QHD.coefficient_rule` selects whether a step uses the point values at its midpoint or the step averages of the interval integrals. [Mathematics](../../mathematics.md#qhd-schedule-arithmetic) derives the point values, step integrals and derivative bounds that planning and `evolution_bound` use, with their rounding.

::: nwqlib.algorithms.qhd.schedules.QuadraticSchedule
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qhd.schedules.CubicSchedule
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qhd.schedules.ShiftedCubicSchedule
    options:
      heading_level: 3
      members: false

## Initial states

Each initial state is a product of nonnegative per-variable amplitude vectors, and `QHD.initial_state_preparation` selects how quantum execution prepares it. The guide's [initial states](../../algorithms/qhd.md#initial-states-and-preparation) section compares them. [Mathematics](../../mathematics.md#qhd-initial-state-vectors) gives the construction error of each vector.

::: nwqlib.algorithms.qhd.initial_state.KineticGroundState
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qhd.initial_state.UniformState
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qhd.initial_state.GaussianState
    options:
      heading_level: 3
      members: false

## Read the result

::: nwqlib.algorithms.qhd.records.QHDAnalysis
    options:
      heading_level: 3
      members: [candidate, objective, valid_probability, position_mean, position_standard_deviation]

## Check a result

::: nwqlib.algorithms.qhd.records.QHDVerification
    options:
      heading_level: 3
      members: false

## Estimate fault-tolerant resources {#fault-tolerant-resources}

`circuit_resources(plan, synthesis_epsilon=E)` reads a quantum QHD `Plan` and builds no circuit. The guide's [fault-tolerant resources](../../algorithms/qhd.md#fault-tolerant-resources) section states what the rotation count includes, how the synthesis budget is split and which error sources are bounded.

::: nwqlib.algorithms.qhd.resources.circuit_resources
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.evolution_bounds.evolution_bound
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.resources.run_resources
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.resources.QHDCircuitResources
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.resources.QHDSynthesisProjection
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.resources.QHDErrorSource
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.evolution_bounds.QHDEvolutionBound
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.resources.QHDRunResources
    options:
      heading_level: 3

::: nwqlib.algorithms.qhd.resources.QHDRunEntry
    options:
      heading_level: 3

## Limits

- QHD is not a global optimizer. The candidate and the most probable point are points of a finite grid of the box, and the candidate is the least evaluated objective only among the observed points with positive weight.
- `max_work` and `max_bytes` limit the work and bytes that planning counts. They do not bound the process memory or undocumented SymPy, SciPy or Qiskit costs.
- The error sources of `circuit_resources` compare the circuit with its finite model. None concerns grid discretization or the optimization gap, and the record forms no total.
- The T estimate is the leading term of the typical Ross-Selinger count, without an error bar, and a requested synthesis budget is not an achieved error.
- [Limitations and open work](../../ROADMAP.md#qhd-core-models) lists the open work on the QHD models.

## Entries on other pages

The augmented-Lagrangian layer and box refinement are documented on [QHD constrained problems and box refinement](qhd_constrained.md):

- <a id="nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult"></a>[`ConstrainedQHDResult`][nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult]
- <a id="nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult.multipliers"></a>[`ConstrainedQHDResult.multipliers`][nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult.multipliers]
- <a id="nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult.report"></a>[`ConstrainedQHDResult.report`][nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult.report]
- <a id="nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult.save"></a>[`ConstrainedQHDResult.save`][nwqlib.algorithms.qhd.constrained.ConstrainedQHDResult.save]
- <a id="nwqlib.algorithms.qhd.constrained.constrained_grid_minimum"></a>[`constrained_grid_minimum`][nwqlib.algorithms.qhd.constrained.constrained_grid_minimum]
- <a id="nwqlib.algorithms.qhd.constrained.load_augmented_lagrangian"></a>[`load_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.load_augmented_lagrangian]
- <a id="nwqlib.algorithms.qhd.constrained.plan_augmented_lagrangian"></a>[`plan_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.plan_augmented_lagrangian]
- <a id="nwqlib.algorithms.qhd.constrained.resume_augmented_lagrangian"></a>[`resume_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.resume_augmented_lagrangian]
- <a id="nwqlib.algorithms.qhd.constrained.solve_augmented_lagrangian"></a>[`solve_augmented_lagrangian`][nwqlib.algorithms.qhd.constrained.solve_augmented_lagrangian]
- <a id="nwqlib.algorithms.qhd.constrained_records.ALEvaluation"></a>[`ALEvaluation`][nwqlib.algorithms.qhd.constrained_records.ALEvaluation]
- <a id="nwqlib.algorithms.qhd.constrained_records.ALIteration"></a>[`ALIteration`][nwqlib.algorithms.qhd.constrained_records.ALIteration]
- <a id="nwqlib.algorithms.qhd.constrained_records.ALResources"></a>[`ALResources`][nwqlib.algorithms.qhd.constrained_records.ALResources]
- <a id="nwqlib.algorithms.qhd.constrained_records.AbsorbedBound"></a>[`AbsorbedBound`][nwqlib.algorithms.qhd.constrained_records.AbsorbedBound]
- <a id="nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian"></a>[`AugmentedLagrangian`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian]
- <a id="nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangianRecord"></a>[`AugmentedLagrangianRecord`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangianRecord]
- <a id="nwqlib.algorithms.qhd.constrained_records.ConstrainedGridMinimum"></a>[`ConstrainedGridMinimum`][nwqlib.algorithms.qhd.constrained_records.ConstrainedGridMinimum]
- <a id="nwqlib.algorithms.qhd.constrained_records.ConstraintPreprocessing"></a>[`ConstraintPreprocessing`][nwqlib.algorithms.qhd.constrained_records.ConstraintPreprocessing]
- <a id="nwqlib.algorithms.qhd.constrained_records.FormTrial"></a>[`FormTrial`][nwqlib.algorithms.qhd.constrained_records.FormTrial]
- <a id="nwqlib.algorithms.qhd.constrained_records.InnerRepresentation"></a>[`InnerRepresentation`][nwqlib.algorithms.qhd.constrained_records.InnerRepresentation]
- <a id="nwqlib.algorithms.qhd.constrained_records.MultiplierBounds"></a>[`MultiplierBounds`][nwqlib.algorithms.qhd.constrained_records.MultiplierBounds]
- <a id="nwqlib.algorithms.qhd.constrained_records.SlackAxis"></a>[`SlackAxis`][nwqlib.algorithms.qhd.constrained_records.SlackAxis]
- <a id="nwqlib.algorithms.qhd.refinement.load_box_refinement"></a>[`load_box_refinement`][nwqlib.algorithms.qhd.refinement.load_box_refinement]
- <a id="nwqlib.algorithms.qhd.refinement.refine_box"></a>[`refine_box`][nwqlib.algorithms.qhd.refinement.refine_box]
- <a id="nwqlib.algorithms.qhd.refinement.resume_box_refinement"></a>[`resume_box_refinement`][nwqlib.algorithms.qhd.refinement.resume_box_refinement]
- <a id="nwqlib.algorithms.qhd.refinement_records.BoxRefinement"></a>[`BoxRefinement`][nwqlib.algorithms.qhd.refinement_records.BoxRefinement]
- <a id="nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult"></a>[`BoxRefinementResult`][nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult]
- <a id="nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult.report"></a>[`BoxRefinementResult.report`][nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult.report]
- <a id="nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult.save"></a>[`BoxRefinementResult.save`][nwqlib.algorithms.qhd.refinement_records.BoxRefinementResult.save]
- <a id="nwqlib.algorithms.qhd.refinement_records.RefinementLevel"></a>[`RefinementLevel`][nwqlib.algorithms.qhd.refinement_records.RefinementLevel]
- <a id="nwqlib.algorithms.qhd.refinement_records.RefinementResources"></a>[`RefinementResources`][nwqlib.algorithms.qhd.refinement_records.RefinementResources]
- <a id="nwqlib.algorithms.qhd.refinement_records.StallSplit"></a>[`StallSplit`][nwqlib.algorithms.qhd.refinement_records.StallSplit]
