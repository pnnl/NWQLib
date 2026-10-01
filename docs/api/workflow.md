# Shared workflow API

Use scientific inputs with a configured Method. The selected Plan, Prepared handle, Run and Result are the actual owners described in the [workflow guide](../scientist.md), [prepared execution guide](../prepared_execution.md) and [run archives](../run_archives.md).

## Scientific inputs and output quantities

The Problem supplies the original mathematical inputs and units. Method initialization belongs to the configured Method. An output names the requested quantity, and an optional Accuracy specifies a criterion without asserting that it has been met.

::: nwqlib.problems.records
    options:
      members:
        - ProblemRecord
        - OutputRecord
        - Eigenproblem
        - LinearDynamics
        - LinearSystem
        - Expectation
        - SpectralEstimation
        - Optimization
        - ConstrainedOptimization
        - Eigenvalue
        - Eigenphase
        - Solution
        - StateVector
        - NormSquared
        - NormalizedExpectation
        - QuadraticForm
        - Samples
        - OptimizationCandidate
        - Accuracy

## Planning, execution and reports

::: nwqlib.scientist.plan

::: nwqlib.scientist.compare

::: nwqlib.scientist.estimate

::: nwqlib.scientist.prepare

::: nwqlib.scientist.submit

::: nwqlib.scientist.solve

::: nwqlib.scientist.load_result

::: nwqlib.scientist.load_run

`load_run(path, backend=..., progress=...)` gives the reopened Run a progress callback with the meaning it has for `prepare`. A callable receives the stages that the continued Run reports, False disables the reports, and None selects the notebook display.

::: nwqlib.scientist.methods

::: nwqlib.scientist.Comparison

::: nwqlib.scientist.ComparisonRow

::: nwqlib.algorithms.protocol.Method

::: nwqlib.core.planning.Plan

A family layer that runs a sequence of Plans is documented with its family. The QHD augmented-Lagrangian layer ([`solve_augmented_lagrangian`, `load_augmented_lagrangian` and `constrained_grid_minimum`](algorithms/qhd.md#constrained-problems)) and [box refinement (`refine_box` and `load_box_refinement`)](algorithms/qhd.md#box-refinement) run one ordinary QHD Plan per round or level. They plan it with its own random streams, as `compare` plans each candidate, and run it with `prepare` and `submit`. With `directory=...` these Runs are durable, and `resume_augmented_lagrangian` and `resume_box_refinement` continue an interrupted run, reopening an unfinished Run with `load_run` and the `progress` of the resume call.

## Scientific results and analysis data

`Result.analyze` reduces the attached data with explicit settings. `Result.assess` evaluates a stated criterion, and `Result.verify` invokes a selected check. Compact display and `Result.report` read existing values and metadata without acquisition or reanalysis. See [result reporting](../development/reports.md) for incomplete data, original environment, artifact inventory and forecast meaning.

::: nwqlib.core.analysis.Result

::: nwqlib.core.analysis.RunData

::: nwqlib.core.analysis.AnalysisOrigin

::: nwqlib.problems.inputs.PhysicalScale

::: nwqlib.artifacts.ArtifactManifest

::: nwqlib.artifacts.ArtifactHandle

## Finite search

::: nwqlib.search.Candidate

::: nwqlib.search.Objective

::: nwqlib.search.scan

::: nwqlib.search.SearchSelection

::: nwqlib.search.SearchResult

## Actual execution state

The common entry points return these existing objects. Native inspection reads the artifacts actually produced by preparation. A symbolic Plan does not supply an observed native inventory. `prepare(plan, settings="all")` prepares every static setting on a local backend without submitting, and `Prepared.setting_names` names the setting of each circuit in the index order of `Prepared.circuits`, `Prepared.circuit` and `Prepared.inspect_resources`.

::: nwqlib._prepared_execution.Prepared
    options:
      show_root_full_path: false

::: nwqlib._prepared_execution.Run
    options:
      show_root_full_path: false

::: nwqlib._prepared_execution.PreparedHandle
    options:
      show_root_full_path: false

::: nwqlib.execution.ExecutionLimits
    options:
      show_root_full_path: false

::: nwqlib.execution.LimitAmendment
    options:
      show_root_full_path: false

::: nwqlib.execution.ExecutionTrace
    options:
      show_root_full_path: false

::: nwqlib.execution.RunFailed
    options:
      show_root_full_path: false

The existing supported protected Method/Result hooks, including `_bind`, `_attach`, `_validate_common_plan` and `_summary_lines`, have explicit signatures and obligations in [the extension contract](../algorithm_protocol.md#supported-protected-extension-hooks). They preserve selected input/data association and do not create extra execution.
