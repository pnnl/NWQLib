# Solve, plan and read results {#shared-workflow-api}

These calls turn a Problem and a configured Method into a Result. `solve` does it in one step, and `plan`, `prepare` and `submit` do it step by step, with `compare`, `estimate` and `scan` to choose before running. The [Plan, compare and solve](../scientist.md) guide shows them in context, and [Choose a problem and output](../problems.md) explains the Problems and their outputs.

```python
from nwqlib import (
    Accuracy, ConstrainedOptimization, Eigenproblem, Eigenvalue, Expectation,
    LinearDynamics, LinearSystem, Optimization, SpectralEstimation,
    compare, estimate, load_result, load_run, methods, plan, prepare, scan,
    solve, submit,
)
from nwqlib.core import Record
from nwqlib.execution import ExecutionLimits, Run, RunFailed
from nwqlib.search import Candidate, Objective
```

The outputs `Eigenphase`, `NormSquared`, `NormalizedExpectation`, `OptimizationCandidate`, `QuadraticForm`, `Samples`, `Solution` and `StateVector` also import from `nwqlib`. The eigenvalues of `[[2, 1], [1, 2]]` are 1 and 3, and Lanczos with a two-dimensional Krylov space finds the smaller one:

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import Lanczos

problem = Eigenproblem(A=[[2.0, 1.0], [1.0, 2.0]])
method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
result = solve(problem, method=method, seed=7)
print(round(result.eigenvalue, 10))  # 1.0
```

| Task | Entries |
| --- | --- |
| Solve a problem in one call | [`solve`](#nwqlib.scientist.solve) |
| Build, copy and identify records | [`Record`](#nwqlib.core.records.Record) |
| State the question | [Problems and outputs](#scientific-inputs-and-output-quantities), [`Accuracy`](#nwqlib.problems.records.Accuracy) |
| Read the answer, check it and save it | [`Result`](#nwqlib.core.analysis.Result) and its answer table |
| Plan, compare or estimate before running | [`plan`](#nwqlib.scientist.plan), [`compare`](#nwqlib.scientist.compare), [`estimate`](#nwqlib.scientist.estimate) |
| Rank candidate Plans | [`scan`](#nwqlib.search.scan) |
| Run step by step, with limits | [`prepare`](#nwqlib.scientist.prepare), [`submit`](#nwqlib.scientist.submit), [`Run`](#nwqlib._prepared_execution.Run), [`ExecutionLimits`](#nwqlib.execution.ExecutionLimits) |
| Reopen a saved Result or Run | [`load_result`](#nwqlib.scientist.load_result), [`load_run`](#nwqlib.scientist.load_run) |

Method authors find the Method protocol, the hooks of Plan and Result and `PreparedHandle` in [Extending NWQLib](extending.md).

## Solve a problem

::: nwqlib.scientist.solve
    options:
      heading_level: 3

## Build, copy and identify records

Every Problem, output, Plan and Result is a `Record`.

::: nwqlib.core.records.Record
    options:
      heading_level: 3
      show_bases: false
      members:
        - revise

## Problems and outputs {#scientific-inputs-and-output-quantities}

::: nwqlib.problems.records
    options:
      heading_level: 3
      show_root_heading: false
      show_bases: false
      filters:
        - "!^_"
        - "!^(evidence_unit|evidence_scope|to_record|frame|metric|default_output)$"
      members:
        - ProblemRecord
        - Eigenproblem
        - LinearDynamics
        - LinearSystem
        - Expectation
        - SpectralEstimation
        - Optimization
        - ConstrainedOptimization
        - OutputRecord
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

## Read and check a Result {#scientific-results-and-analysis-data}

A Result's answer field depends on its Method, as the table in the `Result` entry shows. The [Save, load and reanalyze results](../saved_evidence.md) guide covers saved Results and reanalysis, and [Check accuracy and verify a result](../verification.md) covers `assess` and `verify`.

::: nwqlib.core.analysis.Result
    options:
      heading_level: 3
      members:
        - plan
        - data
        - analyze
        - assess
        - verify
        - report
        - save

::: nwqlib.core.analysis.RunData
    options:
      heading_level: 3
      members:
        - artifact

::: nwqlib.core.analysis.AnalysisOrigin
    options:
      heading_level: 3
      members: false

::: nwqlib.artifacts.ArtifactManifest
    options:
      heading_level: 3
      members: false

::: nwqlib.artifacts.ArtifactHandle
    options:
      heading_level: 3
      members:
        - array
        - available

::: nwqlib.reporting.records.ReportSection
    options:
      heading_level: 3
      show_bases: false
      members: false

## Plan, compare and estimate before running {#planning-execution-and-reports}

`plan` returns a `Plan` without running anything, `compare` plans several Methods and estimates each Plan, and `estimate` counts the resources of one Plan. `estimate` adds up counting formulas, and `Prepared.inspect_resources` counts the operations of the circuits that preparation built.

::: nwqlib.scientist.plan
    options:
      heading_level: 3

::: nwqlib.core.planning.Plan
    options:
      heading_level: 3
      members: false

::: nwqlib.scientist.compare
    options:
      heading_level: 3

::: nwqlib.scientist.Comparison
    options:
      heading_level: 3
      members:
        - select

::: nwqlib.scientist.ComparisonRow
    options:
      heading_level: 3
      members: false

::: nwqlib.scientist.estimate
    options:
      heading_level: 3

::: nwqlib.scientist.methods
    options:
      heading_level: 3

## Rank candidate plans {#finite-search}

::: nwqlib.search.scan
    options:
      heading_level: 3

::: nwqlib.search.Candidate
    options:
      heading_level: 3
      members: false

::: nwqlib.search.Objective
    options:
      heading_level: 3
      members: false

::: nwqlib.search.SearchResult
    options:
      heading_level: 3
      members:
        - select

::: nwqlib.search.SearchSelection
    options:
      heading_level: 3
      members:
        - validate_comparison

::: nwqlib.search.ObjectiveValue
    options:
      heading_level: 3
      members: false

::: nwqlib.search.SearchWork
    options:
      heading_level: 3
      members: false

## Run step by step {#actual-execution-state}

`prepare` builds the circuits and returns them with their open `Run`, `submit` starts the Run, and `Run.wait` returns the Result. The [Run on a backend](../prepared_execution.md) guide describes progress, pending jobs, cancellation and limits.

::: nwqlib.scientist.prepare
    options:
      heading_level: 3

::: nwqlib._prepared_execution.Prepared
    options:
      heading_level: 3
      members:
        - circuits
        - circuit
        - setting_names
        - inspect_resources

::: nwqlib.scientist.submit
    options:
      heading_level: 3

::: nwqlib._prepared_execution.Run
    options:
      heading_level: 3
      members:
        - wait
        - resume
        - result
        - cancel
        - close
        - save
        - directory
        - limits
        - extend_limits
        - limit_amendments
        - trace
        - data
        - observations
        - exposure
        - warnings
        - hydrate
        - artifacts
        - release_native

::: nwqlib.execution.ExecutionLimits
    options:
      heading_level: 3
      members: false

::: nwqlib.execution.LimitAmendment
    options:
      heading_level: 3
      members: false

::: nwqlib.execution.ExecutionTrace
    options:
      heading_level: 3
      members:
        - jobs

::: nwqlib.execution.RunFailed
    options:
      heading_level: 3
      members: false

`submit` and `Run.resume` call the two functions below for a backend that runs jobs outside this process. The backend guides name them where a direct call helps.

::: nwqlib._prepared_execution.submit_detached
    options:
      heading_level: 3

::: nwqlib._prepared_execution.refresh_submissions
    options:
      heading_level: 3

## Save and reopen

`result.save(path)` writes a Result with its Plan and data, and `load_result` reopens it. A Run with a folder saves its state as it runs, `run.save(path)` copies it, and `load_run` reopens either to continue the same work. The [Continue an interrupted run](../run_archives.md) guide compares the two.

::: nwqlib.scientist.load_result
    options:
      heading_level: 3

::: nwqlib.scientist.load_run
    options:
      heading_level: 3

A Method family that runs a sequence of Plans documents it with the family. QHD's augmented-Lagrangian layer and box refinement run one ordinary QHD Plan per round or level, each planned with its own random streams, as `compare` plans each Method, and run with `prepare` and `submit`. With `directory=...` these Runs are saved as they run, and `resume_augmented_lagrangian` and `resume_box_refinement` continue an interrupted run, reopening an unfinished Run with `load_run` and the `progress` of the resume call ([QHD constrained problems and box refinement](algorithms/qhd_constrained.md)).

## Entries on other pages

The `Method` base class and its hooks, `PreparedHandle`, and the Plan, Result and Problem hooks are documented on [Extending NWQLib](extending.md):

- <a id="nwqlib._prepared_execution.PreparedHandle"></a>[`PreparedHandle`][nwqlib._prepared_execution.PreparedHandle]
- <a id="nwqlib._prepared_execution.PreparedHandle.inspect_circuit"></a>[`PreparedHandle.inspect_circuit`][nwqlib._prepared_execution.PreparedHandle.inspect_circuit]
- <a id="nwqlib._prepared_execution.PreparedHandle.inspect_resources"></a>[`PreparedHandle.inspect_resources`][nwqlib._prepared_execution.PreparedHandle.inspect_resources]
- <a id="nwqlib.algorithms.protocol.Method"></a>[`Method`][nwqlib.algorithms.protocol.Method]
- <a id="nwqlib.algorithms.protocol.Method.analyze"></a>[`Method.analyze`][nwqlib.algorithms.protocol.Method.analyze]
- <a id="nwqlib.algorithms.protocol.Method.error_model"></a>[`Method.error_model`][nwqlib.algorithms.protocol.Method.error_model]
- <a id="nwqlib.algorithms.protocol.Method.execute"></a>[`Method.execute`][nwqlib.algorithms.protocol.Method.execute]
- <a id="nwqlib.algorithms.protocol.Method.plan"></a>[`Method.plan`][nwqlib.algorithms.protocol.Method.plan]
- <a id="nwqlib.algorithms.protocol.Method.prepare"></a>[`Method.prepare`][nwqlib.algorithms.protocol.Method.prepare]
- <a id="nwqlib.algorithms.protocol.Method.prepare_all_refusal"></a>[`Method.prepare_all_refusal`][nwqlib.algorithms.protocol.Method.prepare_all_refusal]
- <a id="nwqlib.algorithms.protocol.Method.recover_analysis"></a>[`Method.recover_analysis`][nwqlib.algorithms.protocol.Method.recover_analysis]
- <a id="nwqlib.algorithms.protocol.Method.verify"></a>[`Method.verify`][nwqlib.algorithms.protocol.Method.verify]
- <a id="nwqlib.core.planning.Plan.resolve"></a>[`Plan.resolve`][nwqlib.core.planning.Plan.resolve]
- <a id="nwqlib.core.analysis.Result.validate_plan"></a>[`Result.validate_plan`][nwqlib.core.analysis.Result.validate_plan]
- <a id="nwqlib.problems.records.ProblemRecord.default_output"></a>[`ProblemRecord.default_output`][nwqlib.problems.records.ProblemRecord.default_output]

`PhysicalScale` is documented on [Inputs and input types](inputs.md):

- <a id="nwqlib.problems.inputs.PhysicalScale"></a>[`PhysicalScale`][nwqlib.problems.inputs.PhysicalScale]
- <a id="nwqlib.problems.inputs.PhysicalScale.as_float"></a>[`PhysicalScale.as_float`][nwqlib.problems.inputs.PhysicalScale.as_float]
- <a id="nwqlib.problems.inputs.PhysicalScale.squared_as_float"></a>[`PhysicalScale.squared_as_float`][nwqlib.problems.inputs.PhysicalScale.squared_as_float]
