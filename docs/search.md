# Search selected Plans

`scan` ranks a finite set of already selected Plans. A different method setting or shot count requires a new Plan. Planning and execution remain explicit.

```python
from nwqlib import Expectation, plan, scan, solve
from nwqlib.algorithms import ExpectationMethod
from nwqlib.search import Candidate, Objective

problem = Expectation(state=[1, 0], observable=[[1, 0], [0, -1]])
plans = tuple(plan(problem, method=ExpectationMethod(), shots=n, seed=7)
              for n in (8, 32))
search = scan(tuple(Candidate(p) for p in plans),
              objectives=(Objective(kind="requested_shots"),))
print(search.selection.nondominated)  # (0,)
selected = search.select(0)
result = solve(selected)             # Executes the original 8-shot Plan.
```

A `Candidate(plan, *, allocation=None, facts=(), reference=None, prior_work=())` owns no replacement Problem, output or acquisition settings. `facts` are existing `FramedFact` values, and `reference` is an existing `TargetReference`. Supplied prior-work Facts keep their occurrence multiplicity, scope and provenance. An empty tuple means no prior work was recorded, rather than an observed zero cost.

Without a profile, scan uses the selected construction's logical laws. A profile supplies explicit model data and requires an explicit Allocation on every candidate. Allocation records grants/placement for assessment and provenance; it does not configure backend ranks, threads, devices or scheduler requests. Backend execution uses the backend explicitly passed to `prepare` or `solve`. In particular, two 8 GiB locations cannot satisfy a declared 10 GiB requirement at one location.

The supported minimized objectives are:

- `requested_shots`: selected MeasurementBatch shots plus separately declared direct Experiment readouts. This is the planned population; retries and later unspecified adaptive acquisitions are not silently included or treated as zero.
- `logical_width`: the largest concrete per-location logical peak. A serial-acquisition peak requires `ResourceContext(batch_schedule="serial")`; its condition remains visible.
- `predicted_seconds`: one explicit `model_id` and timing `scope`, summed only under an explicit serial schedule. Every selected acquisition needs a matching available original prediction. Overlapping scopes are not added together. The number remains conditional on its model; summing estimates does not create a confidence guarantee.

Nondominance means no worse on every listed objective and strictly better on at least one, within this supplied set. Equal rows all remain. Unavailable values keep their reasons in `selection.incomparable`; they are not ranked as zero. The comparison also keeps all five assessment axes when available, including infeasible rows. A frontier is neither a feasibility filter nor a global optimum. `select(index)` returns any original row, including a deliberately chosen dominated row. Execution rejects a blocked row before preparation.

Passing a selected row to `prepare` or `solve` keeps its original PlanEstimate and Allocation on the Run and Result. The association is checked against the actual acquisition before submission. It does not replan, change shots or evaluate the model again. Keeping the original Plan is what lets the Result, its observations and the forecast name one selection by `plan_id`. A repeated selection could attach data to a construction it was not acquired from. A plain WorkloadEstimate has no physical forecasts and is not converted into one. Passing the Plan alone remains a direct execution of the same selected science.

Rescore an existing comparison with `scan(search.comparison, objectives=...)`. This reads its original quantities and forecasts; replacement profile, context or assessment time is rejected. No planner, resource fold, model evaluation, backend call or reference computation occurs. Blocked rows keep their method, reason and Allocation without a fabricated Plan. `SearchSelection.model_validate_json(...)` checks the portable scalar selection, and `validate_comparison(comparison)` checks its association. These records do not restore missing native data.

Controls bound the work that this operation actually performs: `max_candidates=256`, `max_values=4096` candidate/objective output cells, `max_assessments=4096` assessment/prediction records, `max_pair_comparisons=1000000`, and `max_integer_bits=4096` for exact rational arithmetic. The complete new forecast expansion is checked before model evaluation. Stored forecast traversal is also bounded, sharing identical objects without copying them. The quadratic frontier admits at most `N*(N-1)*k` coordinate comparisons before running. Counts must remain integer Rationals; conditional predicted seconds may be fractional. These are finite operation controls, not CPU-time or process-memory promises.

Call-local caches reuse identical original forecasts and construction folds. SearchWork records actual evaluation and reuse separately from supplied prior work and prospective execution estimates. It does not infer factorization, synthesis or acquisition costs that were never recorded.

## Code owners

Ranking has no paper source. The dominance relation is the definition stated above.

| Step | Contract | Code owner |
| --- | --- | --- |
| Strict dominance, frontier and incomparable rows | Nondominance paragraph above | `search.SearchSelection` |
| Objective values read from stored Plans, folds and forecasts | List of supported objectives above | `search._objective_value` |
| Association of a selection with its Comparison, and rescoring without new work | Rescoring paragraph above | `search._comparison_identity`, `search.SearchSelection.validate_comparison` |
| Size controls checked before evaluation | ENGINEERING_CONSTANTS.md, "Finite search frontier envelope" | `search.scan`, `search._frontier_size` |
