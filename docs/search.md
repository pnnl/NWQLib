# Rank candidate plans

<a id="search-selected-plans"></a>`scan` ranks a finite set of existing Plans by objectives such as requested shots, and marks the nondominated ones (the Pareto front) among all the rows it returns. Each `Candidate` wraps a Plan you have already made, so a different Method setting or shot count needs a new Plan. You plan and run the Plans yourself.

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

## Candidates

`Candidate(plan, *, allocation=None, facts=(), reference=None, prior_work=())` wraps an existing Plan and cannot change its problem, output or shots. `facts` are existing `FramedFact` values, the error evidence you have for that Plan, and `reference` is an existing `TargetReference`, a reference value of the target that sets the scale of a relative accuracy criterion. `prior_work` holds records (`Fact`) of work done earlier for that Plan, such as the `work_fact` that the conversion of a double-factorized Hamiltonian returns. Supplied `prior_work` entries are kept as given, with their repeats, scope and source. An empty tuple means that no prior work was recorded, not that the prior work cost zero.

Without a device profile, `scan` uses the logical resource formulas of each Plan. A profile supplies device models and requires an Allocation on every `Candidate`. An Allocation records the granted devices and placement, for the assessment and as a record of the grant. It does not configure backend ranks, threads, devices or scheduler requests, and execution uses the backend you pass to `prepare` or `solve`. Memory on separate devices is not pooled ([Check device fit and run time](profiles.md#will-it-fit-in-memory)).

## Objectives

Each objective is minimized:

- `requested_shots`: the shots of the Plan's measurement batches plus the separately declared direct readouts of its experiments. This is the planned number of shots. Retries and later adaptive measurements that the Plan does not specify are outside this count, which does not treat them as zero.
- `logical_width`: the largest exact peak number of logical qubits at one location during one circuit execution. A peak that holds only when the circuit executions of a measurement batch run one after another requires `ResourceContext(batch_schedule="serial")`, and the value records that condition. A sampled Plan usually has such a peak, so without that context its width is unavailable.
- `predicted_seconds`: the predicted times of one timing model and scope, given as `Objective(kind="predicted_seconds", model_id=time_model.content_id, scope=...)`, summed only under a serial schedule declared with `context=ResourceContext(batch_schedule="serial")`. Every planned execution of the Plan needs a matching prediction. Overlapping scopes are not added together. The sum is conditional on its model and carries no confidence guarantee.

## Read the ranking

A row whose values are all available is nondominated when no other such row is at least as good on every listed objective and strictly better on at least one, within this supplied set. Equal rows all remain. `selection.incomparable` lists the indices of rows with an unavailable value. Their reasons are the `ObjectiveValue.reason` fields in `selection.values`, and unavailable values are not ranked as zero. With a profile, the comparison also keeps all five assessment fields when available ([applicability, capability, capacity, time and accuracy](profiles.md#independent-axes-and-original-associations)), including for infeasible rows. The nondominated set (the Pareto front of this set) is neither a feasibility filter nor a global optimum. `select(index)` returns any original row, including a dominated row that you choose deliberately. Execution rejects a row without a Plan before preparation.

## Run a selected row

Passing a selected row to `prepare` or `solve` keeps its original `PlanEstimate` and Allocation on the Run and the Result. Before submission, NWQLib checks that they belong to the circuits being run. It does not plan again, change shots or evaluate the model again. Because the original Plan is kept, the Result, its observations and the forecast all name one Plan by `plan_id`. Planning again could attach data to a construction the data did not come from. A plain `WorkloadEstimate` has no device forecasts and is not converted into one. Passing only the Plan runs the same Plan directly.

## Rescore a comparison

Rescore an existing comparison with `scan(search.comparison, objectives=...)`. It reads the resource quantities and forecasts already held in the comparison rows and rejects a new profile, context or assessment time. It runs no planning, resource summation, model evaluation, backend call or reference computation. Rows without a Plan keep their Method, reason and Allocation, with no Plan made up for them. `SearchSelection.model_validate_json(...)` checks a saved selection, recomputing its nondominated and incomparable rows from its objective values, and `validate_comparison(comparison)` checks that the selection belongs to that comparison. Neither restores missing numerical data or circuits.

## Limits

These limits bound the operations of one `scan` call, not CPU time or process memory ([engineering constants](ENGINEERING_CONSTANTS.md#finite-search-frontier-envelope)):

| Control | Default | What it bounds |
| --- | --- | --- |
| `max_candidates` | 256 | Rows ranked |
| `max_values` | 4096 | Row-objective values |
| `max_assessments` | 4096 | Assessment and prediction records |
| `max_pair_comparisons` | 1000000 | Coordinate comparisons for the nondominated set |
| `max_integer_bits` | 4096 | Integer size in exact rational arithmetic |

The complete set of new forecasts is checked before any model is evaluated. Reading the forecasts held in a Comparison is also counted against `max_assessments`, and identical objects are shared instead of copied. For N rows and k objectives, the nondominated set needs at most `N*(N-1)*k` coordinate comparisons, which are checked against the limit before it is computed. Counts must be integer rationals, and conditional predicted seconds may be fractional.

Within one call, identical forecasts and resource totals are computed once and reused. `SearchWork` records the evaluations and reuses of the scan separately from supplied prior work and from execution estimates. It does not infer factorization, synthesis or measurement costs that were never recorded.

## Source and code {#code-owners}

Ranking has no paper source. Nondominance is the definition given above. The [code tour](CODE_TOUR.md#code-locations-for-inputs-and-search) lists the code for each step.
