# Rank candidate plans

<a id="search-selected-plans"></a>`scan` ranks a finite set of existing Plans by objectives such as requested shots, and returns the nondominated ones. Each candidate is a Plan you have already made, so a different Method setting or shot count needs a new Plan. You plan and execute the candidates yourself.

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

`Candidate(plan, *, allocation=None, facts=(), reference=None, prior_work=())` wraps an existing Plan and cannot change its problem, output or shots. `facts` are existing `FramedFact` values, the error evidence you have for the candidate, and `reference` is an existing `TargetReference`. Supplied `prior_work` entries are kept as given, with their repeats, scope and source. An empty tuple means that no prior work was recorded, not that the prior work cost zero.

Without a device profile, `scan` uses the logical resource formulas of each Plan. A profile supplies explicit model data and requires an explicit Allocation on every candidate. An Allocation records the granted devices and placement, for the assessment and as a record of the grant. It does not configure backend ranks, threads, devices or scheduler requests, and execution uses the backend you pass to `prepare` or `solve`. Memory on separate devices is not pooled ([Check device fit and run time](profiles.md#will-it-fit-in-memory)).

## Objectives

Each objective is minimized:

- `requested_shots`: the shots of the Plan's measurement batches plus the separately declared direct readouts of its experiments. This is the planned number of shots. Retries and later adaptive measurements that the Plan does not specify are outside this count, which does not treat them as zero.
- `logical_width`: the largest exact peak of logical qubits at one location, for one measurement run. A peak that holds only when the measurement runs execute one after another requires `ResourceContext(batch_schedule="serial")`, and the value keeps that condition.
- `predicted_seconds`: the predicted times of one explicit `model_id` and timing `scope`, summed only under an explicit serial schedule. Every measurement run of the Plan needs a matching prediction. Overlapping scopes are not added together. The sum is conditional on its model and carries no confidence guarantee.

## Read the ranking

A row whose values are all available is nondominated when no other such row is at least as good on every listed objective and strictly better on at least one, within this supplied set. Equal rows all remain. Rows with an unavailable value keep their reasons in `selection.incomparable` and are not ranked as zero. With a profile, the comparison also keeps all five assessment fields when available ([applicability, capability, capacity, time and accuracy](profiles.md#independent-axes-and-original-associations)), including for infeasible rows. The nondominated set (the Pareto front of this set) is neither a feasibility filter nor a global optimum. `select(index)` returns any original row, including a dominated row that you choose deliberately. Execution rejects a blocked row before preparation.

## Run a selected row

Passing a selected row to `prepare` or `solve` keeps its original `PlanEstimate` and Allocation on the Run and the Result. Before submission, NWQLib checks that they belong to the circuits being run. It does not plan again, change shots or evaluate the model again. Because the original Plan is kept, the Result, its observations and the forecast all name one Plan by `plan_id`. Planning again could attach data to a construction the data did not come from. A plain `WorkloadEstimate` has no physical forecasts and is not converted into one. Passing only the Plan runs the same Plan directly.

## Rescore a comparison

Rescore an existing comparison with `scan(search.comparison, objectives=...)`. It reads the stored quantities and forecasts and rejects a replacement profile, context or assessment time. It runs no planning, resource summation, model evaluation, backend call or reference computation. Blocked rows keep their Method, reason and Allocation, with no Plan made up for them. `SearchSelection.model_validate_json(...)` checks a saved selection, recomputing its nondominated and incomparable rows from the stored values, and `validate_comparison(comparison)` checks that the selection belongs to that comparison. Neither restores missing numerical data or circuits.

## Limits

These limits bound the operations of one `scan` call, not CPU time or process memory ([engineering constants](ENGINEERING_CONSTANTS.md#finite-search-frontier-envelope)):

| Control | Default | What it bounds |
| --- | --- | --- |
| `max_candidates` | 256 | Candidates |
| `max_values` | 4096 | Candidate-objective output cells |
| `max_assessments` | 4096 | Assessment and prediction records |
| `max_pair_comparisons` | 1000000 | Coordinate comparisons for the nondominated set |
| `max_integer_bits` | 4096 | Integer size in exact rational arithmetic |

The complete set of new forecasts is checked before any model is evaluated. Reading stored forecasts is bounded too, and identical objects are shared instead of copied. For N rows and k objectives, the nondominated set needs at most `N*(N-1)*k` coordinate comparisons, which are checked against the limit before it is computed. Counts must be integer rationals, and conditional predicted seconds may be fractional.

Within one call, identical forecasts and resource totals are computed once and reused. `SearchWork` records the evaluations and reuses of the scan separately from supplied prior work and from execution estimates. It does not infer factorization, synthesis or measurement costs that were never recorded.

## Source and code {#code-owners}

Ranking has no paper source. Nondominance is the definition given above. The [code tour](CODE_TOUR.md#code-locations-for-inputs-and-search) lists the code for each step.
