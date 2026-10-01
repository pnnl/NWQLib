# How NWQLib works

Every NWQLib calculation passes through five objects. You build the Problem and the Method, and NWQLib creates the other three.

| Object | What it is | Example |
| --- | --- | --- |
| [Problem](glossary.md#problem) | The scientific question with its input data and requested output, independent of any algorithm | `LinearSystem(A=A, b=b)` holds A and b |
| [Method](glossary.md#method) | A configured algorithm with its options. It plans, executes and analyzes one Problem | `LCHS()`, `QLS(epsilon_inv=0.01)` |
| [Plan](glossary.md#plan) | The choices a Method makes once for one Problem: the construction, its parameters, the random-stream snapshot and the data needed to reconstruct the output. It exists before any circuit is built | `plan(problem, method=LCHS())` |
| [Run](glossary.md#run) | One execution of a Plan. It records the work it reserves and submits and the observations that come back, and it can be saved and reopened | `prepared.run` after `submit(prepared)` |
| [Result](glossary.md#result) | The analyzed scientific output, with its Plan and the observations (`RunData`) it was computed from. Reanalysis produces a new Result from the same observations | `result.solution`, `result.eigenvalue` |

[Choose a problem and output](problems.md) lists the Problem types and their outputs, and each algorithm guide describes its Method's options.

## How the objects connect

```text
Problem + Method --plan--> Plan
Plan --estimate--> resource counts, before any circuit is built
Plan --solve--> Result
Plan --prepare--> Prepared --submit--> Run --wait--> Result
Result --save--> folder --load_result--> Result
```

| Step | Call | What it does |
| --- | --- | --- |
| Plan | `plan(problem, method=...)` | The Method chooses its construction and computes the numerical data it needs. No measurement is taken |
| Compare | `compare(problem, methods=(...))` | Plans and estimates each Method for the same Problem without measuring anything. Every row stays visible, and the comparison chooses no winner |
| Estimate | `estimate(plan)` | Adds up the resource formulas of the Plan's construction, such as qubits, CX gates and shots, and labels each count as exact, an upper bound, an estimate or unavailable |
| Solve | `solve(problem, method=...)` or `solve(plan)` | Plans when given a Problem, executes, and returns the Result |
| Prepare and submit | `prepare(plan, backend=...)`, `submit(prepared)`, `run.wait()` | Builds the circuits, submits them in a Run and waits for the Result. Use these calls for a provider backend or to control execution yourself |
| Save and load | `result.save(path)`, `load_result(path)` | Saves a completed Result and reopens it for inspection and for reanalysis of its saved observations |
| Continue | `load_run(path, backend=...)`, `run.resume()` | Reopens a saved Run and collects its pending work |

[Plan, compare and solve](scientist.md) walks through these calls for two eigenvalue methods. [Run on a backend](prepared_execution.md) covers prepare and submit, [Save, load and reanalyze results](saved_evidence.md) covers saved Results, and [Continue an interrupted run](run_archives.md) covers saved Runs. The [API reference](api/index.md) gives every signature.
