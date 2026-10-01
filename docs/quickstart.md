# Quickstart

Solve a small linear differential equation, read its physical solution, and inspect the selected work. The same Python workflow accepts other scientific problems and configured Methods.

## Install

From the source checkout, install the package and local Aer executor:

```bash
python -m pip install -e ".[aer]"
```

Python 3.12 or later is required. Method guides describe optional chemistry, tensor and provider dependencies. Maintainers reproduce the validated environment using the [maintenance instructions](MAINTENANCE.md#support-and-external-data-validation).

## Solve linear dynamics

This solves `du/dt = -A u`, starting from `u(0) = (1, 0)`, at time `0.1`:

```python
from nwqlib import LinearDynamics, solve
from nwqlib.algorithms import LCHS

problem = LinearDynamics(
    A=[[0.4, 0.15], [0.05, 0.25]],
    initial_state=[1.0, 0.0],
    time=0.1,
)
result = solve(problem, method=LCHS())
print(result.solution)
```

The result is the physical solution vector, including its scale and phase. It is not normalized to unit length. The default Aer circuit uses one system qubit and eight coefficient ancillas, with 204 physical branches and 256 address slots.

The independent two-by-two matrix exponential gives approximately `[0.96082565, -0.00484022]`. The default finite LCHS approximation has absolute L2 discrepancy about `.000821472` in this example. Its quadrature bound is about `.00431140` and its tail bound is `.005`. The two components each receive half of `LCHS.approximation_tolerance`. They do not bound total physical-output error, which also depends on the selected preparation, evolution and numerical approximations. The [LCHS guide](algorithms/lchs.md) describes these bounds and explicit verification.

## Change the approximation

An explicit finer construction uses more quadrature nodes. Evaluate its selected finite sum classically:

```python
refined = solve(problem, method=LCHS(approximation_tolerance=0.001), execution="classical")
print(refined.solution)
```

For this input, the finer construction selects 396 nodes and has absolute L2 discrepancy about `.0000532605`. Its dense quantum circuit would need 512 address slots and ten total qubits, exceeding the default 256-slot cap. The comparison above evaluates the finite sum classically. These two observations do not establish monotonic convergence for every input. Dense SELECT uses classically computed branch exponentials. Larger systems need a supported structured construction with its own limit check.

## Inspect or select work before running it

The Result keeps the original `Plan` (the selected construction and its costs, computed before any circuit exists). Reading it and estimating its selected construction do not execute another circuit:

```python
from nwqlib import estimate, plan

resources = estimate(result.plan)
print(resources.quantity("logical_width", location="logical_device"))
print(resources.quantity("operations"))  # May be unavailable for a selected native leaf.
selected = plan(problem, method=LCHS())
```

`selected` can later be passed to `solve`, or to `prepare` and `submit` for direct lifecycle control. Planning selects numerical data and can perform the computations needed by that Method. It takes no measurements. [Resource estimates](resources.md) distinguish symbolic construction laws from actual native circuit inspection and device predictions.

Each printed quantity gives its value and unit, evidence interpretation, basis and represented population. An unavailable cost includes its reason and is distinct from zero. The linear dynamics notebook (`examples/lchs_linear_dynamics_intro.ipynb`) applies LCHS to advection-diffusion. It separates product-formula error from quadrature error and compares kernels, quadratures, and exact and MPS state preparation by their errors and success probabilities, with compiled gate counts for the isolated state loaders.

To save the completed result, pass a path that does not exist yet. `save` creates that directory with the result's metadata and supporting payload files, and it raises `FileExistsError` when the path already exists, so a saved result is never overwritten:

```python
from nwqlib import load_result

result.save("linear-dynamics-result")
restored = load_result("linear-dynamics-result")
print(restored.solution)
```

## Next steps

- [Scientist workflow](scientist.md) compares Lanczos and FixedGCIM for the same Eigenproblem.
- [Examples](examples.md) describes the example notebooks and the work each performs by default.
- [Input access](inputs.md) covers dense, sparse and structured scientific inputs.
- [Prepared execution](prepared_execution.md) and [run archives](run_archives.md) cover pending work and continuation.
- [Saved evidence](saved_evidence.md) covers stored results and explicit reanalysis.
- [CLI](cli.md) provides method discovery, parameter schemas and read-only report inspection.
