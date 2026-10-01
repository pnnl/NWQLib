# NWQLib

NWQLib applies quantum algorithms to scientific inputs and returns the computed quantities with their methods, resources and evidence.

| Your scientific problem | Methods and first example |
| --- | --- |
| Linear dynamics, `du/dt = -A u` | [LCHS](algorithms/lchs.md), [quickstart](quickstart.md) |
| Eigenvalues or phases of an operator | [Lanczos](algorithms/lanczos.md), [GCiM / ADAPT](algorithms/gcim.md), [QPE](algorithms/qpe.md) |
| A linear system, `Ax=b` | [QLS](algorithms/qls.md) |
| Optimization over a box | [QHD](algorithms/qhd.md) |
| Optimization over a box with equality or inequality constraints | [QHD augmented Lagrangian](algorithms/qhd.md#constrained-problems) |
| The expectation of an observable | [Expectation](algorithms/expectation.md) |

From a source checkout with Python 3.12 or later, install local Aer execution:

```bash
python -m pip install -e ".[aer]"
```

This complete cell solves a two-coordinate linear differential equation:

```python
from nwqlib import LinearDynamics, solve
from nwqlib.algorithms import LCHS

problem = LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1., 0.], time=.1)
result = solve(problem, method=LCHS())
print(result)
```

The default nine-qubit Aer calculation returns the physical vector, including its scale and phase. Its finite approximation does not certify total error. The [quickstart](quickstart.md) explains the result and an explicit finer construction. Replace A, the initial state and time to use your own input.

## Continue your workflow

| What you want to do | Where to start |
| --- | --- |
| Choose an algorithm | [Examples and method guides](examples.md) |
| Estimate what an algorithm would cost at 80 to 100 qubits | [Resource estimation notebook](examples.md#resource-estimation-at-scale) |
| Compare NWQLib with other quantum packages | [Why NWQLib](why_nwqlib.md) |
| Compare methods for the same scientific problem | [Scientist workflow](scientist.md) |
| Supply an operator or state | [Input access](inputs.md) |
| Estimate work or assess a device | [Resources](resources.md), [profiles](profiles.md) |
| Execute or continue a selected workflow | [Prepared execution](prepared_execution.md), [run archives](run_archives.md) |
| Save results and reanalyze observations | [Saved evidence](saved_evidence.md) |
| Use commands or look up parameters | [CLI](cli.md), [API reference](api/index.md) |
| Understand limitations | [Roadmap](ROADMAP.md), the selected method's guide |
| Check known defects in Qiskit and other dependencies, and how NWQLib handles them | [Dependency issues](dependency_issues.md) |
| Develop a method or maintain the package | [Run your own circuit](own_circuit.md), [algorithm protocol](algorithm_protocol.md), [code tour](CODE_TOUR.md), [maintenance](MAINTENANCE.md) |
| Trace code to its paper, equation and reason | [Code tour](CODE_TOUR.md#find-the-source-and-reason-for-a-line-of-code), [references](references.md) |
| Find the bound, resource law or error budget behind a result, its proof or source, and its code | [Mathematics](mathematics.md) |

Method guides define the scientific quantities, assumptions and evidence limits. [References](references.md) lists the cited literature.
