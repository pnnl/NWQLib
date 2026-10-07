# NWQLib

NWQLib applies quantum algorithms to scientific problems. You state the problem and choose a method, and NWQLib returns the computed quantity with the method, its resource counts and what is known about its error.

| Your scientific problem | Methods and first example |
| --- | --- |
| Linear dynamics, `du/dt = -A u` | [LCHS](algorithms/lchs.md), [first result](quickstart.md) |
| Eigenvalues or phases of an operator | [Lanczos](algorithms/lanczos.md), [GCiM / ADAPT](algorithms/gcim.md), [QPE](algorithms/qpe.md) |
| Ground-state energy estimate of a molecule | [GCiM / ADAPT chemistry inputs](algorithms/gcim.md#chemistry-inputs-and-reference-energies) |
| A linear system, `Ax=b` | [QLS](algorithms/qls.md) |
| Optimization over a box | [QHD](algorithms/qhd.md) |
| Optimization over a box with equality or inequality constraints | [QHD augmented Lagrangian](algorithms/qhd.md#constrained-problems) |
| The expectation of an observable | [Expectation](algorithms/expectation.md) |

NWQLib requires Python 3.12 or later. Install it with the local Aer simulator:

```bash
python -m pip install "nwqlib[aer]"
```

[Install and first result](quickstart.md#install) lists the other extras. This complete cell solves a two-coordinate linear differential equation:

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

```text
[ 0.96006038-1.54102084e-12j -0.00513885+3.61167323e-13j]
```

The default nine-qubit Aer calculation returns the physical vector, including its scale and phase. Its finite approximation comes with component error bounds, which do not bound the total error ([Install and first result](quickstart.md#check-the-answer)). Replace A, the initial state and the time to use your own input. [Install and first result](quickstart.md) checks this answer against an independent reference and shows a finer construction.

## Continue your workflow

| What you want to do | Where to start |
| --- | --- |
| Choose a problem and output | [Choose a problem and output](problems.md) |
| Find a worked example for your problem | [Examples](examples.md) and the method guides |
| Compare methods for the same scientific problem | [Plan, compare and solve](scientist.md) |
| Supply an operator or state | [Supply inputs](inputs.md) |
| Estimate qubits, gates and shots, or check device fit | [Estimate resources](resources.md), [Check device fit and run time](profiles.md) |
| Estimate what an algorithm would cost at 80 to 100 system qubits | [Resource estimation notebook](examples.md#resource-estimation-at-scale) |
| Check a result against a reference or tolerance | [Check accuracy and verify a result](verification.md) |
| Run on a backend or continue an interrupted run | [Run on a backend](prepared_execution.md), [Continue an interrupted run](run_archives.md) |
| Save results and reanalyze observations | [Save, load and reanalyze results](saved_evidence.md) |
| Run your own circuit or compare your own method | [Run your own circuit](own_circuit.md) |
| Use the command line or look up parameters | [Use the command line](cli.md), [API reference](api/index.md) |
| Learn what Problem, Method, Plan, Run and Result are | [How NWQLib works](how_it_works.md) |
| Compare NWQLib with other quantum packages | [Why NWQLib](why_nwqlib.md) |
| Understand limitations | [Limitations and open work](ROADMAP.md), the method's guide |
| Check known defects in Qiskit and other dependencies, and how NWQLib handles them | [Dependency issues](dependency_issues.md) |
| Trace code to its paper, equation and reason | [Code tour](CODE_TOUR.md#find-the-source-and-reason-for-a-line-of-code), [references](references.md) |
| Find the bound, resource formula or error budget behind a result, its proof or source, and its code | [Mathematics](mathematics.md) |
| Maintain NWQLib | [Set up, test and build](development/setup.md) |

Each method guide defines its scientific quantities and assumptions and states what its results do not establish. [References](references.md) lists the cited literature.
