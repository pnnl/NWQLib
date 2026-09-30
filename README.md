# NWQLib: Northwest Quantum Library

NWQLib applies quantum algorithms to scientific problems. A workflow connects your inputs and requested output to a selected method, execution and a report of the result, evidence and resources. Native circuit execution uses Qiskit and an explicitly selected backend.

IR# PNNL-SA-227989

## Start here

**The [example notebooks](examples/) solve complete scientific problems from input to result, error and circuit cost, and the [examples guide](docs/examples.md) points you to the one that matches your problem.**

**The [mathematics page](docs/mathematics.md) states the bounds, resource laws and error budgets that NWQLib uses, each with its proof or source and the code that implements it.**

From a source checkout, install the package with the local Aer executor:

```bash
python -m pip install -e ".[aer]"
```

To solve `du/dt = -A u` with a small physical input:

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

The default returns the physical solution approximation using nine total qubits. The [quickstart](docs/quickstart.md) compares it with an independent reference and shows how to change the approximation. The base install, `python -m pip install -e .`, supports metadata, admitted inputs, default LCHS planning and supported classical methods. Other planning paths follow their Method's input/conversion requirements; native preparation and execution require the selected backend extra. Method guides describe these boundaries.

The notebooks in `examples/` apply NWQLib to molecular ground-state energy, a linear system, linear dynamics and optimization, and one estimates the resources of a linear system, heat flow, two spin chains, a GCiM trial basis and a QHD optimization step at up to 100 qubits. The [examples guide](docs/examples.md) describes the work each notebook performs.

## Choose a task

| Task | Read |
| --- | --- |
| Learn the Python workflow | [Quickstart](docs/quickstart.md), [examples](docs/examples.md) |
| Compare NWQLib with other quantum packages | [Why NWQLib](docs/why_nwqlib.md) |
| Discover methods or use the command line | [CLI guide](docs/cli.md) |
| Compare methods and inspect results | [Scientist workflow](docs/scientist.md) |
| Supply operators and states | [Input access](docs/inputs.md) |
| Estimate resources and assess a device | [Resource estimates](docs/resources.md), [device profiles](docs/profiles.md) |
| Run locally or submit to a provider | [Prepared execution](docs/prepared_execution.md), [backends](docs/backends.md) |
| Save results or continue a workflow | [Saved evidence](docs/saved_evidence.md), [run archives](docs/run_archives.md) |
| Look up a signature | [API reference](docs/api/index.md) |
| Check known defects in Qiskit and other dependencies, and how NWQLib handles them | [Dependency issues](docs/dependency_issues.md) |
| Extend or maintain NWQLib | [Run your own circuit](docs/own_circuit.md), [Contributing](CONTRIBUTING.md), [code tour](docs/CODE_TOUR.md), [maintenance](docs/MAINTENANCE.md) |
| Trace code to its paper, equation and reason | [Code tour](docs/CODE_TOUR.md#find-the-source-and-reason-for-a-line-of-code), [references](docs/references.md) |
| Find the bound, resource law or error budget behind a result, its proof or source, and its code | [Mathematics](docs/mathematics.md) |

## Algorithms

| Scientific task | Methods |
| --- | --- |
| Normalized expectation of a finite real Pauli sum | [Expectation](docs/algorithms/expectation.md) |
| Energy estimates from moments or a selected subspace | [Chebyshev Lanczos](docs/algorithms/lanczos.md), [fixed and adaptive GCiM](docs/algorithms/gcim.md) |
| Phase or energy estimation | [QPE: QCELS, RWPE, SPE and RFE](docs/algorithms/qpe.md) |
| Time-independent linear dynamics | [LCHS](docs/algorithms/lchs.md) |
| Linear systems | [QLS](docs/algorithms/qls.md) |
| Box-constrained optimization | [QHD](docs/algorithms/qhd.md) |
| Optimization over a box with equality or inequality constraints | [QHD augmented Lagrangian](docs/algorithms/qhd.md#constrained-problems) |

Methods report the quantity actually obtained and any unresolved accuracy conditions. A projected energy, local statistical interval or completed simulation does not by itself establish the full requested scientific claim. The [roadmap](docs/ROADMAP.md) records current limitations and open work. Reusable state preparation, block encoding, LCU, QSP/QSVT and evolution primitives have their own [API references](docs/api/index.md).

## Backend support

Aer supports local execution. NWQ-Sim CPU execution and process-exit continuation have been exercised on macOS and Linux with the qualified build described in the [backend guide](docs/backends.md). IBM Runtime, IonQ and Nexus have offline SDK and injected-result qualification. Live provider accounts, queues and QPUs remain unqualified. Slurm and GPU/site support have their documented offline qualification boundaries.

The [Nexus guide](docs/nexus.md#costs-timeout-and-qualification) documents its pandas dependency exception and missing per-request timeout. See [Slurm](docs/slurm.md) for explicit site configuration. Each method/backend/readout combination must satisfy its own capability checks.

## Authors and Developers

For questions, bug reports or collaboration, contact Muqing Zheng (muqing.zheng@pnnl.gov).

Affiliations are those at the time of contribution.

 - Muqing Zheng, Pacific Northwest National Laboratory
 - Chenxu Liu, Pacific Northwest National Laboratory
 - Zhixin Song, Pacific Northwest National Laboratory and Georgia Institute of Technology
 - Zeguan Wu, Pacific Northwest National Laboratory and University of Pittsburgh
 - Xiangyu Li, Pacific Northwest National Laboratory
 - Mingze Li, Pacific Northwest National Laboratory
 - Nicholas P. Bauman, Pacific Northwest National Laboratory
 - Samuel A. Stein, Pacific Northwest National Laboratory
 - Johannes Mülmenstädt, Pacific Northwest National Laboratory
 - Yousu Chen, Pacific Northwest National Laboratory
 - Nathan Wiebe, Pacific Northwest National Laboratory and University of Toronto
 - Ang Li, Pacific Northwest National Laboratory and University of Washington
 - Karol Kowalski, Pacific Northwest National Laboratory

## Acknowledgements

This work was supported by Pacific Northwest National Laboratory's Quantum Algorithms and Architecture for Domain Science (QuAADS) Laboratory Directed Research and Development (LDRD) Initiative. This material is based upon work supported by the U.S. Department of Energy, Office of Science, National Quantum Information Science Research Centers, Quantum Science Center (QSC). The Pacific Northwest National Laboratory is operated by Battelle for the U.S. Department of Energy under Contract DE-AC05-76RL01830.

[Algorithm and subroutine references](docs/references.md).
