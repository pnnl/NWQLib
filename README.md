# NWQLib: Northwest Quantum Library

[![PyPI](https://img.shields.io/pypi/v/nwqlib.svg)](https://pypi.org/project/nwqlib/)    [![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://pypi.org/project/nwqlib/)    [![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](https://github.com/pnnl/NWQLib/blob/main/LICENSE)    [![Documentation](https://img.shields.io/badge/docs-pnnl.github.io%2FNWQLib-blue.svg)](https://pnnl.github.io/NWQLib/)    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23074265.svg)](https://doi.org/10.5281/zenodo.23074265)

NWQLib applies quantum algorithms to scientific problems. A workflow connects your inputs and requested output to a selected method, execution and a report of the result, evidence and resources. Native circuit execution uses Qiskit and an explicitly selected backend.

**Documentation: <https://pnnl.github.io/NWQLib/>**

## Installation

NWQLib requires Python 3.12 or later. Install the package with the local Aer simulator:

```bash
python -m pip install "nwqlib[aer]"
```

The base package, `python -m pip install nwqlib`, supports metadata, admitted inputs, default LCHS planning and the classical methods. Native preparation and execution need the extra of the selected backend (`aer`, `ibm`, `ionq`, `nexus`). The [framework page](https://pnnl.github.io/NWQLib/FRAMEWORK/) lists every optional group, and the [backends guide](https://pnnl.github.io/NWQLib/backends/) explains what each backend supports. For development, install an editable checkout with `python -m pip install -e ".[dev,aer]"`.

## Start here

**The [example notebooks](https://github.com/pnnl/NWQLib/tree/main/examples) solve complete scientific problems from input to result, error and circuit cost, and the [examples guide](https://pnnl.github.io/NWQLib/examples/) points you to the one that matches your problem. The [mathematics page](https://pnnl.github.io/NWQLib/mathematics/) states the bounds, resource laws and error budgets that NWQLib uses, each with its proof or source and the code that implements it.**

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

The default returns the physical solution approximation using nine total qubits. The [quickstart](https://pnnl.github.io/NWQLib/quickstart/) compares it with an independent reference and shows how to change the approximation. Other planning paths follow their Method's input and conversion requirements, and the method guides describe these boundaries.

The notebooks in `examples/` apply NWQLib to molecular ground-state energy, a linear system, linear dynamics and optimization, and one estimates the resources of a linear system, heat flow, two spin chains, a GCiM trial basis and a QHD optimization step at up to 100 qubits. The [examples guide](https://pnnl.github.io/NWQLib/examples/) describes the work each notebook performs.

## Choose a task

| Task | Read |
| --- | --- |
| Learn the Python workflow | [Quickstart](https://pnnl.github.io/NWQLib/quickstart/), [examples](https://pnnl.github.io/NWQLib/examples/) |
| Compare NWQLib with other quantum packages | [Why NWQLib](https://pnnl.github.io/NWQLib/why_nwqlib/) |
| Discover methods or use the command line | [CLI guide](https://pnnl.github.io/NWQLib/cli/) |
| Compare methods and inspect results | [Scientist workflow](https://pnnl.github.io/NWQLib/scientist/) |
| Supply operators and states | [Input access](https://pnnl.github.io/NWQLib/inputs/) |
| Estimate resources and assess a device | [Resource estimates](https://pnnl.github.io/NWQLib/resources/), [device profiles](https://pnnl.github.io/NWQLib/profiles/) |
| Run locally or submit to a provider | [Prepared execution](https://pnnl.github.io/NWQLib/prepared_execution/), [backends](https://pnnl.github.io/NWQLib/backends/) |
| Save results or continue a workflow | [Saved evidence](https://pnnl.github.io/NWQLib/saved_evidence/), [run archives](https://pnnl.github.io/NWQLib/run_archives/) |
| Look up a signature | [API reference](https://pnnl.github.io/NWQLib/api/) |
| Check known defects in Qiskit and other dependencies, and how NWQLib handles them | [Dependency issues](https://pnnl.github.io/NWQLib/dependency_issues/) |
| Extend or maintain NWQLib | [Run your own circuit](https://pnnl.github.io/NWQLib/own_circuit/), [Contributing](https://github.com/pnnl/NWQLib/blob/main/CONTRIBUTING.md), [code tour](https://pnnl.github.io/NWQLib/CODE_TOUR/), [maintenance](https://pnnl.github.io/NWQLib/MAINTENANCE/) |
| Trace code to its paper, equation and reason | [Code tour](https://pnnl.github.io/NWQLib/CODE_TOUR/#find-the-source-and-reason-for-a-line-of-code), [references](https://pnnl.github.io/NWQLib/references/) |
| Find the bound, resource law or error budget behind a result, its proof or source, and its code | [Mathematics](https://pnnl.github.io/NWQLib/mathematics/) |

## Algorithms

| Scientific task | Methods |
| --- | --- |
| Normalized expectation of a finite real Pauli sum | [Expectation](https://pnnl.github.io/NWQLib/algorithms/expectation/) |
| Energy estimates from moments or a selected subspace | [Chebyshev Lanczos](https://pnnl.github.io/NWQLib/algorithms/lanczos/), [fixed and adaptive GCiM](https://pnnl.github.io/NWQLib/algorithms/gcim/) |
| Phase or energy estimation | [QPE: QCELS, RWPE, SPE and RFE](https://pnnl.github.io/NWQLib/algorithms/qpe/) |
| Time-independent linear dynamics | [LCHS](https://pnnl.github.io/NWQLib/algorithms/lchs/) |
| Linear systems | [QLS: QSVT inverse polynomial (`qsvt_inverse`, the default) and the Dalzell kernel shortcut (`shortcut_native_svp`, `shortcut_dilation`)](https://pnnl.github.io/NWQLib/algorithms/qls/) |
| Box-constrained optimization | [QHD](https://pnnl.github.io/NWQLib/algorithms/qhd/) |
| Optimization over a box with equality or inequality constraints | [QHD augmented Lagrangian](https://pnnl.github.io/NWQLib/algorithms/qhd/#constrained-problems) |

Methods report the quantity actually obtained and any unresolved accuracy conditions. A projected energy, local statistical interval or completed simulation does not by itself establish the full requested scientific claim. The [roadmap](https://pnnl.github.io/NWQLib/ROADMAP/) records current limitations and open work. Reusable state preparation, block encoding, LCU, QSP/QSVT and evolution primitives have their own [API references](https://pnnl.github.io/NWQLib/api/).

## Backend support

Aer supports local execution. NWQ-Sim CPU execution and process-exit continuation have been exercised on macOS and Linux with the qualified build described in the [backend guide](https://pnnl.github.io/NWQLib/backends/). IBM Runtime, IonQ and Nexus have offline SDK and injected-result qualification. Live provider accounts, queues and QPUs remain unqualified. Slurm and GPU/site support have their documented offline qualification boundaries.

The [Nexus guide](https://pnnl.github.io/NWQLib/nexus/#costs-timeout-and-qualification) documents its pandas dependency exception and missing per-request timeout. See [Slurm](https://pnnl.github.io/NWQLib/slurm/) for explicit site configuration. Each method/backend/readout combination must satisfy its own capability checks.

## Citation

If you use NWQLib in your work, please cite it through its Zenodo record, which resolves to the latest version:

```bibtex
@software{nwqlib,
  author  = {Zheng, Muqing and Liu, Chenxu and Song, Zhixin and Wu, Zeguan and Li, Xiangyu and Li, Mingze and Bauman, Nicholas P. and Stein, Samuel A. and M{\"u}lmenst{\"a}dt, Johannes and Chen, Yousu and Wiebe, Nathan and Li, Ang and Kowalski, Karol},
  title   = {NWQLib},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.23074265},
  url     = {https://github.com/pnnl/NWQLib}
}
```

The [references page](https://pnnl.github.io/NWQLib/references/) lists the papers behind each algorithm and subroutine, with the equations NWQLib implements.

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

NWQLib is released under the BSD 2-Clause License; see [`LICENSE`](https://github.com/pnnl/NWQLib/blob/main/LICENSE). Information release number PNNL-SA-227989.
