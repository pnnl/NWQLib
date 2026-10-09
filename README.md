# NWQLib: Northwest Quantum Library

[![PyPI](https://img.shields.io/pypi/v/nwqlib.svg)](https://pypi.org/project/nwqlib/)    [![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://pypi.org/project/nwqlib/)    [![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](https://github.com/pnnl/NWQLib/blob/main/LICENSE)    [![Documentation](https://img.shields.io/badge/docs-pnnl.github.io%2FNWQLib-blue.svg)](https://pnnl.github.io/NWQLib/)    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23074265.svg)](https://doi.org/10.5281/zenodo.23074265)

NWQLib applies quantum algorithms to scientific problems. You state the problem and choose a method, and NWQLib plans the construction, runs it on the backend you choose and returns the result with its resource counts and what is known about its error. Circuits are built with Qiskit.

Documentation: <https://pnnl.github.io/NWQLib/>

## Installation

NWQLib requires Python 3.12 or later. Install the package with the local Aer simulator:

```bash
python -m pip install "nwqlib[aer]"
```

The base package depends on NumPy 2.5.2 or later, SciPy 1.18.1 or later, SymPy 1.14.0 or later and Pydantic 2.13.5 or later, below 3. The extras `aer`, `qiskit`, `ibm`, `ionq`, `nexus`, `chemistry`, `tensor`, `qasm` and `nwqec` require Qiskit 2.5.2 or later. The validated CI environment in [`docs/ENVIRONMENT_LOCK.txt`](https://github.com/pnnl/NWQLib/blob/main/docs/ENVIRONMENT_LOCK.txt) uses Python 3.12.14, Qiskit 2.5.2 and Qiskit Aer 0.17.2.

The base package, `python -m pip install nwqlib`, describes the methods, accepts numerical inputs, plans LCHS with its default settings and supports classical evaluation with `execution="classical"` for LCHS, QLS, QCELS, SPE, RFE, RWPE, QHD, Expectation, Lanczos, FixedGCIM and ADAPT within their supported numerical-input and method-option domains. Classical evaluation that expands a circuit-defined initial or trial state also requires Qiskit. Building circuits needs the `qiskit` extra, and running them needs the extra of the chosen backend (`aer`, `ibm`, `ionq`, `nexus`), each of which includes Qiskit. For example, `python -m pip install "nwqlib[aer,notebook,chemistry]"` adds the example notebooks and the molecular Hamiltonian builder, and `python -m pip install "nwqlib[ibm]"` adds IBM Runtime. [Install and first result](https://pnnl.github.io/NWQLib/quickstart/#install) lists every extra, and [Choose a backend](https://pnnl.github.io/NWQLib/backends/) explains what each backend supports. To work on NWQLib itself, see [Set up, test and build](https://pnnl.github.io/NWQLib/development/setup/).

## First result

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

```text
[ 0.96006038-1.54102084e-12j -0.00513885+3.61167323e-13j]
```

The default returns an approximation of the physical solution using nine qubits in total. [Install and first result](https://pnnl.github.io/NWQLib/quickstart/) compares it with an independent reference and shows how to change the approximation. Each method accepts its own input forms and conversions, which its algorithm guide describes.

The [example notebooks](https://github.com/pnnl/NWQLib/tree/main/examples) solve complete scientific problems from input to result, error and circuit cost: molecular ground-state energy, a linear system, linear dynamics and optimization. One more notebook estimates the resources of a linear system, heat flow, two spin chains and a GCiM trial basis at 80 to 100 system qubits, where the largest circuit, LCHS for heat flow, has 109 qubits in total. It also estimates one QHD step on a three-variable grid of 32 points per variable, which takes 96 qubits in the one-hot encoding. The [examples guide](https://pnnl.github.io/NWQLib/examples/) points you to the notebook that matches your problem. The [mathematics page](https://pnnl.github.io/NWQLib/mathematics/) states the bounds, resource formulas and error budgets that NWQLib uses, each with its proof or source and the code that implements it.

## Choose a task

| Task | Read |
| --- | --- |
| Install and get a first result | [Install and first result](https://pnnl.github.io/NWQLib/quickstart/), [examples](https://pnnl.github.io/NWQLib/examples/) |
| Learn what Problem, Method, Plan, Run and Result are | [How NWQLib works](https://pnnl.github.io/NWQLib/how_it_works/) |
| Choose a problem and output | [Choose a problem and output](https://pnnl.github.io/NWQLib/problems/) |
| Compare NWQLib with other quantum packages | [Why NWQLib](https://pnnl.github.io/NWQLib/why_nwqlib/) |
| Compare methods and inspect results | [Plan, compare and solve](https://pnnl.github.io/NWQLib/scientist/) |
| Supply operators and states | [Supply inputs](https://pnnl.github.io/NWQLib/inputs/) |
| Estimate resources and check device fit | [Estimate resources](https://pnnl.github.io/NWQLib/resources/), [Check device fit and run time](https://pnnl.github.io/NWQLib/profiles/) |
| Check a result against a reference or tolerance | [Check accuracy and verify a result](https://pnnl.github.io/NWQLib/verification/) |
| Run locally or submit to a provider | [Run on a backend](https://pnnl.github.io/NWQLib/prepared_execution/), [Choose a backend](https://pnnl.github.io/NWQLib/backends/) |
| Save results or continue an interrupted run | [Save, load and reanalyze results](https://pnnl.github.io/NWQLib/saved_evidence/), [Continue an interrupted run](https://pnnl.github.io/NWQLib/run_archives/) |
| Run your own circuit or compare your own method | [Run your own circuit](https://pnnl.github.io/NWQLib/own_circuit/), [Add a method](https://pnnl.github.io/NWQLib/algorithm_protocol/) |
| List methods or use the command line | [Use the command line](https://pnnl.github.io/NWQLib/cli/) |
| Look up a signature | [API reference](https://pnnl.github.io/NWQLib/api/) |
| Check known defects in Qiskit and other dependencies, and how NWQLib handles them | [Dependency issues](https://pnnl.github.io/NWQLib/dependency_issues/) |
| Trace code to its paper, equation and reason | [Code tour](https://pnnl.github.io/NWQLib/CODE_TOUR/#find-the-source-and-reason-for-a-line-of-code), [references](https://pnnl.github.io/NWQLib/references/) |
| Find the bound, resource formula or error budget behind a result, its proof or source, and its code | [Mathematics](https://pnnl.github.io/NWQLib/mathematics/) |
| Maintain NWQLib | [Set up, test and build](https://pnnl.github.io/NWQLib/development/setup/), [contribution policy](https://pnnl.github.io/NWQLib/FRAMEWORK/), [Contributing](https://github.com/pnnl/NWQLib/blob/main/CONTRIBUTING.md) |

Each method's tests are in `tests/`, in files named after the method, such as `tests/test_lchs_*.py` and `tests/test_qls_*.py`. The [examples guide](https://pnnl.github.io/NWQLib/examples/) lists the notebook of each method.

## Algorithms

| Scientific task | Methods |
| --- | --- |
| Normalized expectation of a finite real Pauli sum | [Expectation](https://pnnl.github.io/NWQLib/algorithms/expectation/) |
| Energy estimates from moments or a chosen subspace | [Chebyshev Lanczos](https://pnnl.github.io/NWQLib/algorithms/lanczos/) (arXiv:2208.00567v4), [fixed and adaptive GCiM](https://pnnl.github.io/NWQLib/algorithms/gcim/) (fixed arXiv:2212.09205v1, adaptive arXiv:2312.07691v3) |
| Phase or energy estimation | [QPE: QCELS, RWPE, SPE and RFE](https://pnnl.github.io/NWQLib/algorithms/qpe/) (QCELS arXiv:2211.11973v2, RWPE arXiv:2208.04526v1, SPE arXiv:2110.12071v2, RFE arXiv:2209.11322v3) |
| Time-independent linear dynamics | [LCHS](https://pnnl.github.io/NWQLib/algorithms/lchs/) (arXiv:2312.03916v2) |
| Linear systems | [QLS: QSVT inverse polynomial (`qsvt_inverse`, the default) and the Dalzell kernel shortcut (`shortcut_native_svp`, `shortcut_dilation`)](https://pnnl.github.io/NWQLib/algorithms/qls/) (QSVT arXiv:1806.01838v1, Dalzell arXiv:2406.12086v2) |
| Box-constrained optimization | [QHD](https://pnnl.github.io/NWQLib/algorithms/qhd/) (arXiv:2303.01471v1) |
| Optimization over a box with equality or inequality constraints | [QHD augmented Lagrangian](https://pnnl.github.io/NWQLib/algorithms/qhd/#constrained-problems) (arXiv:2605.12066v1) |

Each method reports the quantity it obtained and any accuracy conditions that remain unresolved. Each method is checked on small cases against independent references, and [Limitations and open work](https://pnnl.github.io/NWQLib/ROADMAP/) lists what has been checked for each method and its current limitations. The resource-estimation notebook runs no circuit. A projected energy, a local statistical interval or a completed simulation does not by itself establish the full requested scientific claim. State preparation, block encoding, LCU, QSP/QSVT and evolution subroutines have their own [API reference pages](https://pnnl.github.io/NWQLib/api/).

## Backend support

| Backend | Runs where | Tested against the live service |
| --- | --- | --- |
| [Aer](https://pnnl.github.io/NWQLib/aer/) | Locally | Not applicable |
| [NWQ-Sim](https://pnnl.github.io/NWQLib/nwqsim/) | Locally, with the NWQ-Sim build described in its guide. CPU execution, and continuing a run after Python exits, were tested on macOS and Linux with that build | Not applicable |
| [Slurm](https://pnnl.github.io/NWQLib/slurm/) | NWQ-Sim on your cluster, with explicit site configuration | No. Scheduler and site-configuration handling is tested offline. Site allocation, MPI and GPU execution are not tested |
| [IBM Runtime](https://pnnl.github.io/NWQLib/ibm/) | IBM Quantum service | No. SDK calls and result handling are tested offline. Live accounts, queues and QPUs are not tested |
| [IonQ](https://pnnl.github.io/NWQLib/ionq/) | IonQ service | No. Circuit conversion and result handling are tested offline. QPUs are not tested |
| [Quantinuum Nexus](https://pnnl.github.io/NWQLib/nexus/) | Nexus H2 service | No. Conversion, remote-job handling and results are tested offline. Nexus needs a [pandas dependency exception and has no per-request timeout](https://pnnl.github.io/NWQLib/nexus/#costs-timeout-and-qualification) |

A backend runs a method only when it supports that method's circuits and readout. [Choose a backend](https://pnnl.github.io/NWQLib/backends/) gives the details.

## Citation

If you use NWQLib in your work, please cite it through its Zenodo record, which resolves to the latest version:

```bibtex
@software{nwqlib,
  author  = {Zheng, Muqing and Liu, Chenxu and Song, Zhixin and Wu, Zeguan and Li, Xiangyu and Li, Mingze and Bauman, Nicholas P. and Stein, Samuel A. and M{\"u}lmenst{\"a}dt, Johannes and Chen, Yousu and Wiebe, Nathan and Li, Ang and Kowalski, Karol},
  title   = {NWQLib},
  year    = {2026},
  version = {1.0.2.post1},
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

NWQLib is released under the BSD 2-Clause License, given in [`LICENSE`](https://github.com/pnnl/NWQLib/blob/main/LICENSE). Information release number PNNL-SA-227989.
