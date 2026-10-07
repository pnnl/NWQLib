# Install and first result

<a id="quickstart"></a>Install NWQLib, solve a small linear differential equation, check the answer against an independent reference, and see what the calculation costs. The same workflow applies to the other problems on the [home page](index.md).

## Install {#install}

NWQLib requires Python 3.12 or later. Install it from PyPI with the local Aer simulator:

```bash
python -m pip install "nwqlib[aer]"
```

Add an extra for each further capability you need, for example `python -m pip install "nwqlib[aer,notebook,chemistry]"`:

| Extra | Use it for | Packages it adds |
| --- | --- | --- |
| `aer` | Running circuits on the local Aer simulator | Qiskit ≥ 2.5.2, Qiskit Aer ≥ 0.17.2 |
| `qiskit` | Building Qiskit circuits, passing Qiskit objects as input, exporting circuits, and the Qiskit-based classical evaluations that some methods choose | Qiskit ≥ 2.5.2 |
| `notebook` | Running the [example notebooks](examples.md) | Jupyter, ipykernel, nbclient, Matplotlib |
| `chemistry` | Building molecular Hamiltonians | Qiskit, PySCF, OpenFermion |
| `tensor` | Matrix-product-state (MPS) circuit state preparation | Qiskit, plus `scikit_tt` installed as shown below |
| `qasm` | Reading an exported OpenQASM 3 file back into a Qiskit circuit ([Export OpenQASM](qasm-streaming.md)) | Qiskit, the OpenQASM 3 parser, the Qiskit QASM3 importer |
| `ibm` | [IBM Runtime](ibm.md) | Qiskit, qiskit-ibm-runtime ≥ 0.49.0 |
| `ionq` | [IonQ](ionq.md) | Qiskit, qiskit-ionq ≥ 1.1.1, requests ≥ 2.34.2, urllib3 ≥ 2.8.0 |
| `nexus` | [Quantinuum Nexus](nexus.md) H2 | Qiskit, qnexus ≥ 0.49.0, pytket ≥ 2.18.1, pytket-qiskit ≥ 0.78.0, selene-core ≥ 0.3.2 |
| `nwqec` | Logical Clifford+T compilation of small circuits ([Estimate fault-tolerant resources](fault-tolerant-resources.md)) | Qiskit, nwqec 0.1.2 |
| `qre` | QDK physical resource projection of a compiled circuit ([Estimate fault-tolerant resources](fault-tolerant-resources.md)) | qdk[qre] 1.32.3 |
| `dev` | Running the test suite (add `qasm` for the parser tests) | pytest, pytest-xdist, Ruff, jsonschema |
| `docs` | Building this documentation with MkDocs | mkdocs-material ≥ 9.5, mkdocstrings[python] ≥ 0.25 |

The MPS route of `tensor` also needs `scikit_tt`, which is not on PyPI. Install it separately:

```bash
python -m pip install "scikit_tt @ git+https://github.com/PGelss/scikit_tt.git"
```

The stable test environment pins its `scikit_tt` commit in `docs/ENVIRONMENT_LOCK.txt`. To work on NWQLib itself, install an editable checkout as described in [Set up, test and build](development/setup.md).

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

```text
[ 0.96006038-1.54102084e-12j -0.00513885+3.61167323e-13j]
```

The result is the physical solution vector u(0.1), including its scale and phase. It is not normalized to unit length.

LCHS, the linear combination of Hamiltonian simulation of An, Childs and Lin (ACL, arXiv:2312.03916v2, Eq. (6)), writes `exp(-tA)` for a matrix A with positive semidefinite Hermitian part as an integral over a kernel variable k of unitary evolutions, and approximates the integral by a quadrature sum. NWQLib shifts an A whose Hermitian part is not positive semidefinite and restores the resulting growth. Each quadrature node is one branch of that sum. SELECT is the circuit block that applies the branch whose index an address register holds (ACL Appendix A.3, Lemma 24, Eq. (178)). The default Aer circuit uses one system qubit and eight coefficient ancillas, which hold the address. It has 204 branches, padded to 256 address slots, the branch count rounded up to a power of two. The [LCHS guide](algorithms/lchs.md) describes the construction.

## Check the answer

Compute the same solution with SciPy's matrix exponential and compare:

```python
import numpy as np
from scipy.linalg import expm

A = np.array([[0.4, 0.15], [0.05, 0.25]])
reference = expm(-0.1 * A) @ np.array([1.0, 0.0])
print(reference)
print(np.linalg.norm(result.solution - reference))
```

```text
[ 0.96082565 -0.00484022]
0.0008214720329548587
```

The printed absolute L2 discrepancy, about `.000821472`, is that of the default finite LCHS approximation in this example. LCHS gives half of `LCHS.approximation_tolerance`, 0.01 by default, to the tail of the k integral that the cutoff drops and half to the k quadrature. Here the tail bound is `.005` and the quadrature bound is about `.00431140`. For this unit input without a shift, they equal the facts `kernel_approximation` and `k_quadrature` in `result.plan.facts` ([LCHS frames](conventions.md#lchs-frames)). These two component bounds do not bound the total physical-output error, which also depends on the preparation, evolution and numerical approximations. The [LCHS guide](algorithms/lchs.md#selection-and-accuracy) describes these bounds, and [Check accuracy and verify a result](verification.md) describes NWQLib's own checks.

## Change the approximation

A smaller `approximation_tolerance` uses more quadrature nodes. Evaluate the finer finite sum classically:

```python
refined = solve(
    problem,
    method=LCHS(approximation_tolerance=0.001),
    execution="classical",
)
print(refined.solution)
print(np.linalg.norm(refined.solution - reference))
```

```text
[ 0.96084571+1.45318296e-17j -0.00488956-7.01766797e-19j]
5.3260455987202175e-05
```

For this input, the finer construction uses 396 nodes and has absolute L2 discrepancy about `.0000532605`. Its dense quantum circuit would need 512 address slots and ten total qubits, which exceeds the default `max_dense_select_slots=256`, so this comparison evaluates the finite sum classically. These two observations do not establish monotonic convergence for every input. Each branch of the dense SELECT is a classically computed matrix exponential. Larger systems need a supported structured construction with its own limit check.

## See the cost before running {#inspect-or-select-work-before-running-it}

`plan` chooses the construction for this input and returns it, with its costs, as a [Plan](how_it_works.md) before any circuit exists. `estimate` reads the Plan and runs no circuit:

```python
from nwqlib import estimate, plan

lchs_plan = plan(problem, method=LCHS())
resources = estimate(lchs_plan)
print(resources.quantity("logical_width", location="logical_device"))
operations = resources.quantity("operations")
print(operations.interpretation)
```

```text
logical_width at logical_device: 9 count [exact]
  basis=selected_logical; lifecycle=planned; population=simultaneous live footprint
unavailable
```

Each quantity prints its value, its unit and a label (`exact`, `upper_bound`, `estimate`, `conditional` or `unavailable`). The second line gives the gate basis (`basis`), the workflow stage (`lifecycle`, which is `planned` before any circuit exists) and what is counted (`population`). Here the circuit holds 9 qubits at the same time, an exact count. The operation count is unavailable because the [blocks](glossary.md#block) (subroutines) of this construction have no gate-count formula. An unavailable count carries its reason and is never treated as zero.

Planning computes the numerical data the Method needs and takes no measurements. Run the Plan with `solve(lchs_plan)`, or pass it to `prepare` and `submit` to control preparation and submission yourself ([Run on a backend](prepared_execution.md)). A Result keeps its Plan as `result.plan`. [Estimate resources](resources.md) explains how counts from formulas differ from counts of a built circuit and from device predictions.

## Save and load the result

Pass a path that does not exist yet. `save` creates that directory with the result's metadata and data files, and it raises `FileExistsError` when the path already exists, so a saved result is never overwritten:

```python
from nwqlib import load_result

result.save("linear-dynamics-result")
restored = load_result("linear-dynamics-result")
print(restored.solution)
```

```text
[ 0.96006038-1.54102084e-12j -0.00513885+3.61167323e-13j]
```

## Next steps

- [Examples](examples.md): the linear dynamics notebook (`examples/lchs_linear_dynamics_intro.ipynb`) applies LCHS to advection-diffusion. It separates product-formula error from quadrature error and compares kernels, quadratures, and exact and MPS state preparation by their errors and success probabilities, with compiled gate counts for the isolated state loaders.
- [Plan, compare and solve](scientist.md) compares Lanczos and FixedGCIM for the same `Eigenproblem`.
- [Supply inputs](inputs.md) covers dense, sparse and structured inputs.
- [Run on a backend](prepared_execution.md) and [Continue an interrupted run](run_archives.md) cover submitting to a backend and continuing a saved run.
- [Save, load and reanalyze results](saved_evidence.md) covers saved results and reanalysis.
- [Use the command line](cli.md) lists methods and their parameters and inspects saved reports.
- [How NWQLib works](how_it_works.md) explains Problem, Method, Plan, Run and Result.
