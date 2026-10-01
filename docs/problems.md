# Choose a problem and output

Build the Problem that states your scientific question, then choose the output you want back. Every Problem has a default output, and `plan`, `compare` and `solve` take `output=` for another one. Each Method supports some Problems and outputs, and `python -m nwqlib algorithms` lists them ([Use the command line](cli.md)).

```python
from nwqlib import Expectation, QuadraticForm, solve
from nwqlib.algorithms import ExpectationMethod

Z = [[1, 0], [0, -1]]
problem = Expectation(state=[3, 4], observable=Z)
normalized = solve(problem, method=ExpectationMethod())
quadratic = solve(problem, method=ExpectationMethod(),
                  output=QuadraticForm(observable=Z))
print(normalized.value, quadratic.value)
```

```text
-0.28000000000000014 -7.0000000000000036
```

The default output of an Expectation is the normalized expectation `u† O u / (u† u)`, here `(9 - 16) / 25`. The explicit `QuadraticForm` returns `u† O u`, which keeps the physical magnitude of the state.

## Problem types {#problem-types}

Import each Problem from `nwqlib`. [Supply inputs](inputs.md) lists the matrix, Pauli, state and structured forms that each field accepts.

| Problem | Scientific question | Default output | Methods |
| --- | --- | --- | --- |
| `Eigenproblem(A=...)` | Smallest eigenvalue of a Hermitian operator, optionally in an explicit subspace or sector | Eigenvalue | [Lanczos](algorithms/lanczos.md), [FixedGCIM and ADAPT](algorithms/gcim.md), [QCELS, SPE, RFE and RWPE](algorithms/qpe.md) |
| `LinearDynamics(A=..., initial_state=..., time=...)` | Time-independent `du/dt = -A u + source` from the given initial time and state | Solution | [LCHS](algorithms/lchs.md) |
| `LinearSystem(A=..., b=...)` | Physical solution of `A x = b` | Solution | [QLS](algorithms/qls.md) |
| `Expectation(state=..., observable=...)` | Normalized expectation of the given Hermitian observable in the given state | NormalizedExpectation | [ExpectationMethod](algorithms/expectation.md) |
| `SpectralEstimation(hamiltonian=... or unitary=..., initial_state=...)` | Phase information for a Hamiltonian or unitary, with a given initial state | Eigenphase | [QCELS, SPE, RFE and RWPE](algorithms/qpe.md) |
| `Optimization(objective=..., variables=..., bounds=...)` | A SymPy objective on ordered real variables with finite box bounds | OptimizationCandidate | [QHD](algorithms/qhd.md) |
| `ConstrainedOptimization(..., equalities=..., inequalities=...)` | The Optimization question subject to at least one equality `h_i(x) = 0` or inequality `g_j(x) <= 0` | OptimizationCandidate | [QHD through `solve_augmented_lagrangian`](algorithms/qhd.md#constrained-problems) |

Not every Method supports every valid Problem. Construction checks Hermiticity where the Problem requires it, dimensions and coordinate order. Each Method checks its own further assumptions. For example, a `unitary` given to SpectralEstimation is not proved unitary by the Problem, and the Method checks it. Saved JSON stores symbolic expressions as descriptions and never evaluates them as code.

Accepted inputs keep their physical magnitude, complex phase and original coordinates. A full-space eigenvalue and an eigenvalue in an explicitly supplied subspace are recorded as different quantities. The Problem sets the scope of its output, and the output records the condition its value depends on, such as a nonzero norm or an explicitly supplied subspace.

## Outputs {#outputs}

Import each output from `nwqlib` and pass it as `output=`. `NormalizedExpectation`, `QuadraticForm` and `NormSquared` are separate quantities.

| Output | Meaning | Error metric |
| --- | --- | --- |
| `Eigenvalue()` | An eigenvalue. For an Eigenproblem, the smallest one in the full space, an explicit sector or an explicit subspace | Absolute error |
| `Eigenphase()` | Phase `phi` in turns, in `[0, 1)`, with `U v = exp(2 pi i phi) v` | Circular distance in dimensionless turns |
| `NormalizedExpectation(observable=O)` | `u† O u / (u† u)` for nonzero `u` | Absolute error |
| `QuadraticForm(observable=O)` | `u† O u`, using the physical magnitude of `u` | Absolute error |
| `NormSquared()` | `u† u` | Absolute error |
| `Solution()` | Physical solution amplitudes in the original coordinates, including phase | `l2` |
| `StateVector(normalization=..., global_phase=...)` | Simulator amplitudes with an explicit choice of physical or unit normalization, and of physical phase or equivalence modulo global phase | `l2`, or `phase_aligned_l2` modulo global phase |
| `Samples()` | Measurement outcomes in the original coordinates, for the shots in which the Method's success event occurred | Total variation between distributions |
| `OptimizationCandidate()` | A candidate point of the box and its objective value | Objective gap, which does not imply an optimality certificate |

Declaring an output computes nothing. It creates no measured data and no live quantum state.

## Units {#units}

A unit is a label, such as `Eigenproblem(A=H, unit="Hartree")`. NWQLib converts no units, and an explicit output unit records your convention without proving it.

- For LinearDynamics and LinearSystem, `problem.unit` labels the physical solution amplitude. Solution and a physical StateVector carry that unit, and NormSquared carries its square. Errors of a unit-normalized StateVector are dimensionless.
- For an Expectation, `problem.unit` labels the normalized expectation. It supplies no unit for the state amplitudes, so a QuadraticForm, a norm or a vector of that Problem has an unspecified unit unless the output declares one, as in `QuadraticForm(observable=O, unit="...")`.
- Eigenphase is in turns, and Samples is dimensionless.

An output unit whose symbol differs from the unit that the Problem defines, or that the output kind fixes, raises `ValueError` at planning, because the value would carry the wrong label without conversion.

## Constraint residuals {#constraint-residuals}

ConstrainedOptimization stores each constraint as a residual, with `h_i(x) = 0` for equalities and `g_j(x) <= 0` for inequalities. This is the form of problem (4.1) in Birgin and Martínez, *Practical Augmented Lagrangian Methods for Constrained Optimization*, SIAM 2014, doi:10.1137/1.9781611973365, with the box as its set Ω.

| You pass | Stored residual | Constraint |
| --- | --- | --- |
| `Eq(a, b)` in `equalities` | `a - b` | `a - b = 0` |
| `Le(a, b)` in `inequalities` | `a - b` | `a - b <= 0` |
| `Ge(a, b)` in `inequalities` | `b - a` | `b - a <= 0` |
| A plain SymPy expression `e` | `e` | `e = 0` in `equalities`, `e <= 0` in `inequalities` |

```python
import sympy as sp
from nwqlib import ConstrainedOptimization

x, y = sp.symbols("x y")
problem = ConstrainedOptimization(
    objective=(x - 1) ** 2 + y**2,
    variables=(x, y),
    bounds=((-2, 2), (-2, 2)),
    equalities=(sp.Eq(x, y),),
    inequalities=(sp.Ge(x + y, 1),),
)
print(problem.equalities, problem.inequalities)  # (x - y,) (-x - y + 1,)
```

Strict relations are rejected, because `g(x) <= 0` cannot express strictness, so state a margin explicitly. `Ne`, relations that SymPy has already evaluated to True or False, matrices and constraints with symbols outside `variables` are also rejected.

ConstrainedOptimization is not an Optimization, so a box-only Method such as QHD refuses it at planning instead of ignoring its constraints. The QHD family solves it with `solve_augmented_lagrangian`, a sequence of QHD box problems ([QHD guide](algorithms/qhd.md#constrained-problems)).
