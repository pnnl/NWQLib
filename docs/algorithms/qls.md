# QLS

<a id="quantum-linear-systems"></a>QLS (quantum linear system solver) approximates the solution $x=A^{-1}b$ of a linear system by quantum singular value transformation (QSVT) of the block-encoded matrix $A/\alpha$. The default solver `qsvt_inverse` applies an odd polynomial $P$ that approximates the inverse on the polynomial domain,

```math
\bigl|\kappa_{\mathrm{poly}}\,y\,P(y)-1\bigr|\le\epsilon_{\mathrm{inv}}\quad\text{for }1/\kappa_{\mathrm{poly}}\le|y|\le1,
```

and rescales the result to the physical solution (Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1, Theorem 17 and Corollary 18). The polynomial is NWQLib's own Chebyshev fit with a proven relative residual bound ([Inverse polynomial accuracy](#inverse-polynomial-accuracy)). Two shortcut solvers return only the solution direction, following Dalzell, arXiv:2406.12086v2.

`LinearSystem(A=A, b=b)` describes the original equation $Ax=b$. `solve(problem, method=QLS())` runs `qsvt_inverse` on the local Aer simulator and returns the physical `result.x` in the original coordinates, including scale and phase. `QLS()` uses `epsilon_inv=0.01` to choose its inverse polynomial. This construction target does not bound the total physical error. For the time evolution of a linear ODE, use [LCHS](lchs.md).

## Solve a linear system {#solve-a-linear-system}

```python
from nwqlib import LinearSystem, solve
from nwqlib.algorithms.qls import QLS

problem = LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25])
result = solve(problem, method=QLS(), seed=7)
print(result)
x = result.x
```

```text
Physical solution: [0.89659603+0j, 0.18131412+0j]; shape=(2,), physical, phase=physical
Array acquisition: quantum circuit
QLS solver: qsvt_inverse; selected finite polynomial model
Method: qls; execution: quantum
first target: aer_statevector; data: 1/1 chunks used; 1 attempts
accuracy not assessed
```

The two eigenvalues are `1 ± sqrt(2)/10`, and the exact solution is `(25/28, 5/28)` ≈ `(0.892857, 0.178571)`. This small system exercises both eigenspaces and a non-unit physical solution norm. The computed vector approximates that solution without running an exact solve for comparison. Full amplitudes are a simulator readout, not a hardware measurement. Replace `A` and `b` to solve your own system. The output above was printed with Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, Qiskit 2.5.2 and Aer 0.17.2 on macOS arm64.

Select `execution="classical"` to evaluate the planned polynomial classically instead of running the quantum circuit. The linear system notebook (`examples/qls_linear_system_intro.ipynb`) solves a discretized steady heat equation and reports its physical solution, condition number, polynomial degree, success probability and resources. It also estimates a quadratic form of the solution from finite shots, compares the unit-direction shortcut for several norm guesses and shows how the requested accuracy sets the CX count.

The QLS scientific notebook (`examples/qls_scientific.ipynb`) builds a 15-coordinate history system from single-node D1Q3 collision. Running its cells performs one Dalzell QLS solve (arXiv:2406.12086v2) on ten qubits. The example notebooks simulate at most 12 qubits, and the local simulator accepts 20 by default. Its supplied norm guess `t=4` is a pre-run estimate of the encoded solution norm, whose value is 4.18, and the computed unit direction is compared with an Euler reference and a no-evolution baseline. Its canonical source is `examples/generators/qls_scientific.py`.

## Inspect the plan before running {#inspect-the-plan-before-running}

```python
from nwqlib import plan, estimate

selected = plan(problem, method=QLS(), seed=7)
resources = estimate(selected)
print(selected.reconstruction.degree)
print(selected.reconstruction.alpha, selected.reconstruction.kappa_be)
```

```text
5
1.1414213562373097 1.3294313392598154
```

`plan(...)` returns a `Plan`, the chosen construction and its costs, computed before any circuit exists. Planning chooses the polynomial and phase table once, under the Method's numerical limits. It does not run a reference solve, build circuits or submit them. `solve(selected)` executes that `Plan` without choosing again. `alpha` is the encoding scale, and `kappa_be` is that scale divided by the smallest singular value when that value is known.

With `ResourceContext(basis="cx")`, the Hermitian inverse route can report a CX estimate. The dilation of a non-Hermitian A and the shortcut circuits contain controlled operations without a CX formula, so their CX total is reported as unavailable ([Circuit construction and resource counts](#actual-construction-and-resource-meaning)).

`prepare(selected).circuits` returns copies of the built Qiskit circuits for explicit inspection. Preparing builds them, while resource estimation alone never compiles. The Result keeps its `Plan` and run data. `result.save(path)` and `load_result(path)` restore the planned data and observations, and `result.analyze()` analyzes those observations again. Loading does not run phase search, eigensystems, polynomial selection or a reference solve.

## Inputs, scale and padding

`block_encoding_implementation` selects one explicit encoding family, `"pauli_lcu"`, `"multiplexed_pauli"`, `"banded"` or `"dense_dilation"`, which the input must support. Its default `"auto"` follows the routing below. See the [QLS API](../api/algorithms/qls.md) for the solver and work-limit fields.

| Input or route | What fits the defaults | Default limit |
| --- | --- | --- |
| Dense $A$ whose query is a controlled dense dilation (the Hermitian dilation of a non-Hermitian inverse, and both shortcuts) | At most 64 padded coordinates | `max_work=1_000_000_000` and `max_bytes=10_000_000_000`, which count the exact synthesis of the dilation |
| Inverse polynomial | Degree at most 256 | `max_degree=256` |
| Phase solving | At most 20,000 residual evaluations | `max_qsp_evaluations=20000` |
| Quantum execution on the local simulator | 20 qubits | the simulator's default limit |
| Periodic stencil $A$ with a product-state $b$ | `plan` and `estimate` at $2^{40}$ unknowns, 44 qubits ([example](#plan-and-estimate-beyond-simulation)) | stored compactly, no dense arrays |

`max_work` caps each planning phase separately, for example spectral selection, dense completion, encoding construction, the polynomial fit and the comparisons of sampled Pauli grouping, so no phase counts the work of another. The exact projected reductions of one run count together against `max_work`. These are limits on counted work and peak live arrays, not elapsed-time or memory (RSS) guarantees. [Planning limits](#planning-limits) gives the work each stage counts.

Accepted forms of $A$:

- **Dense array.** QLS uses the dense block-encoding constructor that planning chooses. Automatic routing chooses between banded, Pauli and dense encodings, each with its own normalization and cost meaning. Dense-to-Pauli selection may use the current Qiskit numerical routine.
- **Pauli operator.** QLS computes no singular values of a Pauli $A$, so it needs `QLS(kappa=K)` with $K\ge\alpha/\sigma_{\min}(A)$, where $\alpha$ is the Pauli coefficient one-norm. Classical execution needs explicit dense access. To run a Pauli $A$ classically, pass its dense matrix (for a Qiskit `SparsePauliOp`, `A.to_matrix()`), which must fit within `max_bytes`.
- **Supplied encoding.** Pass a `SelectedBlock` block encoding of the original $A$, with its declared operator, as `QLS(encoding=...)`. Its projected equation and error are assumptions you supply, not a hidden dense test. QLS computes no singular values of a non-dense $A$ with a supplied encoding, so it also needs `QLS(kappa=K)` with $K\ge\alpha/\sigma_{\min}(A)$, where $\alpha$ is the normalization of the supplied encoding.
- **Compact periodic stencil.** See the periodic paragraph below.
- **Sparse matrix.** QLS has no CSR or CSC route. A SciPy sparse $A$ whose dense copy fits in memory can be passed as `A.toarray()`, and the dense route then applies the Method's `max_bytes` and `max_work` limits to it. Otherwise quantum QLS needs Pauli access, the compact periodic stencil or a supplied encoding.

The encoded inverse-gap parameter `kappa_be` is the ratio $\alpha/\sigma_{\min}(A)$ or an accepted user bound $K$ on that ratio, where $\alpha$ is the encoding normalization and $\sigma_{\min}(A)$ the smallest singular value of the original matrix. It differs from the original condition number $\sigma_{\max}(A)/\sigma_{\min}(A)$, where $\sigma_{\max}(A)$ is the largest singular value, whenever the encoding normalization or the supplied bound makes `kappa_be` larger.

| Field | Meaning |
| --- | --- |
| `alpha` | Normalization of the chosen encoding |
| `kappa_be` | Encoded inverse-gap parameter, computed as $\max(1,\alpha/\sigma_{\min}(A))$ for automatic dense spectral selection, or the accepted user bound. Compact periodic selection uses the analytic endpoints described below. |
| `condition_number` | Original $\sigma_{\max}/\sigma_{\min}$ |
| `polynomial_kappa` | Separate polynomial domain parameter, at least 1.01 |

None of the first three includes the polynomial-domain floor. A perfectly conditioned $A$ therefore reports condition 1 and polynomial domain 1.01. Physical recovery uses the polynomial's own parameter:

```math
 x_{\mathrm{selected}}=
 \|b\|\frac{\kappa_{\mathrm{poly}}s}{\alpha}
 \Bigl(\frac{P}{s}\Bigr)(A/\alpha)\frac{b}{\|b\|}.
```

Here $P$ is the stored inverse fit and $s$ the positive rescale chosen to put $P/s$ within the QSP amplitude bound $\lvert P(y)/s\rvert\le1$ for every $y\in[-1,1]$. Planning sets $s=(1+10^{-3})B_P$, where $B_P$ is the Chebyshev norming bound on $\sup_{y\in[-1,1]}|P(y)|$, giving a strict interior margin in exact arithmetic (Martyn et al., arXiv:2105.02859v5, App. A.1, Theorem 10, condition (iii)). The bound is evaluated in binary64, not interval arithmetic. For Hermitian $A$, the success branch of the circuit, or of the classical model, carries `(P/s)(A/alpha) b/||b||`. General input uses the lower block of the Hermitian-dilation action described below. In both cases, recovery multiplies that computed vector by `||b|| kappa_poly s / alpha`. The implementation composes scales in binary mantissa and exponent form. Overflow or loss of nonzero components makes the affected physical quantity unavailable. A valid unit direction can remain available independently.

When the original singular endpoints are computed, their comparisons with the chosen `alpha` and `kappa` allow a relative numerical window of `1e-12`, with no absolute floor. This prevents roundoff at a unitary endpoint from rejecting `alpha=1`. The original endpoint estimates and the chosen normalization stay recorded. Automatic `kappa` is at least one, its mathematical minimum. Larger deficits in the assumed `alpha` or `kappa` are rejected. This window is not a proven spectral enclosure or an added physical-error budget.

General or non-power-of-two dense $A$ uses the dense dilation with explicit normalization. Quantum selection computes the original full SVD once, keeps its left and right singular vectors and conjugation, and pads $A$ with the known positive block $\alpha I$, while $b$ gets zero dummy entries. Dense completion appends analytic dummy singular vectors to that SVD. It does not decompose the padded matrix or solve a padded reference to obtain a scale. The output excludes all dummy coordinates. Scalar dimension 1 receives the same positive extension.

Hermiticity uses exact equality of the supplied entries, and a near-Hermitian matrix uses the general route. For a general inverse, QLS forms

```math
 H=\begin{pmatrix}0&A\\A^\dagger&0\end{pmatrix},\qquad
 b_H=(b,0),
```

and takes the second half after the polynomial action. The Hermitian dummy extension and this embedding do not change the original Problem.

Compact periodic $A=mI+\nu(2I-S-S^\dagger)$, where $m$ is the mass, $\nu\ge0$ is the diffusion coefficient, $I$ is the identity and $S$ is the cyclic shift on the periodic grid, requires positive mass and zero potential. For one qubit, the banded encoding merges the wrapped edges. The original endpoints stay $m$ and $m+4\nu$, while exact rational arithmetic bounds the rounding of the stored coefficients and normalization outward ([Proposition 20](../mathematics.md#r20)). `polynomial_kappa` also covers the rounded encoding gap. Planning stays compact for wide inputs given only as metadata. This does not permit or claim simulation at those widths.

Automatic dense-to-Pauli decomposition keeps every nonzero coefficient, and automatic dense circulant detection requires exact structure. Tiny terms can increase the displayed term count or change the chosen encoding and cost. `alpha` and the encoded inverse-gap parameter `kappa_be` describe the same operator in the classical polynomial model and in the quantum query. Both classical and quantum branch masses are probabilities, and physical solution scaling is applied separately.

### Inverse polynomial accuracy {#inverse-polynomial-accuracy}

For the odd inverse polynomial $P$ of degree at most $d$, the fit bounds the magnitude of the residual $\kappa_{\mathrm{poly}}yP(y)-1$ on $[1/\kappa_{\mathrm{poly}},1]$ using an affine Chebyshev grid and the shared polynomial norming inequality, with $\kappa_{\mathrm{poly}}$ and $y$ as in the opening polynomial-domain condition. The residual has degree at most $d+1$, and its even symmetry covers the negative interval. The recorded `certificate` is this bound, rather than the uninflated maximum of uniform samples. It is evaluated in binary64, not interval arithmetic. Errors in the spectral assumptions and in the quantum phases remain separate from this polynomial residual bound.

The grid nodes matter. Ehlich and Zeller's Satz 2 (doi:10.1007/BF01111276), Eqs. (12)–(14), proves the factor `sec(pi*d/(2*N))` for Chebyshev zeros, and the implementation uses that theorem ([Proposition 22](../mathematics.md#r22)). Their Satz 1 treats an equidistant x grid separately. Sünderhauf et al. arXiv:2507.15537v1, Eq. (25), states the Chebyshev factor for equidistant x points, where it fails. For example, at degree 20 and N=500, a linear program in the Chebyshev coefficients gives a polynomial with absolute value at most 1 at every point `x_j=-1+2*j/499`, yet value greater than 1.0254 at x=.999, while the claimed factor is less than 1.002. Converting the linear-program coefficients to exact rational numbers, dividing by their exact grid maximum and evaluating the Chebyshev recurrence at all 500 nodes and at .999 confirms this counterexample without relying on solver tolerances. Proposition 22 gives a shorter exact counterexample of degree one.

Because the residual is relative, it bounds the polynomial step's solution error component by component. When every eigenvalue (or, through the dilation, singular value) of `A/alpha` lies in the polynomial domain, the transformed vector `y` satisfies `||y - (kappa_poly A/alpha)^-1 b|| <= epsilon_inv ||(kappa_poly A/alpha)^-1 b||`.

## Error budget {#error-budget}

For the first example, the relative error budget is the polynomial residual tolerance 0.01 plus a phase-fit term of about 1e-15, and the measured relative error of the solution is 0.0051. QLS forms no combined output-error bound, so `result.assess(absolute_tolerance=.01)` is INCONCLUSIVE, with the sources `algorithmic_approximation`, `floating_point` and `native_execution` unknown ([Check against a tolerance](../verification.md#check-against-a-tolerance)).

Continuing the first example:

```python
from nwqlib.algorithms.qls import QLSVerification

reconstruction = result.plan.reconstruction
print(reconstruction.polynomial.certificate, reconstruction.phase_error)
receipt, facts = result.verify(
    checks=QLSVerification(comparisons=("inverse_relative_error",)))
for item in receipt.applications[0].facts:
    print(item.fact.quantity, item.fact.value.value)
```

```text
0.00625690026574319 8.88364227200309e-16
reference.inverse_relative_error.raw_relative_error 0.005092570207648435
reference.inverse_relative_error.method_allowance 0.010000000000001272
reference.inverse_relative_error 0.5092570207647787
```

The first line gives the residual bound of the chosen polynomial and the phase-fit residual bound $\delta$. `method_allowance` is the relative error budget of [Explicit verification and evidence](#explicit-verification-and-evidence), and the last line is the measured error divided by it. The output above was printed with Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, Qiskit 2.5.2 and Aer 0.17.2 on an Apple M3 Max (macOS arm64), after the quantum solve of the first example on the default local Aer statevector.

| Quantity | Value | How it is obtained |
| --- | --- | --- |
| Inverse polynomial | Odd Chebyshev fit of degree 5, solver `qsvt_inverse` | Default `QLS()` |
| Encoding normalization $\alpha$ | 1.14142 | Normalization of the dense encoding chosen at planning |
| Encoded inverse-gap parameter `kappa_be` and polynomial domain `polynomial_kappa` | 1.32943 and 1.32943 | $\alpha/\sigma_{\min}(A)$, from the singular values of $A$ at planning |
| Residual tolerance `epsilon_inv` | 0.01 | Default |
| Residual bound of the chosen polynomial, `certificate` | 0.00625690 | Bound on $\lvert\kappa_{\rm poly}yP(y)-1\rvert$ for $1/\kappa_{\rm poly}\le\lvert y\rvert\le1$ from Chebyshev samples, evaluated in binary64 ([Inverse polynomial accuracy](#inverse-polynomial-accuracy)) |
| Phase-fit residual bound $\delta$, `phase_error` | 8.9e-16 | Recorded with the QSP phases of the circuit |
| Relative error budget, `method_allowance` | 0.01 + 1.3e-15 | $\epsilon_{\rm inv}+\kappa_{\rm poly}s\delta/\lVert y_{\rm ref}\rVert$, with $s$ the rescale and $y_{\rm ref}=(A/\alpha)^{-1}b/\lVert b\rVert$. It requires every eigenvalue of $A/\alpha$, or every singular value for a non-Hermitian $A$, to lie in the polynomial domain |
| Measured relative error, `raw_relative_error` | 0.00509257 | $\lVert x-A^{-1}b\rVert/\lVert A^{-1}b\rVert$ against the dense solve of `QLSVerification`, with $A^{-1}b=(25/28,5/28)$. It also contains the rounding in the simulated circuit, which the budget excludes |
| Error sources `algorithmic_approximation`, `floating_point` and `native_execution` | Unknown | Listed by the Result, with no combined output-error bound |

A smaller `epsilon_inv` raises the degree and the CX count, while the phase-fit bound stays below 1e-14. For the compact ring of [Plan and estimate beyond simulation](#plan-and-estimate-beyond-simulation) at q = 40, with 44 qubits, planning and resource estimation give:

| `epsilon_inv` | Degree | Residual bound of the polynomial | Phase-fit bound $\delta$ | CX estimate per attempt |
| --- | --- | --- | --- | --- |
| 0.1 | 17 | 0.0749587 | 2.7e-15 | 58,154 |
| 0.01 | 27 | 0.00990748 | 4.7e-15 | 92,354 |
| 0.001 | 39 | 0.000869822 | 4.6e-15 | 133,394 |
| 0.0001 | 51 | 0.0000763610 | 8.9e-15 | 174,434 |

At q = 10 and 20 the degree and both bounds are the same, and only the qubit and CX counts change. The CX estimates use Qiskit 2.5.2's multi-controlled X counts, so another Qiskit version can give other counts. `docs/scripts/error_budget_figures.py` prints every value of this section.

## Methods and outputs

For the shortcuts, let $B$ be $A/\alpha$ after input padding, let $\hat b$ be the corresponding normalized, zero-padded right-hand side, and let $e$ be the unit vector at the first added coordinate when this space is doubled. Then $A_t=(B\oplus0)+ee^\dagger/t$, $b'=((\hat b\oplus0)+e)/\sqrt2$ and $G_t=(I-b'b'^\dagger)A_t$, where $t$ is the encoded solution-norm estimate, the direct sums add a zero block of the same dimension and $I$ is the identity on the doubled space (Dalzell, arXiv:2406.12086v2, Eqs. (8)–(11) and App. A.6).

| Configuration | Action | Available output |
| --- | --- | --- |
| `QLS()` | Odd inverse Chebyshev polynomial with its stored norming bound, applied by a QSVT circuit of phases, projectors and queries | Physical `Solution`, physical or unit `StateVector`, `NormSquared`, `QuadraticForm`, `NormalizedExpectation`, `Samples` |
| `QLS(solver="shortcut_native_svp", encoded_solution_norm_estimate=t)` | Dalzell kernel reflection on the right singular vectors of $G_t$ | Unit `StateVector` modulo global phase, `NormalizedExpectation`, `Samples` |
| `QLS(solver="shortcut_dilation", encoded_solution_norm_estimate=t)` | Even polynomial on the Hermitian dilation of $G_t$ | Same unit-only outputs |

For the small problem above, an explicitly supplied `t=1.2` gives a valid unit-direction request:

```python
from nwqlib import StateVector

shortcut = QLS(solver="shortcut_native_svp",
               encoded_solution_norm_estimate=1.2)
direction = StateVector(normalization="unit",
                        global_phase="modulo_global_phase")
direction_result = solve(problem, method=shortcut, output=direction, seed=7)
print(direction_result)
```

```text
State vector: [0.9805597+6.4235962e-15j, 0.19622099+7.6448198e-16j]; shape=(2,), unit, phase=modulo_global_phase; unit=1
Normalized direction; physical solution magnitude is not supplied by these amplitudes
Array acquisition: quantum circuit
QLS solver: shortcut_native_svp; selected finite polynomial model
shortcut supplies no physical reconstruction scale
Method: qls; execution: quantum
first target: aer_statevector; data: 1/1 chunks used; 1 attempts
accuracy not assessed
```

The exact unit direction is $(5,1)/\sqrt{26}\approx(0.980581,0.196116)$. That `t` is specific to this example and does not provide a physical solution magnitude.

Shortcut methods need an explicit numeric `1 <= t <= polynomial_kappa` for quantum execution. Classical execution also accepts `"grid"`, `"noisy_binary_search"` and `"linear_kappa_sequence"` as values of `encoded_solution_norm_estimate`, not of `solver`. These compute the classical probability model and report its solves, decompositions and search rows. Planning checks each model's search rows, trials and queries from `polynomial_kappa` against the limits before the model runs, with the same exact ladder length `ceil(log2(polynomial_kappa))` that the model uses. Planned trials and queries are model accounting, not executed quantum jobs. The linear model evaluates its recurrence from the same original factorization, obtained at planning or, when supplied assumptions avoid spectral selection, once by the evaluation. Grid and noisy search share their one target solve, and their application records the resulting encoded reference norm. No shortcut uses a reference to reconstruct a missing physical magnitude or physical global phase.

`execution="classical"` explicitly evaluates the planned inverse or shortcut numerical model. It does not claim quantum execution. `Samples` requires quantum execution with a positive `shots`. Vector outputs require exact amplitude readout.

With `shots=None`, a scalar observable is evaluated exactly from one simulation of the circuit, with no finite-shot error. Circuit, observable-action and normalization rounding remain separate error contributions. With finite `shots`, Pauli terms that commute qubit-wise share a measurement basis, and `shots` is the number of shots per group. Normalized outputs are unavailable at zero physical mass. [Readout](#readout) gives the details, and [readout tolerance branches](../error_evidence.md#readout-tolerance-branches) explains how the masses of an exact readout are checked.

For `Samples`, returned shots count all shots of the measurement. `samples` stores only the outcomes in the success-and-physical subset, as increasing int64 original-coordinate indices and their positive int64 counts. Read `algorithm_selected_shots` (shots that satisfy algorithmic success) and `physical_selected_shots` (shots that also satisfy the conditions and lie in original coordinates) together with the displayed empirical selection masses. A modeled success probability is a different quantity and cannot replace these observed counts.

## Circuit construction and resource counts {#actual-construction-and-resource-meaning}

One circuit description, a `Program` (NWQLib's description of a circuit as named steps), holds the state preparation, the forward and adjoint queries of the original $A$, valued controls, global phase, every converted RZ phase, projector predicates, the Dalzell augmentation and projector bodies, and readout. Each query keeps the exact meaning of its base encoding. For a square matrix $M$, write its Hermitian dilation as $H(M)=\begin{pmatrix}0&M\\M^\dagger&0\end{pmatrix}$, so $H(A)$ is the matrix $H$ defined under [Inputs, scale and padding](#inputs-scale-and-padding), and $G_t$ is defined under [Methods and outputs](#methods-and-outputs). The circuit descriptions of $H(A)$ and $H(G_t)$ contain their controlled child operations. Inspection, resource estimation and circuit building all use that same description, including its arguments and ports, so no step treats QLS as one opaque block or plans it again.

Known per-query CX formulas remain estimates, including the per-family formulas that a caller asserts for its own encoding. A multi-controlled X with `2 <= k <= 64` controls, such as a projector flip or a `Q_b'` or `A_t` predicate, is counted at the CX count of Qiskit 2.5.2's synthesis of that gate (`MCX_CX_BY_CONTROLS`, registered under [circuit-free synthesis laws](../ENGINEERING_CONSTANTS.md#circuit-free-synthesis-laws)). The count is an estimate because another Qiskit version can synthesize the gate differently, and a gate with more controls has no CX formula. One-qubit primitives and singly controlled X gates are exact at their leaf. Other controlled primitives, controlled queries, vendor workspace and hardware routing have no CX formula and remain unknown. The shortcut circuits and the dilation of a non-Hermitian inverse contain such controlled operations, so their logical CX total is unknown, and only the Hermitian inverse route can report a concrete total. Resource estimation supplies the multiplicities of each block. Explicit inspection of a prepared circuit counts that circuit separately.

Select `ResourceContext(basis="cx")` when asking `estimate` for these CX formulas. [Resource estimation](../resources.md) shows the call. This does not prepare or transpile a circuit.

## Plan and estimate beyond simulation

A periodic stencil for $A$ and a product state for $b$ are stored in a compact form that grows at most linearly with the number of qubits, so `plan` and `estimate` work at sizes that no simulator holds. The example below is compact planning and resource estimation only. Nothing is solved or simulated at these sizes, and no dense matrix or full state vector is formed.

```python
import numpy as np
from nwqlib import LinearSystem, estimate, plan
from nwqlib.algorithms import QLS
from nwqlib.operators import PeriodicStencil, operator_input
from nwqlib.problems import ingest_product
from nwqlib.resources import ResourceContext

q = 40  # 2**40 unknowns on a ring
# A = 0.1 I + 0.1 (2I - S - S†)
A = operator_input(PeriodicStencil(q, mass=0.1, diffusion=0.1))
# Row j holds the two amplitudes of qubit j.
b = ingest_product([[np.cos(0.3 + 0.1 * j), np.sin(0.3 + 0.1 * j)]
                    for j in range(q)])
selected = plan(LinearSystem(A=A, b=b), method=QLS(epsilon_inv=0.01), seed=7)
workload = estimate(selected, context=ResourceContext(basis="cx"))
print(selected.reconstruction.kappa_be, selected.reconstruction.degree)
width = workload.quantity("logical_width", location="logical_device")
print(width.fact.value.numerator, workload.quantity("cx").fact.value.value)
```

```text
5.0 27
44 92354.0
```

| Unknowns | Encoded inverse-gap parameter `kappa_be` | Polynomial degree | Qubits | CX gates, estimate |
| --- | --- | --- | --- | --- |
| `2**10` | 5 | 27 | 14 | 7,304 |
| `2**20` | 5 | 27 | 24 | 24,854 |
| `2**40` | 5 | 27 | 44 | 92,354 |

The rows set `q` to 10, 20 and 40 in the example above. The CX counts use Qiskit 2.5.2's multi-controlled X counts, so another Qiskit version can give other counts. Each plan with its estimate took under 0.2 s after the imports, with Python 3.12.14, NumPy 2.5.2 and Qiskit 2.5.2 on an Apple M3 Max (macOS arm64). With mass $m=0.1$ and diffusion coefficient $\nu=0.1$, the spectrum of $A$ lies in $[m,m+4\nu]=[0.1,0.5]$ at every size, and the chosen normalization is `alpha = 0.5`, so the encoded inverse-gap parameter `kappa_be` and the polynomial degree stay fixed while the width grows with q. The $b$ chosen here is not an eigenvector of the ring, so $x$ is not proportional to $b$. The CX counts assume this product-state $b$. A general $b$ with `2**40` amplitudes costs far more to prepare and cannot be stored as an array.

## Explicit verification and evidence

```python
from nwqlib.algorithms.qls import QLSVerification
receipt, facts = result.verify(checks=QLSVerification(
    comparisons=("spectral_domain", "inverse_relative_error", "inverse_success")
))
```

A set of comparisons shares one solve of the original `A/alpha` with the original normalized $b$. It reuses known original singular endpoints, and a spectral-only set does no solve. A set whose comparisons need only the norm of that solution, such as `eq17`, reuses the encoded reference norm that a grid or noisy-search norm model recorded from its own solve. That norm is the model's own intermediate value, not independent evidence, and the reference description of each such check states this. A missing original spectrum is computed only when that comparison is explicitly requested. Vector access, method compatibility and finite `max_work` and `max_bytes` are checked first. Physical inverse error keeps sign and phase. The shortcut direction allows only its declared comparison modulo global phase. The verification record keeps the call counts and every raw fact, and `facts` holds the top-level fact of each requested comparison.

Obtaining a new spectral or solution reference requires original dense input access. Compact Pauli or sparse input is not densified implicitly, and unsupported requests receive an explanation from QLS. A spectral-only comparison can reuse already computed endpoints without dense access or a solve. A new `Plan` from dense input is the explicit alternative when its construction and reference costs are intended.

Without user tolerances, the inverse error is compared with the stated polynomial and phase error budget. That relative budget is `epsilon_inv + kappa_poly * s * delta / ||y_ref||`, with `delta` the phase-fit residual bound (zero for `execution="classical"`), `s` the rescale and `y_ref = (A/alpha)^-1 b/||b||` the reference solution of the verification solve, as derived in the `_comparisons` docstring of `algorithms/qls/verification.py`. The shortcut's default `5*epsilon_inv` direction window and the mass windows are heuristic. The Eq. 17 phase error budget (Dalzell arXiv:2406.12086v2) widens both endpoints while keeping the original raw discrepancy. The binomial term uses the predicted reference rate and the returned shots of the mass measurement. An out-of-range reference prediction is not forced into a Bernoulli model. Missing phase or spectral evidence leaves dependent error budgets unknown and keeps independent raw quantities. These checks do not bound the total physical error or establish confidence coverage.

Without an explicit relative tolerance, the top-level `inverse_relative_error` comparison uses the metric `relative_error_over_method_allowance`, the raw relative discrepancy divided by the error budget. A value such as 0.509, printed in [Error budget](#error-budget), means 0.509 of that budget, not 50.9% physical inverse error. The verification record's `.raw_relative_error` fact keeps the relative discrepancy itself. Display that fact's scalar value and the check metric together, instead of the repr of the whole nested verification record.

Each requested comparison is one check of `QLSVerification.verification_checks`. Its threshold is the comparison's explicit tolerance, or `spectral_tolerance` for `spectral_domain`, or `5*epsilon_inv` for a shortcut direction without `direction_tolerance`. An inverse or mass comparison without its tolerance compares the ratio to its error budget with 1. [Check accuracy and verify a result](../verification.md#evidence-and-receipts) shows how `Certificate.with_verification` turns `facts` into a PASS or FAIL for each check, or INCONCLUSIVE when a value is unavailable. A verdict against the default direction window or a heuristic mass window remains heuristic.

NWQLib's tests check the inverse fit against the reference polynomial of CKS (arXiv:1511.02306v2), keep fixed reference values for the kernel-reflection, right-singular-vector and search arithmetic, and check the quantum circuits against independent polynomial recurrences. They also cover physical phase and scale, padding of a noncommuting complex 3×3 system, rejection of a stale SVD or input of the same shape, valid cache reuse, the query and control counts, the shortcut G_t query block with one controlled synthesis per control value and adjoint flag, one exact reduction against an independent slice of the circuit state, grouped count settings, the check of the complete norm against the preparation record, rejection of masses outside their valid range or nesting, and saved results that load without choosing again. The [examples guide](../examples.md) lists the QLS notebooks.

## Limits

- `epsilon_inv` sets the residual tolerance of the inverse polynomial. It does not bound the total physical error, and QLS forms no combined output-error bound ([Error budget](#error-budget)).
- The shortcut solvers return only the unit solution direction, and quantum execution needs an explicit numeric `1 <= t <= polynomial_kappa` ([Methods and outputs](#methods-and-outputs)).
- QLS computes no singular values of a Pauli $A$ or of a non-dense $A$ with a supplied encoding, so these inputs need `QLS(kappa=K)` with $K\ge\alpha/\sigma_{\min}(A)$. QLS has no CSR or CSC route ([Inputs, scale and padding](#inputs-scale-and-padding)).
- Only the Hermitian inverse route can report a concrete CX total ([Circuit construction and resource counts](#actual-construction-and-resource-meaning)).
- The shortcut's default `5*epsilon_inv` direction window and the mass windows are heuristic, and the verification checks do not bound the total physical error or establish confidence coverage ([Explicit verification and evidence](#explicit-verification-and-evidence)).
- Compact periodic planning at large widths does not permit or claim simulation at those widths ([Plan and estimate beyond simulation](#plan-and-estimate-beyond-simulation)).

[Limitations and open work](../ROADMAP.md#qls) lists the open items for this method.

## Planning limits and numerical scope {#planning-limits}

`max_admission_steps` (default 1,000,000) limits the work of checking the planned circuit descriptions and does not change the polynomial or the quantum operations. Program validation, preparation and circuit building each apply this limit to their own checks. Passing one stage does not establish that a later stage fits ([planning work limit](../development/program_checks.md#planning-work-limit)).

### Fit and dense-dilation work {#fit-and-dense-dilation-work}

Each fit candidate's bytes and work are checked against the limits before it runs, and each phase's work accumulates across attempts. A dense-dilation query has valued controls in the Hermitian dilation of a non-Hermitian inverse and in both shortcuts, and its base dilation unitary acts on $\log_2(p)+1$ qubits for padded dimension $p$. The forward and adjoint versions each require one exact synthesis per Run. `dense_control_route` chooses between synthesizing the controlled matrix on $\log_2(p)+1+c$ qubits for $c$ controls and synthesizing the base unitary on $\log_2(p)+1$ qubits before Qiskit controls each synthesized gate. Planning counts both syntheses, with Qiskit's control of them on the gate-wise route, for one control or for two in `shortcut_dilation`, against `max_work` and `max_bytes` ([exact synthesis check](../development/dense_synthesis.md#admission-of-the-exact-synthesis)). It does so before the original SVD when the dense dilation is requested or forced, and after encoding selection when automatic selection or a supplied encoding gives one.

`dense_control_route` selects how a controlled query controls a planned dense dilation, `"whole_matrix"` (synthesize the controlled dilation), `"gatewise"` (let Qiskit control each synthesized gate) or `"auto"`. The default `"auto"` synthesizes the forward and the adjoint controlled dilation for the one control of the Hermitian dilation and of `shortcut_native_svp`, and keeps the gate-wise route for the two controls of `shortcut_dilation` ([dense control route](../development/dense_synthesis.md#dense-control-route)). A supplied encoding circuit (`encoding.native`) in a query with valued controls is controlled gate-wise on every route and counted for the dense unitaries it holds, and so is a supplied right-hand-side circuit, which `shortcut_native_svp` controls in two specializations and `shortcut_dilation` in four. The Hermitian inverse queries the dilation without controls, so its construction synthesizes nothing, and a backend that compiles the circuit to a gate basis counts the synthesis against the run's `max_synthesis_work`.

The original SVD arrays and phase tables are saved once. Saved runs also keep the query gates already built, in QPY storage. The [engineering constants registry](../ENGINEERING_CONSTANTS.md#qls-construction-and-readout-limits) records these defaults with their revisit condition.

### Spectral data and classical evaluation {#spectral-data-and-classical-evaluation}

Classical dense QLS reuses one original spectral factorization when it needs both spectral endpoints and polynomial or norm-model data. A general inverse uses the original SVD to apply the odd polynomial in the correct singular vectors. Hermitian input uses signed eigenvalues. Positive dummy padding needs no additional decomposition. The shortcut's augmented projected matrix has its own spectral action and is counted separately. Numerical singular endpoints remain estimates, not proven spectral bounds.

Classical selection computes only the spectral data it needs, and supplied `alpha` and `kappa` assumptions avoid that spectral work. The inverse and the linear norm model take their endpoints from the original factorization they later reuse. When supplied assumptions avoid spectral selection, the classical evaluation obtains that factorization once and counts it against its own work. A shortcut with a numeric `t` needs endpoints only. Spectral selection first applies a binary change of scale. Hermitian input takes all eigenvalues with `eigvalsh` and uses their magnitudes, so the smallest magnitude of a spectrum that spans zero is included. General matrices use the singular-values-only SVD. No route squares $A$ to infer its gap. The `Plan` records the factorization it used in `spectral_method` and `spectral_calls`, and `QLSWork` counts the decompositions of one classical evaluation. When supplied assumptions avoid obtaining singular endpoints, the Result summary states that they were not independently checked at selection. Dense quantum encoding may already have the endpoints needed to reject a false assumption. Classical or compact selection does not compute an extra spectrum just to match that check.

The classical factor-reuse and rank-one formulas compute the same polynomial model in exact arithmetic. Floating-point results can differ because factorizations, basis changes and reductions use different arithmetic. A numerical comparison depends on the encoded matrix scale, polynomial coefficients and branch mass. Discrete choices can change at numerical thresholds, and normalization of a small success branch can amplify an otherwise small vector discrepancy.

Classical observable reduction combines the observable's binary scale with the vector recovery before converting the requested scalar to binary64. It performs one observable action and counts its storage-preserving copy and scans. If that copy would lose a nonzero entry, it uses the original unscaled action instead, without retrying or pruning. This extends the recoverable range without bounding cancellation accuracy or arbitrary heterogeneous dynamic range. Unrepresentable outputs remain unavailable.

### Readout {#readout}

An exact scalar readout (`shots=None`) fixes the method's success and physical-condition bits and excludes dummy coordinates. A Pauli observable with $L$ terms then needs $O(LN)$ classical work on the $N$-coordinate encoded system. The normalized expectation divides the projected quadratic moment by the physical-slice mass. A physical quadratic form also applies the method's recovery scale.

Exact scalar readout evaluates the stored Pauli observable on the success-and-physical slice $v$, returning $p=v^\dagger v$ and $q=v^\dagger\tilde O v$. The normalized output is $q/p$ for positive $p$. A dense observable is extended by zeros before coefficient conversion. Its conversion error contributes separately to the error against the original dense observable. Projected and full-block Pauli sums coincide for an exact zero-extension representation, while rounded coefficients can change the cancellation of dummy-coordinate contributions. Recovery and numerical reduction errors follow their own bounds.

The `Plan` of an exact scalar output has one experiment, `projected_moments`. It runs the coherent body and reduces the saved state at the body's end, with parameters that bind the success bits, condition bits, coordinate order, original dimension and stored observable. `QLSAnalysis.reduction` saves the complete norm of the simulated state, the success-only mass `p_alg`, the physical-slice mass `p` and the projected moment `q` of that one readout, and `physical_slice_mass` is `p`. The complete, success and physical masses use the scaled squared-norm routine and its classical error budget. The complete mass is checked against the producing preparation record before projection. The success and physical masses satisfy nested-subset comparisons on the same computed state. A binary-scale mass keeps its nonzero magnitude outside the ordinary binary64 output range, and physical recovery scales both the mass and its error budget. Reloading analyzes these saved statistics without a new simulation. [Readout tolerance branches](../error_evidence.md#readout-tolerance-branches) explains which tolerance checks the masses.

Sampled observable readout groups qubit-wise-commuting Pauli terms into shared local bases. `shots` is the number of shots per group. Terms in a group share outcomes, so uncertainty calculations use the weighted group outcome and its covariance. Grouping reduces the number of settings. Its variance at fixed total shots depends on the state, coefficients and allocation. A padded normalized output also measures its physical-coordinate mass in an unrotated basis. A group whose basis has no X or Y supplies that mass, and otherwise a separate unrotated `physical_mass` setting measures it. Each group measures its support coordinates and valued selector bits. `Samples`, `physical_mass` and the unrotated group that supplies a padded prefix mass measure every coordinate. Outcomes use a shared classical layout sized for the largest measured register, with unused suffix bits fixed at zero. `QLSAnalysis.groups` records each group's labels, basis, returned and selected shots and measurement. `mass_contribution_id` identifies the measurement that supplies the branch mass, and its returned shots are the ones the mass comparison uses. A conditional sample requires a positive number of selected shots. Planning requires at most 64 measured bits in every setting, so a group with small support can use a circuit wider than its count register.

## Sources and code map

Equation and section numbers refer to the arXiv versions named in the Source column. [References](../references.md) lists the full citations. Rows marked NWQLib are the library's own constructions, derived in the named docstrings.

| Step | Source | Location | Code |
| --- | --- | --- | --- |
| Wx QSP convention, where zero phases give `T_d` | Martyn et al., arXiv:2105.02859v5 | Sec. II.A, Eqs. (1)-(3), Theorem 1, App. A.1 | `subroutines/qsp/phases.py` `evaluate_qsp_polynomial` |
| Wx to reflection phases | Martyn et al., arXiv:2105.02859v5 | Eq. (14) and App. A.2, Eq. (A5), with `i^d` kept as a global phase | `subroutines/qsp/phases.py` `wx_phases_to_reflection` |
| Symmetric phase solving | Dong et al., arXiv:2002.11649v2 | Sec. III.1-III.5, Eqs. (23), (24), (27)-(28), (30). The larger node set and the polish are NWQLib's. The exact Jacobian and its chain rule onto the symmetric phases are derived in the `_symmetric_problem` docstring. | `subroutines/qsp/phases.py` `solve_symmetric_qsp_phases` |
| Newton start when L-BFGS misses the tolerance | Dong, Lin, Ni and Wang, arXiv:2307.12468v1 | Sec. 2.2, Sec. 3, Eq. (3.1) and Algorithm 3.1. The step-halving safeguard is NWQLib's. The least-squares step on the node residual equals the Newton step on the Chebyshev coefficients, as derived in the `_damped_newton` docstring. | `subroutines/qsp/phases.py` `_damped_newton` |
| Sup norm from Chebyshev samples | Ehlich and Zeller (1964), doi:10.1007/BF01111276 | Satz 2, Eqs. (12)–(14), pp. 42–43. Sünderhauf (arXiv:2507.15537v1) Eq. (25)'s equidistant-x statement is not valid ([Proposition 22](../mathematics.md#r22)) | `subroutines/qsp/phases.py` `chebyshev_polynomial_sup_bound` |
| Projector-controlled phase and QSVT sequence | Martyn et al., arXiv:2105.02859v5, and Gilyén et al., arXiv:1806.01838v1 | Martyn Eq. (27), Fig. 3, Theorems 3-4. Gilyén Theorem 17, Lemma 19 | `subroutines/qsp/evolution.py` `build_qsvt_circuit`, `algorithms/qls/quantum.py` |
| Real part through the `+Phi`/`-Phi` pair | Gilyén et al., arXiv:1806.01838v1 | Corollary 18, Eq. (33) | `subroutines/qsp/evolution.py` `build_real_chebyshev_encoding`, `algorithms/qls/quantum.py` |
| Inverse fit with proven relative residual bound | NWQLib | `subroutines/qsp/inverse.py` module docstring | `subroutines/qsp/inverse.py` `InverseChebyshevFit` |
| Inverse degree law behind the quick upper bound on the degree | Childs, Kothari and Somma, arXiv:1511.02306v2 | Lemmas 17-19, Eqs. (74), (77), (88) | `subroutines/qsp/inverse.py` `inverse_degree_law_bound` |
| Residual-bound grid density `N = 25d` | Sünderhauf et al., arXiv:2507.15537v1 | Sec. III, Eq. (26) | `subroutines/qsp/inverse.py` |
| Reference `1/x` polynomial for tests | Childs, Kothari and Somma, arXiv:1511.02306v2 | Lemmas 17-19 | `tests/_qsp_references.py` |
| Rescale and physical recovery | NWQLib, with the QSP bound of Martyn et al., arXiv:2105.02859v5 | App. A.1, Theorems 9-10, condition (iii) | `algorithms/qls/host_planning.py` `select_polynomial`, `algorithms/qls/method.py` |
| Spectral assumptions, padding and Hermitian dilation | NWQLib | `select_inputs` and `_spectrum` docstrings, and the dilation paragraph of this guide | `algorithms/qls/host_planning.py`, `algorithms/qls/method.py`, `algorithms/qls/quantum.py` |
| Original factor reuse for the inverse and the linear norm model | NWQLib | `inverse_polynomial_action`, `original_factor_laws` and `selected_work` docstrings | `algorithms/qls/numerical.py` `inverse_polynomial_action`, `linear_model_weights`, `algorithms/qls/host_planning.py` `_original_factors`, `selected_work` |
| Kernel-reflection polynomial and `eta` | Dalzell, arXiv:2406.12086v2 | Eqs. (6), (22), App. B.2, Eqs. (51)-(54) and Lemma 1, App. B.3, Eq. (62) and Lemma 3. The closed form `T_ell(z0) = cosh(ell arccosh z0)` and the proof that Eq. (6) gives `F(Delta) <= eta` are in the `_realize_kernel_reflection` docstring. | `subroutines/qsp/shortcut.py` `plan_kernel_reflection`, `dalzell_eta_from_precision` |
| Augmented system `A_t`, `b'`, `G_t` and their circuits | Dalzell, arXiv:2406.12086v2 | Eqs. (8)-(11), App. A.1-A.4 and A.6. With the augmented coordinate on its own qubit, preparing `b'` needs no extra ancilla, unlike App. A.3. | `algorithms/qls/numerical.py` `shortcut_matrix`, `projected_augmented_inplace`, `algorithms/qls/quantum.py` |
| Algorithm 1 (prepare `e_n`, reflect, project) | Dalzell, arXiv:2406.12086v2 | Algorithm 1, p. 5, and App. B.1, Eq. (50) | `algorithms/qls/numerical.py` `shortcut_polynomial_action`, `hermitian_dilation`, `project_shortcut_branch` |
| Success-probability check tolerance | Dalzell, arXiv:2406.12086v2 | Eqs. (7), (17) | `algorithms/qls/norm_search.py`, `algorithms/qls/verification.py` |
| Norm search: grid, noisy binary search, linear-in-kappa sequence | Dalzell, arXiv:2406.12086v2 | Sec. 5.1, Eqs. (23)-(25). Sec. 5.2, Eqs. (26)-(29), (44). Sec. 5.3, Eqs. (31)-(36), (41)-(42) | `algorithms/qls/norm_search.py`, `algorithms/qls/numerical.py` `linear_model_weights` |
| Periodic stencil gap and error accounting | NWQLib ([Proposition 20](../mathematics.md#r20)) | `select_periodic_encoding` docstring | `algorithms/qls/periodic.py` |
| Verification error budgets | NWQLib, with Dalzell arXiv:2406.12086v2, Eq. (17) | `_comparisons` docstring | `algorithms/qls/verification.py` `QLSVerification` |
| CX count of a multi-controlled X | Qiskit 2.5.2 synthesis, tabulated. No paper location | `MCX_CX_BY_CONTROLS` registry entry | `algorithms/qls/quantum.py`, `subroutines/_mcx_counts.py` |

The noisy binary search eliminates candidates above a probe whose predicted success probability exceeds 1/2, the direction implied by Eq. (44). The Sec. 5.2 text of Dalzell arXiv:2406.12086v2 states the opposite direction. Dalzell arXiv:2406.12086v2 Lemma 3, item 2, prints the kernel-reflection bound `-1 + 4 eta/(1 + eta)` for `|K(x)|`, which is negative for `eta < 1/3`. The proof through Eq. (63) and the statement on p. 4 bound `K(x)` itself, `-1 <= K(x) <= -1 + 4 eta/(1 + eta)` on `1/kappa <= x <= 1`. That signed bound is the one behind Eqs. (5) and (17), and so behind the Eq. (17) success window of `QLSVerification`. Costa, Dalzell, An and Berry (arXiv:2604.22185v2) compare the shortcut and adiabatic solvers numerically and find the shortcut better for non-Hermitian matrices when the solution norm is known. Their constants and the discrete-adiabatic and initial-state-query solvers are background here, not implemented routes.
