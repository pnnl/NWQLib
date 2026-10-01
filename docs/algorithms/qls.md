# Quantum linear systems

`LinearSystem(A=A, b=b)` describes the original equation Ax=b. `solve(problem, method=QLS())` executes quantum `qsvt_inverse` on local Aer and returns physical `result.x` in the original coordinates, including scale and phase. `QLS()` uses `epsilon_inv=0.01` to select its inverse polynomial. This construction target does not prove a bound on total physical error.

```python
from nwqlib import LinearSystem, solve
from nwqlib.algorithms.qls import QLS

problem = LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25])
result = solve(problem, method=QLS(), seed=7)
print(result)
x = result.x
```

The two eigenvalues are `1 ± sqrt(2)/10`, and the exact solution is `(25/28, 5/28)`. This small system exercises both eigenspaces and a non-unit physical solution norm. The obtained vector approximates that solution without running an exact solve for comparison. Full amplitudes are a simulator readout, not a hardware measurement. Replace A and b to solve your own system.

When original singular endpoints are computed, their comparisons with selected `alpha` and `kappa` allow a relative numerical window of `1e-12`, with no absolute floor. This prevents roundoff at a unitary endpoint from rejecting `alpha=1`. The original endpoint estimates and selected normalization remain recorded; automatic `kappa` is at least one, its mathematical minimum. Larger premise deficits reject. This premise-check window is not a proven spectral enclosure or an added physical-error budget.

Select `execution="classical"` to evaluate the selected polynomial on the host instead of executing the quantum circuit. The linear system notebook (`examples/qls_linear_system_intro.ipynb`) solves a discretized steady heat equation and reports its physical solution, condition number, polynomial degree, success probability and resources. It also estimates a quadratic form of the solution from finite shots, compares the unit-direction shortcut for several norm guesses and shows how the requested accuracy sets the CX count.

The QLS scientific notebook (`examples/qls_scientific.ipynb`) builds a 15-coordinate history system from single-node D1Q3 collision. Running its cells performs one Dalzell QLS solve (arXiv:2406.12086v2) with ten default qubits and a 12-qubit cap. Its supplied norm guess `t=4` is a pre-run estimate of the encoded solution norm, whose actual value is 4.18, and the obtained unit direction is compared with an Euler reference and a no-evolution baseline. Its canonical source is `examples/generators/qls_scientific.py`.

To inspect the selection before execution, use the same planning path:

```python
from nwqlib import plan, estimate

selected = plan(problem, method=QLS(), seed=7)
resources = estimate(selected)
print(selected.reconstruction.degree)
print(selected.reconstruction.alpha, selected.reconstruction.kappa_be)
```

Planning selects the actual polynomial and phase table once under the Method's numerical limits. It does not run a reference solve, construct native circuits or submit circuits. `plan(...)` returns a `Plan`, the selected construction and its costs, computed before any circuit exists. `solve(selected)` executes that `Plan` without repeating selection. The selected `alpha` is the encoding scale, and `kappa_be` is its scale divided by the smallest singular value when that value is known.

`prepare(selected).circuits` exposes copies of actual native circuits for explicit inspection. Preparing constructs them; resource estimation alone never compiles. The scientific Result keeps its actual `Plan` and RunData. `result.save(path)` / `load_result(path)` restore selected data and observations; `result.analyze()` consumes those observations again. Loading does not run phase search, eigensystems, polynomial selection, or a reference solve.

## Inputs, scale and padding

`block_encoding_implementation` selects the supported explicit encoding family; its default `"auto"` follows the routing below. See the [QLS API](../api/algorithms/qls.md) for the configured solver and work-limit fields.

Dense input uses the actual selected block-encoding constructor. Auto routing keeps the existing banded/Pauli/dense choices and their normalization/cost meaning. Dense-to-Pauli selection may use the current Qiskit numerical kernel; record/structured/dense-dilation boundaries remain independent of native SDK imports. A supplied `SelectedBlock` encoding must bind to the original A. Its projected equation and error are caller premises, not a hidden dense test. QLS computes no singular values of a Pauli A or of a non-dense A with a supplied encoding, so these inputs need `QLS(kappa=K)` with K at least `alpha / sigma_min(A)`, where alpha is the Pauli coefficient 1-norm or the normalization of the supplied encoding. This bound is the encoded gap parameter `kappa_be` defined below, not the condition number `sigma_max / sigma_min`, which is smaller whenever alpha exceeds `sigma_max`. Classical execution needs explicit dense A access, so a Pauli A goes to classical QLS as its dense matrix, for a Qiskit `SparsePauliOp` `A.to_matrix()`, when that copy fits within `max_bytes`. QLS has no CSR or CSC route. A SciPy sparse A whose dense copy fits in memory can be passed as `A.toarray()`, and the dense route then applies the Method's `max_bytes` and `max_work` limits to it. Otherwise quantum QLS needs Pauli access, the compact periodic stencil described below or a supplied encoding.

`alpha` is the normalization of the actual selected encoding. `kappa_be` is `alpha / sigma_min(A)` when endpoints are numerically known, or the accepted user bound. `condition_number` is the original `sigma_max / sigma_min`; none of these silently includes the polynomial-domain floor. `polynomial_kappa` is the separate domain parameter, at least 1.01. A perfectly conditioned A therefore reports condition 1 and polynomial domain 1.01. Physical recovery uses the polynomial's actual parameter:

```math
 x_{\mathrm{selected}}=
 \|b\|\frac{\kappa_{\mathrm{poly}}s}{\alpha}
 \Bigl(\frac{P}{s}\Bigr)(A/\alpha)\frac{b}{\|b\|}.
```

Here `P` is the stored inverse fit and `s` the selected positive rescale. The success branch of the circuit, or of the classical model, carries `(P/s)(A/alpha) b/||b||`, and recovery multiplies that obtained vector by `||b|| kappa_poly s / alpha`. The implementation composes scales in binary mantissa/exponent form. Overflow or loss of nonzero components makes the affected physical quantity unavailable. A legal unit direction can remain available independently.

General or non-power-of-two dense A uses the existing dense dilation with explicit normalization. Quantum selection computes the original full SVD once, preserves its left/right frames and conjugation, and pads A with the known positive block `alpha * I`; b gets zero dummy entries. Dense completion appends analytic dummy singular frames to that existing SVD. It does not decompose the padded matrix or solve a padded reference to manufacture scale. The output excludes all dummy coordinates. Scalar dimension 1 receives the same positive extension.

Classical dense QLS reuses one original spectral factorization when its selected consumers need both spectral endpoints and polynomial or norm-model data. General inverse input uses the original SVD to apply the odd polynomial in the correct singular frames. Hermitian input uses signed eigenvalues. Positive dummy padding needs no additional decomposition. The shortcut's augmented projected matrix has its own spectral action and is accounted for separately. Numerical singular endpoints remain estimates rather than proven spectral bounds.

Classical selection computes only the spectral data it needs, and supplied `alpha`/`kappa` premises avoid that spectral work. The inverse and the linear norm model take their endpoints from the original factorization they later reuse. When supplied premises avoid spectral selection, the classical evaluation obtains that factorization once and charges it to its own work. A shortcut with a numeric `t` needs endpoints only. Spectral selection first applies a binary change of scale. Hermitian input takes all eigenvalues with `eigvalsh` and uses their magnitudes, so the smallest magnitude of a spectrum that spans zero is included. General matrices use singular-values-only SVD. No route squares A to infer its gap. The `Plan` records the factorization it used in `spectral_method` and `spectral_calls`, and `QLSWork` counts the decompositions of one classical evaluation. When supplied premises avoid obtaining singular endpoints, the Result summary states that they were not independently checked at selection. Dense quantum encoding may already possess endpoints needed to reject a false premise. Classical or compact selection does not obtain an extra spectrum just to match that premise-check outcome.

The classical factor-reuse and rank-one formulas compute the same selected polynomial model in exact arithmetic. Floating-point results can differ because factorizations, basis changes and reductions use different arithmetic. Numerical comparison depends on the encoded matrix scale, polynomial coefficients and branch mass. Discrete selections can change at numerical thresholds, and normalization of a small success branch can amplify an otherwise small vector discrepancy.

The inverse fit bounds the degree-`d+1` residual `kappa*x*P(x)-1` on `[1/kappa,1]` using an affine Chebyshev grid and the shared polynomial norming inequality. Its even symmetry covers the negative interval. The recorded `certificate` is this bound, rather than the uninflated maximum of uniform samples; it is evaluated in binary64, not interval arithmetic. Spectral-premise and native phase errors remain separate from this polynomial residual bound.

The nodes matter. Ehlich and Zeller's Satz 2 (doi:10.1007/BF01111276), Eqs. (12)–(14), proves the factor `sec(pi*d/(2*N))` for Chebyshev zeros. Their Satz 1 treats an equidistant x grid separately. Sünderhauf et al. arXiv:2507.15537v1, Eq. (25), states the Chebyshev factor for equidistant x points, where it fails. For example, at degree 20 and N=500, a linear program in the Chebyshev coefficients gives a polynomial with absolute value at most 1 at every point `x_j=-1+2*j/499`, yet value greater than 1.0254 at x=.999. The claimed factor is less than 1.002. Converting the LP coefficients to exact rational numbers, dividing by their exact grid maximum and evaluating the Chebyshev recurrence at all 500 nodes and .999 verifies this counterexample without relying on solver tolerances. The implementation uses the Chebyshev-zero theorem.

Because the residual is relative, it bounds the polynomial step's solution error component by component. When every eigenvalue (or, through the dilation, singular value) of `A/alpha` lies in the polynomial domain, the transformed vector `y` satisfies `||y - (kappa_poly A/alpha)^-1 b|| <= epsilon_inv ||(kappa_poly A/alpha)^-1 b||`.

Automatic dense-to-Pauli decomposition keeps every nonzero coefficient, and automatic dense circulant detection requires exact structure. Tiny terms can increase the displayed term count or change the selected encoding and cost. The selected alpha and encoded gap describe that same operator in the host polynomial and native query. Both classical and quantum algorithm branch masses are probabilities; physical solution scaling is applied separately.

Classical observable reduction combines the observable's binary scale with the vector recovery before converting the requested scalar to binary64. It performs one observable action and accounts for its storage-preserving copy and scans. If that copy would lose a nonzero entry, it uses the original unscaled action instead, without retrying or pruning. This extends recoverable range without proving a bound on cancellation accuracy or arbitrary heterogeneous dynamic range. Genuinely unrepresentable outputs remain unavailable.

Hermiticity uses exact equality of the supplied entries. A near-Hermitian matrix uses the general route. For general inverse QLS the method selects

```math
 H=\begin{pmatrix}0&A\\A^\dagger&0\end{pmatrix},\qquad
 b_H=(b,0),
```

and takes the second half after polynomial action. The Hermitian dummy extension and this method-owned embedding do not change the original Problem.

Compact periodic `A=mI+d(2I-S-S†)` requires positive mass and zero potential. The band owner merges wrapped edges for one qubit. Original endpoints stay `m` and `m+4d`, while exact rational arithmetic bounds stored coefficient and normalization rounding outward. `polynomial_kappa` additionally covers the actual rounded encoding gap. Planning remains compact for wide metadata-only inputs; this is not permission or a claim to simulate those widths.

## Methods and outputs

| Configuration | Actual selected action | Available output |
| --- | --- | --- |
| `QLS()` | Selected odd inverse Chebyshev polynomial with stored norming-bound evidence; QSVT phase/projector/query body | Physical Solution, physical/unit StateVector, NormSquared, QuadraticForm, NormalizedExpectation, Samples |
| `QLS(solver="shortcut_native_svp", encoded_solution_norm_estimate=t)` | Dalzell kernel reflection on right singular vectors of actual G_t | Unit StateVector modulo global phase, NormalizedExpectation, Samples |
| `QLS(solver="shortcut_dilation", encoded_solution_norm_estimate=t)` | Even polynomial on the Hermitian dilation of actual G_t | Same unit-only outputs |

For the small problem above, an explicitly supplied `t=1.2` gives a legal unit-direction request:

```python
from nwqlib import StateVector

direction_result = solve(problem,
    method=QLS(solver="shortcut_native_svp", encoded_solution_norm_estimate=1.2),
    output=StateVector(normalization="unit", global_phase="modulo_global_phase"),
    seed=7)
print(direction_result)
```

That t is example-specific and does not provide a physical solution magnitude.

Shortcut methods need explicit numeric `1 <= t <= polynomial_kappa` for quantum execution. Classical execution also accepts `"grid"`, `"noisy_binary_search"` and `"linear_kappa_sequence"` as values of `encoded_solution_norm_estimate`, not of `solver`. These compute the selected classical probability model and report its actual solves/decompositions/search rows. Planning checks each model's search rows, trials and queries from `polynomial_kappa` against the limits before the model runs, with the same exact ladder length `ceil(log2(polynomial_kappa))` that the model uses. Planned trials and queries are model accounting, not executed quantum jobs. The linear model evaluates its recurrence from the same original factorization, obtained at planning or, when supplied premises avoid spectral selection, once by the evaluation. Grid and noisy search share their one target solve, and their application records the resulting encoded reference norm. No shortcut uses a reference to reconstruct a missing physical magnitude or physical global phase. There is no `reference_assistance` option.

`execution="classical"` explicitly evaluates the selected inverse or shortcut numerical model. It does not claim quantum execution. Samples requires quantum execution with a positive `shots`. Vector outputs require exact amplitude readout. The simulator vector is an explicit output, not a hardware readout.

With `shots=None`, a scalar observable is evaluated from one simulation of the selected coherent circuit. The readout fixes the method's success and physical-condition bits and excludes dummy coordinates. A Pauli observable with L terms then needs O(LN) classical work on the N-coordinate encoded system. The normalized expectation divides the projected quadratic moment by the physical-slice mass. A physical quadratic form also applies the method's recovery scale. Exact readout has no finite-shot error, while circuit, observable-action and normalization rounding remain separate error contributions. Normalized outputs are unavailable at zero physical mass.

Exact scalar readout evaluates the stored Pauli observable on the success-and-physical slice v, returning `p=v†v` and `q=v† O_tilde v`. Normalized output is q/p for positive p. A dense observable is zero extended before coefficient conversion. Its conversion error contributes separately to error against the original dense observable. Projected and full-block Pauli sums coincide for an exact zero-extension representation, while rounded coefficients can change the cancellation of dummy-coordinate contributions. Recovery and numerical reduction errors follow their separate owners.

The `Plan` of an exact scalar output has one experiment, `projected_moments`. It runs the coherent body and reduces the saved state at the body's end, with parameters that bind the success bits, condition bits, coordinate order, original dimension and stored observable. `QLSAnalysis.reduction` saves the complete native norm, the success-only mass p_alg, the physical-slice mass p and the projected moment q of that one readout, and `physical_slice_mass` is p. Complete, success and physical masses use the selected scaled squared-norm kernel and its host error budget. The complete population is checked against the producing preparation record before projection. Success and physical populations satisfy nested-subset comparisons on the same computed state. A binary-scale mass preserves its nonzero magnitude outside the ordinary binary64 output range, and physical recovery scales both the mass and its error budget. Reload analyzes these saved statistics without a new simulation. At readout and publication, QLS resolves the amplitude-derived masses label of the producing preparation record, because the projected kernel bounds its own scaled squared-norm reductions. It uses the resulting qualified saved-state budget (`saved_state_error`, the native-state budget propagated through the envelope of the host phase correction of the saved state) when available and the preparation record's probability-check tolerance convention, propagated through the same correction, otherwise. A preparation record whose host phase correction was not assessed supplies neither, and the mass check refuses the point. Both branches include the host mass error budget, preserve the raw masses and check their nested populations. The `probability_window_exclusions` of the preparation record, in `result.data.receipts`, shows which branch applies. The budget applies when `amplitude-derived masses` is its only label. Any other label, such as `unitary` or `multiplexer` for a supplied matrix instruction, `optimization_level` for a non-default compiler level or `unchecked qiskit-aer version`, keeps the probability-check tolerance convention, and so does a preparation record whose exclusions were not assessed. For example, the preparation record of exact QLS on Aer with `A = diag(1, -2, 1.5)` lists `unitary`, so its masses are checked under the probability-check tolerance convention.

Sampled observable readout groups qubit-wise-commuting Pauli terms into shared local bases. `shots` is the number of shots per selected group. Terms in a group share outcomes, so uncertainty calculations use the weighted group outcome and its covariance. Grouping reduces the number of settings. Its variance at fixed total shots depends on the state, coefficients and allocation. A padded normalized output also measures its physical-coordinate mass in an unrotated basis. A group whose basis has no X or Y supplies that mass, and otherwise a separate unrotated `physical_mass` setting measures it. Each group measures its support coordinates and valued selector bits. Samples, `physical_mass` and the unrotated group supplying a padded prefix mass measure every coordinate. Outcomes use a shared classical layout sized for the largest measured register, with unused suffix bits fixed at zero. `QLSAnalysis.groups` records each group's labels, basis, returned and selected shots and measurement. `mass_contribution_id` identifies the measurement that supplies branch mass, and its returned population owns the mass comparison. A conditional sample requires a positive selected population. Planning requires at most 64 measured bits in every selected setting, so a small-support group can use a circuit wider than its count register.

For Samples, returned shots count the original population. `samples` stores only the selected original-coordinate population, as increasing int64 original-coordinate indices and their positive int64 counts. Read `algorithm_selected_shots` and `physical_selected_shots` together with the displayed empirical selection masses. A modeled success probability is a different quantity and cannot replace these observed counts.

## Actual construction and resource meaning

The shared Program contains selected preparation, forward/adjoint original-A queries, valued controls, global phase, every converted RZ phase, projector predicates, Dalzell augmentation/projector bodies and readout. Query leaves keep the exact base encoding semantic contract. H(A) and H(G_t) contain their actual controlled child operations. Inspection, resource fold and native lowering all consume that same composition, including the selected arguments and ports. There is no opaque whole-QLS block followed by an independent reselection.

Known per-query CX models remain selected estimates, including caller-asserted native family models. A multi-controlled X with `2 <= k <= 64` controls, such as a projector flip or a `Q_b'` or `A_t` predicate, is priced at the CX count of Qiskit 2.5.2's synthesis of that gate (`MCX_CX_BY_CONTROLS`, registered under [circuit-free synthesis laws](../ENGINEERING_CONSTANTS.md#circuit-free-synthesis-laws)). The count is an estimate because another Qiskit version can synthesize the gate differently, and a gate with more controls has no CX law. One-qubit primitives and singly controlled X gates are exact at their leaf. Other controlled primitives, controlled queries, vendor workspace and hardware routing have no CX law and remain unknown. The shortcut circuits and the dilation of a non-Hermitian inverse contain such controlled operations, so their logical CX total is unknown, and only the Hermitian inverse route can report a concrete total. Shared folding supplies the actual multiplicities. Explicit prepared-circuit inspection observes its separate native population.

Select `ResourceContext(basis="cx")` when asking the logical fold for these CX laws; [resource estimation](../resources.md) shows the public call. This does not prepare or transpile a circuit.

`max_degree=256`, `max_qsp_evaluations=20000`, `max_work=1_000_000_000` and `max_bytes=10_000_000_000` bound documented numerical work and peak live arrays. Each fit candidate's bytes and work are checked against the limits before it runs, and each phase's charges accumulate across attempts. `max_work` caps each planning phase separately, for example spectral selection, dense completion, encoding construction, the polynomial fit and the comparisons of sampled Pauli grouping, so no phase is charged for the work of another. The exact projected reductions of one Run are charged together against `max_work`. A query of a dense dilation with valued controls, in the Hermitian dilation of a non-Hermitian inverse and in both shortcuts, synthesizes the dilation unitary on log2(p) + 1 qubits, for padded dimension p, exactly, once for the forward and once for the adjoint specialization of a Run, with its controls or before Qiskit controls each synthesized gate, as `dense_control_route` selects. Planning charges both syntheses, with Qiskit's control of them on the gate-wise route, for one control or for two in `shortcut_dilation`, to `max_work` and `max_bytes` ([selected blocks](../blocks.md#admission-of-the-exact-synthesis)), before the original SVD when the dense dilation is requested or forced and after encoding selection when automatic selection or a supplied encoding gives one. `dense_control_route` selects how a controlled query controls a planned dense dilation. The default `"auto"` synthesizes the forward and the adjoint controlled dilation for the one control of the Hermitian dilation and of `shortcut_native_svp`, and keeps the gate-wise route for the two controls of `shortcut_dilation` ([selected blocks](../blocks.md#dense-control-route)). The default accepts a controlled dense dilation of at most 64 padded coordinates. A supplied encoding circuit (`encoding.native`) in a query with valued controls is controlled gate-wise on every route and charged for the dense unitaries it holds, and so is a supplied RHS circuit, which `shortcut_native_svp` controls in two specializations and `shortcut_dilation` in four. The Hermitian inverse queries the dilation without controls, its construction synthesizes nothing, and a backend that lowers the circuit to a gate basis charges the synthesis to the Run's `max_synthesis_work`. These are not elapsed-time or RSS guarantees. Already-produced original SVD arrays and selected phase tables are saved once. Run archives also keep already-constructed query gates through common QPY storage rather than rebuilding numerical children to populate a cache. The [engineering constants registry](../ENGINEERING_CONSTANTS.md#explicit-workflow-and-reference-controls) records these defaults with their revisit condition.

`max_admission_steps=1_000_000` bounds the quantum Program's kept field slots and the admission work through Program validation, preparation and native lowering. Planning checks the largest requirement of those consumers for the selected Programs. A complete measured refusal names a value that admits those checks. A measurement stopped by `max_work` is identified as a lower bound. The resource fold may use up to 24 times the Program's structural-check ceiling. Raising `max_admission_steps` changes the metadata-work allowance without changing the selected polynomial or quantum operations.

## Plan and estimate beyond simulation

A periodic stencil for A and a product state for b are stored in a compact form that grows at most linearly with the number of qubits, so `plan` and `estimate` work at sizes that no simulator holds. The example below is compact planning and resource estimation only. Nothing is solved or simulated at these sizes, and no dense matrix or full state vector is formed.

```python
import numpy as np
from nwqlib import LinearSystem, estimate, plan
from nwqlib.algorithms import QLS
from nwqlib.operators import PeriodicStencil, operator_input
from nwqlib.problems import ingest_product
from nwqlib.resources import ResourceContext

q = 40  # 2**40 unknowns on a ring
A = operator_input(PeriodicStencil(q, mass=0.1, diffusion=0.1))  # 0.1 I + 0.1 (2I - S - S†)
b = ingest_product([[np.cos(0.3 + 0.1 * j), np.sin(0.3 + 0.1 * j)] for j in range(q)])  # row j is qubit j
selected = plan(LinearSystem(A=A, b=b), method=QLS(epsilon_inv=0.01), seed=7)
workload = estimate(selected, context=ResourceContext(basis="cx"))
print(selected.reconstruction.kappa_be, selected.reconstruction.degree)
print(workload.quantity("logical_width", location="logical_device").fact.value.numerator,
      workload.quantity("cx").fact.value.value)
```

| Unknowns | Encoded κ | Polynomial degree | Qubits | CX gates, estimate |
| --- | --- | --- | --- | --- |
| `2**10` | 5 | 27 | 14 | 7,304 |
| `2**20` | 5 | 27 | 24 | 24,854 |
| `2**40` | 5 | 27 | 44 | 92,354 |

These values were measured with Qiskit 2.5.2, and each plan with its estimate took under 0.2 s after the imports. With mass m = 0.1 and diffusion d = 0.1, the spectrum of A lies in `[m, m+4d] = [0.1, 0.5]` at every size, and the selected normalization is `alpha = 0.5`, so the encoded condition number and the polynomial degree stay fixed while the width grows with q. The b chosen here is not an eigenvector of the ring, so x is not proportional to b. The CX counts assume this product-state b. A general b with `2**40` amplitudes costs far more to prepare and cannot be stored as an array.

## Explicit verification and evidence

```python
from nwqlib.algorithms.qls import QLSVerification
receipt, facts = result.verify(checks=QLSVerification(
    comparisons=("spectral_domain", "inverse_relative_error", "inverse_success")
))
```

A bundle shares one solve of original `A/alpha` with the original normalized b. It reuses known original singular endpoints, and a spectral-only bundle does no solve. A bundle whose comparisons need only the norm of that solution, such as `eq17`, reuses the encoded reference norm that a grid or noisy-search norm model recorded from its own solve. That norm is the model's own intermediate, not independent evidence, and the reference description of each such check states this. A missing original spectrum is computed only when that comparison is explicitly selected. Vector access, method compatibility and finite `max_work`/`max_bytes` are checked first. Physical inverse error preserves sign and phase. Shortcut direction allows only its declared modulo-global-phase comparison. The verification record keeps actual call counts and every raw fact, and `facts` holds the top-level fact of each selected comparison. Obtaining a new spectral or solution reference requires original dense input access. Compact Pauli or sparse input is not implicitly densified, and unsupported requests receive a QLS-owned explanation. A spectral-only comparison can reuse already selected endpoints without dense access or a solve. A new dense-input `Plan` is an explicit alternative when its construction and reference costs are intended, rather than a hidden conversion of the original selection.

Absent user tolerances, inverse error compares with the disclosed polynomial and phase error budget. That relative budget is `epsilon_inv + kappa_poly * s * delta / ||y_ref||`, with `delta` the phase-fit residual bound (zero for `execution="classical"`), `s` the rescale and `y_ref = (A/alpha)^-1 b/||b||` the reference solution of the verification solve, as derived in the `_comparisons` docstring of `algorithms/qls/verification.py`. Shortcut's default `5*epsilon_inv` direction window and mass windows are heuristic. Eq.17 phase error budget (Dalzell arXiv:2406.12086v2) widens both endpoints while preserving the original raw discrepancy. The binomial term uses the predicted reference rate and the mass measurement's returned shots. An out-of-range reference prediction is not forced into a Bernoulli model. Missing phase or spectral evidence leaves dependent error budgets unknown and keeps independent raw quantities. These checks do not prove a bound on total physical error or establish confidence coverage.

Without an explicit relative tolerance, the top-level `inverse_relative_error` comparison uses metric `relative_error_over_method_allowance`, the raw relative discrepancy divided by the selected error budget. A value such as .386 means .386 of that budget, not 38.6% physical inverse error. The verification record's `.raw_relative_error` fact keeps the actual relative discrepancy. Display that fact's scalar value and the check metric together, instead of the whole nested verification record's repr.

Each selected comparison is one check of `QLSVerification.verification_checks`. Its threshold is the comparison's explicit tolerance, or `spectral_tolerance` for `spectral_domain`, or `5*epsilon_inv` for a shortcut direction without `direction_tolerance`. An inverse or mass comparison without its tolerance compares the ratio to its error budget with 1. [Verification](../verification.md#evidence-and-receipts) shows how `Certificate.with_verification` turns `facts` into a PASS or FAIL for each check, or INCONCLUSIVE when a value is unavailable. A verdict against the default direction window or a heuristic mass window remains heuristic.

`tests/test_qls_workflow.py` compares the inverse fit with the CKS (arXiv:1511.02306v2) and SNWPB (arXiv:2507.15537v1) reference polynomials in `tests/_qsp_references.py` and keeps kernel-reflection, right-singular and search arithmetic anchors. `test_qls_native.py` compares actual native paths with independent polynomial recurrences. Primary/quantum/periodic tests cover physical phase and scale, noncommuting complex 3×3 padding, stale same-shape SVD/input binding rejection, legal cache reuse, actual query/control populations, the shortcut G_t query block with one controlled synthesis per control value and adjoint flag, one exact reduction against an independent selected-circuit slice, grouped count settings, the complete-norm mass junction, invalid populations, and no-reselection archives. The [examples guide](../examples.md) lists the QLS notebooks.

## Sources and code map

Equation and section numbers refer to the arXiv versions named in the Source column. [References](../references.md) lists the full citations. Rows marked NWQLib are the library's own constructions, derived in the named docstrings.

| Step | Source | Location | Code owner |
| --- | --- | --- | --- |
| Wx QSP convention, where zero phases give `T_d` | Martyn et al., arXiv:2105.02859v5 | Sec. II.A, Eqs. (1)-(3), Theorem 1, App. A.1 | `subroutines/qsp/phases.py` `evaluate_qsp_polynomial` |
| Wx to reflection phases | Martyn et al., arXiv:2105.02859v5 | Eq. (14) and App. A.2, Eq. (A5), with `i^d` kept as a global phase | `subroutines/qsp/phases.py` `wx_phases_to_reflection` |
| Symmetric phase solving | Dong et al., arXiv:2002.11649v2 | Sec. III.1-III.5, Eqs. (23), (24), (27)-(28), (30). The larger node set and the polish are NWQLib's. The exact Jacobian and its chain rule onto the symmetric phases are derived in the `_symmetric_problem` docstring. | `subroutines/qsp/phases.py` `solve_symmetric_qsp_phases` |
| Newton start when L-BFGS misses the tolerance | Dong, Lin, Ni and Wang, arXiv:2307.12468v1 | Sec. 2.2, Sec. 3, Eq. (3.1) and Algorithm 3.1. The step-halving safeguard is NWQLib's. The least-squares step on the node residual equals the Newton step on the Chebyshev coefficients, as derived in the `_damped_newton` docstring. | `subroutines/qsp/phases.py` `_damped_newton` |
| Sup norm from Chebyshev samples | Ehlich and Zeller (1964), doi:10.1007/BF01111276 | Satz 2, Eqs. (12)–(14), pp. 42–43. Sünderhauf (arXiv:2507.15537v1) Eq. (25)'s equidistant-x statement is not valid | `subroutines/qsp/phases.py` `chebyshev_polynomial_sup_bound` |
| Projector-controlled phase and QSVT sequence | Martyn et al., arXiv:2105.02859v5, and Gilyén et al., arXiv:1806.01838v1 | Martyn Eq. (27), Fig. 3, Theorems 3-4. Gilyén Theorem 17, Lemma 19 | `subroutines/qsp/evolution.py` `build_qsvt_circuit`, `algorithms/qls/quantum.py` |
| Real part through the `+Phi`/`-Phi` pair | Gilyén et al., arXiv:1806.01838v1 | Corollary 18, Eq. (33) | `subroutines/qsp/evolution.py` `build_real_chebyshev_encoding`, `algorithms/qls/quantum.py` |
| Inverse fit with proven relative residual bound | NWQLib | `subroutines/qsp/inverse.py` module docstring | `subroutines/qsp/inverse.py` `InverseChebyshevFit` |
| Inverse degree law behind the tripwire | Childs, Kothari and Somma, arXiv:1511.02306v2 | Lemmas 17-19, Eqs. (74), (77), (88) | `subroutines/qsp/inverse.py` `inverse_degree_law_bound` |
| Residual-bound grid density `N = 25d` | Sünderhauf et al., arXiv:2507.15537v1 | Sec. III, Eq. (26) | `subroutines/qsp/inverse.py` |
| Reference `1/x` polynomials for tests | Childs, Kothari and Somma, arXiv:1511.02306v2, and Sünderhauf et al., arXiv:2507.15537v1 | CKS Lemmas 17-19. Sünderhauf Theorem 1, Eqs. (4)-(13) | `tests/_qsp_references.py` |
| Rescale and physical recovery | NWQLib, with the QSP bound of Martyn et al., arXiv:2105.02859v5 | App. A.1, Theorems 9-10, condition (iii) | `algorithms/qls/host_planning.py` `select_polynomial`, `algorithms/qls/method.py` |
| Spectral premises, padding and Hermitian dilation | NWQLib | `select_inputs` and `_spectrum` docstrings, and the dilation paragraph of this guide | `algorithms/qls/host_planning.py`, `algorithms/qls/method.py`, `algorithms/qls/quantum.py` |
| Original factor reuse for the inverse and the linear norm model | NWQLib | `inverse_polynomial_action`, `original_factor_laws` and `selected_work` docstrings | `algorithms/qls/numerical.py` `inverse_polynomial_action`, `linear_model_weights`, `algorithms/qls/host_planning.py` `_original_factors`, `selected_work` |
| Kernel-reflection polynomial and `eta` | Dalzell, arXiv:2406.12086v2 | Eqs. (6), (22), App. B.2, Eqs. (51)-(54) and Lemma 1, App. B.3, Eq. (62) and Lemma 3. The closed form `T_ell(z0) = cosh(ell arccosh z0)` and the proof that Eq. (6) gives `F(Delta) <= eta` are in the `_realize_kernel_reflection` docstring. | `subroutines/qsp/shortcut.py` `plan_kernel_reflection`, `dalzell_eta_from_precision` |
| Augmented system `A_t`, `b'`, `G_t` and their circuits | Dalzell, arXiv:2406.12086v2 | Eqs. (8)-(11), App. A.1-A.4 and A.6. With the augmented coordinate on its own qubit, preparing `b'` needs no extra ancilla, unlike App. A.3. | `algorithms/qls/numerical.py` `shortcut_matrix`, `projected_augmented_inplace`, `algorithms/qls/quantum.py` |
| Algorithm 1 (prepare `e_n`, reflect, project) | Dalzell, arXiv:2406.12086v2 | Algorithm 1, p. 5, and App. B.1, Eq. (50) | `algorithms/qls/numerical.py` `shortcut_polynomial_action`, `hermitian_dilation`, `project_shortcut_branch` |
| Success-probability check tolerance | Dalzell, arXiv:2406.12086v2 | Eqs. (7), (17) | `algorithms/qls/norm_search.py`, `algorithms/qls/verification.py` |
| Norm search: grid, noisy binary search, linear-in-kappa sequence | Dalzell, arXiv:2406.12086v2 | Sec. 5.1, Eqs. (23)-(25). Sec. 5.2, Eqs. (26)-(29), (44). Sec. 5.3, Eqs. (31)-(36), (41)-(42) | `algorithms/qls/norm_search.py`, `algorithms/qls/numerical.py` `linear_model_weights` |
| Periodic stencil gap and error accounting | NWQLib | `select_periodic_encoding` docstring | `algorithms/qls/periodic.py` |
| Verification error budgets | NWQLib, with Dalzell arXiv:2406.12086v2, Eq. (17) | `_comparisons` docstring | `algorithms/qls/verification.py` `QLSVerification` |
| CX count of a multi-controlled X | Qiskit 2.5.2 synthesis, tabulated. No paper location | `MCX_CX_BY_CONTROLS` registry entry | `algorithms/qls/quantum.py`, `subroutines/_mcx_counts.py` |

The noisy binary search eliminates candidates above a probe whose predicted success probability exceeds 1/2, the direction implied by Eq. (44). The Sec. 5.2 text of Dalzell arXiv:2406.12086v2 states the opposite direction. Dalzell arXiv:2406.12086v2 Lemma 3, item 2, prints the kernel-reflection bound `-1 + 4 eta/(1 + eta)` for `|K(x)|`, which is negative for `eta < 1/3`. The proof through Eq. (63) and the statement on p. 4 bound `K(x)` itself, `-1 <= K(x) <= -1 + 4 eta/(1 + eta)` on `1/kappa <= x <= 1`. That signed bound is the one behind Eqs. (5) and (17), and so behind the Eq. (17) success window of `QLSVerification`. Costa, Dalzell, An and Berry (arXiv:2604.22185v2) compare the shortcut and adiabatic solvers numerically and find the shortcut better for non-Hermitian matrices when the solution norm is known. Their constants and the discrete-adiabatic and initial-state-query solvers are background here, not implemented routes.
