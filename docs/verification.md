# Check accuracy and verify a result

<a id="verification"></a>NWQLib checks a result in two ways. `result.verify` compares it with an independent reference computation, such as an exact classical solution, and `result.assess` tests the error bounds that the method attached to it against a tolerance you choose. Neither runs during `solve` and neither takes new measurements. Each runs only when you call it.

## Checks for each method {#checks-for-each-method}

`result.verify(checks=options)` takes one options record of the types below.

| Method | Options type | Returned `facts` |
| --- | --- | --- |
| LCHS | `LCHSVerification` | The discrepancy from the selected reference solution, and for `ivp_closed_form` the reference consistency |
| LCHS | `LCHSRefinement` | Refined output-error components for `result.assess`, which answer no check |
| QLS | `QLSVerification` | The top-level fact of each selected comparison |
| QPE | `QPEVerification` | `component_error` and `overlap_deficit` |
| Lanczos, FixedGCIM, ADAPT | `ProjectedVerificationOptions` | One fact per selected criterion ([projected quantities](#projected-quantities)) |
| Lanczos, FixedGCIM | `EnergyShiftOptions` | The energy-shift discrepancy between two Results ([energy shift](#two-result-energy-shift)) |
| ADAPT | `AdaptVerificationOptions` | One fact per scalar of the selected checks |
| QHD | `QHDVerification` | `grid_minimum` and the infidelity of each selected fidelity comparison ([grid minimum](#qhd-grid-minimum)) |
| QHD, one-hot encoding | `NumberSectorOptions` | The observed number-sector leakage ([number sector](#measured-number-sector-leakage)) |

A Method without checks, such as `ExpectationMethod`, raises `ValueError` from `result.verify`. The method guides describe each check: [LCHS](algorithms/lchs.md#explicit-checks-and-saved-results), [QLS](algorithms/qls.md#explicit-verification-and-evidence), [QPE](algorithms/qpe.md), [QHD](algorithms/qhd.md#explicit-verification) and [GCiM / ADAPT](algorithms/gcim.md).

## Compare a result with a classical reference {#compare-with-a-classical-reference}

This cell solves `du/dt = -A u` from the [quickstart](quickstart.md) and compares the result with the exact solution `u(t) = exp(-A t) u(0)`, first directly and then through `result.verify`:

```python
import numpy as np
from scipy.linalg import expm
from nwqlib import LinearDynamics, solve
from nwqlib.algorithms import LCHS
from nwqlib.algorithms.lchs import LCHSVerification

A = np.array([[0.4, 0.15], [0.05, 0.25]])
u0 = np.array([1.0, 0.0])
result = solve(LinearDynamics(A=A, initial_state=u0, time=0.1), method=LCHS())

reference = expm(-0.1 * A) @ u0
print(np.linalg.norm(result.solution - reference))

checks = LCHSVerification(reference="expm", metric="absolute_l2",
                          threshold=0.01)
receipt, facts = result.verify(checks=checks)
print(facts[0].fact.value.value)
```

```text
0.0008214720329548587
0.0008214720329548587
```

Both lines are the absolute L2 distance between the LCHS solution and the matrix exponential. Every built-in options type returns the pair `(receipt, facts)`:

- `receipt` is the verification record (`VerificationReceipt`). It names the `Plan`, Result and options, and holds the reference computation, its numerical calls and every raw number they produced.
- `facts` is a tuple of `FramedFact` records that cite `receipt`. Each holds one value together with its error frame (`ErrorFrame`, the quantity, metric, unit, scope and conditioning that the value refers to). Except for `LCHSRefinement`, `facts` holds the values that answer the options' `verification_checks`, here one discrepancy against the threshold .01.

A reference discrepancy is empirical numerical evidence for this input, not a complete proven error bound. [Record check verdicts](#evidence-and-receipts) turns `facts` into a PASS or FAIL for each check.

When a Lanczos energy differs from its reference, the checks in [If the energy looks wrong](algorithms/lanczos.md#if-the-energy-looks-wrong) separate sampling error, circuit error and the limits of the trial space.

## Check against a tolerance {#check-against-a-tolerance}

`result.assess` asks whether the error bounds attached to the result prove a tolerance. Continue the cell above:

```python
assessment = result.assess(absolute_tolerance=0.01)
print(assessment.status)
for reason in assessment.prerequisites:
    print(reason)
print(assessment.unverified)

sampling = result.assess(absolute_tolerance=0.01, component="sampling")
print(sampling.status)
```

```text
INCONCLUSIVE
native_floating_point: selected error component is unavailable
required failure probability is unsupported
('algorithmic_approximation',)
PASS
```

The reference comparison measured a discrepancy of about .000821, but the bounds do not prove a total error of at most .01. No bound is available for native floating-point error, and the method's algorithmic approximation bound, about .00931, is a numerical estimate rather than a proof, so it is listed in `unverified`. The sampling component passes because this run used exact readout ([sampling error by method](#sampling-error-by-method)).

`assess` works on the saved experiment and data. It does not measure data, change shots or rerun the method. A Result does not store an accuracy assessment. Each `ClaimAssessment` stores its own `accuracy`, the experiment and data it assessed and the evidence it used, and a later stricter criterion is a new assessment of the same experiment. The default confidence is .95.

`Result.facts` holds the method's error evidence even when no criterion was selected, and an assessment reads those facts without changing the Result. A replacement `FramedFact` passed as `assess(facts=...)` carries its own `failure_probability`. Omitting it leaves the confidence unknown, even if the original model term had a known probability. Saved assessment facts keep the probability actually used, so changing the requested confidence cannot relabel older evidence.

### Read the assessment

| Field | Meaning |
| --- | --- |
| `status` | `PASS`, `INCONCLUSIVE` or `NOT_APPLICABLE` (below) |
| `threshold` | Sufficient absolute threshold in the output unit, or `None` without an established scale |
| `covered_subtotal` | Triangle-inequality sum of the covered bounds |
| `covered` | Sources whose bounds entered the subtotal |
| `remaining` | Needed sources without an applicable additive bound |
| `unverified` | Covered sources whose bound, prerequisites or inputs lack supported proof |
| `prerequisites` | Reasons that prevent PASS |
| `failure_probability` | Union-bound failure probability of the covered bounds and reference, capped at one, or `None` when any probability is unknown |

| Status | Meaning |
| --- | --- |
| `PASS` | The complete supported sufficient criterion meets the tolerance and confidence. Every needed source has a supported bound, the subtotal meets the threshold and the failure probability meets the confidence |
| `INCONCLUSIVE` | Every other case. An upper bound above the tolerance is inconclusive, because it does not prove that the error exceeds the tolerance. Numerical or asserted terms remain visible in the subtotal and in `unverified` |
| `NOT_APPLICABLE` | Only for a `component="sampling"` criterion on a model that declares no sampling source |

### How a PASS is reached

The method's error model lists the error sources of its output, each already propagated into the output's error frame. `assess` combines them as follows.

- **Bounds add by the triangle inequality.** This rests on an assumption that NWQLib does not check, namely that the Method has propagated every source into the output frame, so that the output error is the sum of the propagated source errors. Where each source bound then holds, the absolute output error is at most the sum of the bounds.
- **Failure probabilities combine by the union bound**, which needs no independence. If bound k fails with probability at most `delta_k`, all bounds hold together with probability at least `1 - sum_k delta_k`, so PASS at a required confidence `c` needs `sum_k delta_k <= 1 - c`.
- **The source list must be complete.** The Method that builds the model must list every error source its output needs, and NWQLib does not check that list for completeness. For example, a zero residual for the excited eigenpair of `diag(0,1)` does not establish ground identification. A missing required source stays unknown.
- **A component is not the total.** `component="sampling"` assesses only the sampling contribution and does not establish total physical accuracy.

## Relative criteria {#relative-criteria}

A relative criterion `|error| <= r*|target|` needs a supported nonzero scale for the target, passed as a `TargetReference`. A `TargetReference` states one of three relations to the target: an exact target, a lower bound on its magnitude or an upper bound on its magnitude. An exact nonzero target or a positive lower bound on the magnitude supplies a sufficient relative threshold. An observed estimate or an upper bound on the magnitude does not. The reference's failure probability enters the union bound.

The Bell state $(|00\rangle+|11\rangle)/\sqrt2$ has $\langle ZZ\rangle=\langle XX\rangle=1$, so the expectation of $ZZ+0.5\,XX$ is exactly 3/2 ([Run your own circuit](own_circuit.md#measure-an-observable-on-your-circuit)). This cell states that value as an exact target and assesses the result against the relative tolerance 0.05:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from nwqlib import Expectation, solve
from nwqlib.algorithms import ExpectationMethod
from nwqlib.core import Rational, Source
from nwqlib.evidence import Evidence, Fact, FramedFact, TargetReference

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
H = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
bell = solve(Expectation(state=qc, observable=H), method=ExpectationMethod())

frame = bell.plan.error_model.frame
derivation = Source(name="Bell-state expectation", version="1",
                    domain=frame.scope.domain,
                    reference="<ZZ> = <XX> = 1 for (|00> + |11>)/sqrt(2)")
proof = Evidence(kind="proved_relation", source=derivation,
                 status="witnessed", artifact="1 + 0.5 * 1 = 3/2",
                 witnessed_scope=frame.scope,
                 subject_id=bell.plan.problem.content_id)
exact = Fact(quantity=frame.quantity, unit=frame.unit, scope=frame.scope,
             availability="concrete",
             value=Rational(numerator=3, denominator=2), evidence=proof)
bell_target = TargetReference(relation="exact_target", failure_probability=0.0,
    fact=FramedFact(frame=frame, bindings=(), fact=exact,
                    failure_probability=0.0))

relative = bell.assess(relative_tolerance=0.05, reference=bell_target)
print(relative.threshold.numerator / relative.threshold.denominator)
print(relative.status)
```

```text
0.07500000000000001
INCONCLUSIVE
```

The exact target gives the sufficient absolute threshold 0.05 × 3/2 = 0.075, printed with the binary64 rounding of 0.05. The status stays INCONCLUSIVE because `ExpectationMethod` attaches no bound for its `native_preparation`, `native_simulation`, `physical_model` and `reconstruction` error sources, which `relative.prerequisites` lists. `kind="proved_relation"` with `status="witnessed"` is your statement that `artifact` records the derivation of the value for the Problem named by `subject_id`. NWQLib does not check that derivation. With the default `status="declared"`, the threshold stays unavailable.

If the relative scale is unavailable or zero, the threshold stays unavailable. Passing `absolute_fallback=...` to `assess` selects a separate positive absolute criterion for that case. It does not change the `Accuracy` used at planning or establish the relative criterion. Supplying both an absolute and a relative tolerance is invalid.

## Sampling error by method {#sampling-error-by-method}

| Method | Sampling error with exact readout (`shots=None`) |
| --- | --- |
| QCELS, Expectation, Lanczos, FixedGCIM, ADAPT, LCHS | Zero, recorded in `Result.facts`, so `component="sampling"` can PASS |
| SPE, RFE, and RWPE with at least one update | Unknown, because no fact bounds the draws the Method makes itself |
| QLS, QHD | No sampling source, because these Methods list one only when shots are drawn. `component="sampling"` gives `NOT_APPLICABLE` rather than missing |

Counts and provider estimates draw outcomes and never receive this zero.

The `sampling` source is the statistical error of an estimate that averages or emulates the outcomes of random draws. Measurement outcomes drawn by a backend are such draws, and so are the draws a Method makes itself. Classical RWPE emulates one measurement bit from each exact signal. RFE and SPE draw their powers or frequencies at random, and their finite average over the draws estimates a sum over the draw distribution. A draw that only selects an input, such as the reference state that Lanczos draws when `initial_state` is omitted, fixes the state that the Plan records. The estimate averages nothing over it, so its effect belongs to `ground_identification` and `projected_solve`.

Exact readout means that every observation is an exact probability, Pauli value, amplitude or host scalar, as with `shots=None`, or that the Result needed no observation. With exact readout and an estimate that averages or emulates no random draw, sampling contributes zero. A Method whose error model lists a sampling source records that zero in `Result.facts` as a proved relation, with the saved observations as its evidence (`evidence.error_model.exact_readout_sampling`). The observations cannot show the draws a Method makes, so the Method states them when it requests the fact. Exact readouts are themselves checked against a binary64 roundoff window recorded in each preparation record. That window bounds roundoff in the saved values and is not a tolerance for scientific accuracy ([readout tolerance branches](error_evidence.md#readout-tolerance-branches)).

A sampling bound that meets the threshold does not always PASS. The conditional Hoeffding radius of `ExpectationMethod` (doi:10.1080/01621459.1963.10500830) can meet the numerical threshold while `component="sampling"` remains INCONCLUSIVE, because its sampling-model assumption and its binary64 evaluation are not certified support, as the [saved-results example](saved_evidence.md#save-load-and-reanalyze) shows. The interval remains useful under its stated assumptions. A replacement bound you supply needs genuine evidence for the same frame, subject, parameter point and failure probability. Changing a record's kind or status, or deleting assumptions, does not establish those facts.

## Record check verdicts {#evidence-and-receipts}

Each call to `verify` produces one verification record (`VerificationReceipt`). It names its `Plan`, Result and construction, the options, an invocation ID, the reference and the numerical calls, and it holds every raw fact of the computation. The returned facts cite it as their source. Loading a saved record creates no new invocation and repeats no computation. `verify_projected`, `verify_energy_shift` and `verify_number_sector` return the same `(receipt, facts)` pair as `result.verify`.

Each check is a `CheckSpec`, which names the quantity, its mathematical domain, threshold, required data access, computation and stored output. A `Certificate` joins an accuracy assessment (`ClaimAssessment`) with the facts of completed checks and gives each check a status. Continue the cell of [the first section](#compare-with-a-classical-reference):

```python
from nwqlib.evidence import Certificate

certificate = Certificate(plan_id=result.plan_id, result_id=result.content_id,
    assessment=result.assess(absolute_tolerance=0.01), checks=())
certificate = certificate.with_verification(result, options=checks,
                                            evidence=facts)
print(certificate.checks[0].status)
print(certificate.assessment.status)
```

```text
PASS
INCONCLUSIVE
```

| Check status | Meaning |
| --- | --- |
| `PASS`, `FAIL` | The value, checked against the check's domain, compared with the threshold |
| `INCONCLUSIVE` | The value, threshold, accepted parameter point or support tied to this Result is missing, or an assumption is open |
| `NOT_RUN` | No fact was supplied |
| `NOT_APPLICABLE` | The fact states that the check does not apply |

A check PASS concerns only its own scalar threshold. The aggregate accuracy assessment can independently remain INCONCLUSIVE, as it does here. Attaching facts does not rerun verification, change the experiment or turn numerical evidence into a theorem. A negative value of an error or norm quantity is rejected by its domain. A roundoff adjustment needs its own justification, keeps the raw value and is disclosed. See the [Accuracy and verification API](api/evidence.md).

### Facts belong to the options that produced them

Each returned fact with a concrete value records, in `fact.evidence.options_id`, the content hash of the complete options record that produced it. `with_verification` compares that hash with the options you pass and accepts the fact only when they are the same record. A fact whose value is unavailable carries no evidence or hash and attaches as INCONCLUSIVE.

The hash covers every field and the record's revision history (`parent_id`):

- Keep the options record that you passed to `verify`, or its saved JSON. Options and facts saved as JSON and read back still attach.
- A record with the same values but another revision history, for example one made by `revise` instead of the constructor, has a different hash.
- Changing any field, including only the threshold or tolerance, also gives a new hash, and the existing facts then no longer answer the check. Attaching them raises `ValueError`. Its message reports an options identity mismatch, or a frame mismatch when the changed option is part of the check's frame, such as a number-sector particle count.

A fact that answers a selected check also records, in `fact.evidence.check_id`, the content hash of that exact `CheckSpec`. The direct `assemble_check(check, artifact_id=..., fact=...)` form compares it as well, so a `CheckSpec` revised after verification, for example with another threshold, raises `ValueError` there too. Output-error facts from `LCHSRefinement` answer no check and carry no check hash.

Attaching never runs verification, a reference solver or a backend to fill that gap. To use different options, run the verification again with them and attach the new facts:

```python
stricter = checks.revise(threshold=0.001)
# certificate.with_verification(result, options=stricter, evidence=facts)
# raises ValueError: those facts were produced by checks.
stricter_receipt, stricter_facts = result.verify(checks=stricter)
certificate = certificate.with_verification(result, options=stricter,
                                            evidence=stricter_facts)
print([check.status for check in certificate.checks])
```

```text
['PASS', 'PASS']
```

The certificate now holds both checks, and the discrepancy of about .000821 is also below .001.

## Projected quantities

These checks apply to Lanczos, FixedGCIM and ADAPT and read values the Result already stores:

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.operators import ingest_pauli
from nwqlib.evidence.verification import (
    ProjectedVerificationOptions, verify_projected)

operator = ingest_pauli((("Z", 1.),), num_qubits=1)
method = FixedGCIM(basis=([1., 0.], [0., 1.]))
result = solve(Eigenproblem(A=operator), method=method, execution="classical")
projected_options = ProjectedVerificationOptions(
    name="projected",
    comparisons=("gram_hermiticity", "gram_psd_deficit",
                 "projected_backward_error", "overlap_normalization"),
    tolerance=1e-10,
)
projected_receipt, projected_facts = verify_projected(
    result, options=projected_options)
for fact in projected_facts:
    value = fact.fact.value  # An exact Rational record here.
    print(fact.fact.quantity, value.numerator, value.denominator)
```

```text
projected.gram_hermiticity 0 1
projected.gram_psd_deficit 0 1
projected.projected_backward_error 0 1
projected.overlap_normalization 0 1
```

For this diagonal one-qubit example every value is exactly zero.

The Method's `projected_diagnostics()` supplies the stored Gram coordinates, their raw spectrum before filtering, the normalization defect and the projected backward error. The check keeps their original basis convention, including raw Chebyshev columns in Lanczos and normalized preparation columns in GCIM. A matrix or spectrum that the Result does not store is not reconstructed.

| Criterion | Value |
| --- | --- |
| `gram_hermiticity` | The maximum absolute real or imaginary component of `S-S†` over the stored square coordinates. This does not establish agreement between independently measured opposite entries |
| `gram_psd_deficit` | `max(0,-min(spectrum))` over the stored raw spectrum |
| `projected_backward_error` | `||H c - E S c|| / ((||H||_F + |E| ||S||_F) ||c||)` for the lowest Ritz pair, with c its coefficient vector, on the pencil described below |
| `overlap_normalization` | The stored coefficient normalization defect. The basis is not normalized again |

The two families evaluate `projected_backward_error` on different pencils. Lanczos uses the normalized pencil (K, S) of its solve, with `K = (H - center S) / alpha` and the Ritz value `(E - center) / alpha`, where `center` and `alpha` are the midpoint and half-width of its spectral enclosure. Adding an identity term `c_I I` to a Pauli input leaves the value unchanged in exact arithmetic. For a matrix input it changes the value only through the roundoff budget of the Gershgorin frame, which grows with `|c_I|`. FixedGCIM and ADAPT use the physical pencil (H, S) in the Problem's energy unit, so adding `c_I I` changes `||H||_F` and `|E|` and therefore the value. Compare values of this criterion only within one family.

These criteria concern the projected problem. They do not provide a full-state residual, ground-state identity or total physical-error guarantee.

## Two-result energy shift

With the trial space fixed, adding `cI` to the Hamiltonian shifts every Ritz value by exactly c in exact arithmetic. This check compares two Results to test that relation:

```python
from nwqlib.evidence.energy_shift import EnergyShiftOptions, verify_energy_shift

baseline_result = result  # The projected example above.
shifted_operator = ingest_pauli((("Z", 1.), ("I", .5)), num_qubits=1)
shifted_result = solve(Eigenproblem(A=shifted_operator), method=method,
                       execution="classical")
options = EnergyShiftOptions.for_result(
    baseline_result, name="identity_shift", shift=0.5, tolerance=1e-10
)
receipt, facts = verify_energy_shift(shifted_result, options=options)
value = facts[0].fact.value
print(value.numerator, value.denominator)
```

```text
0 1
```

The check reads from the two Results their operator tables, basis, the content hashes of their preparations, subspace rule and order, sector, error frame and scalar values. It first establishes `H_shifted - H_baseline = cI` exactly for the recorded binary64 entries of the stored tables, where absent entries denote exact zeros, and compares the operators in the same basis and trial-space construction. Only then does it report the energy discrepancy `abs(E_shifted-E_baseline-c)`. Neither workload is executed again. An explicitly asserted relation remains an unresolved assumption, and numerical agreement does not prove that assertion. A valid operator relation does not imply zero discrepancy between independently rounded or sampled Ritz estimates.

With the trial space held fixed, adding `cI` leaves the Gram matrix `S` unchanged and adds `cS` to the projected Hamiltonian, so every Ritz value shifts by exactly `c` in exact arithmetic. A Chebyshev trial space equals the power Krylov space of the operator (Kirby, Motta and Mezzacapo, arXiv 2208.00567v4, Eq. (14)), which the shift does not change. The table check requires matched preparations and subspace rule and order so that the two trial spaces are the same. The discrepancy is then zero in exact arithmetic. For exact or classical moments it measures floating-point departure, and for sampled moments it also contains the sampling difference between the two runs.

## Measured number-sector leakage

This check reports the fraction of measured shots whose bitstring does not have exactly N ones, with N set by `particles`:

```python
import sympy as sp
from nwqlib import Optimization
from nwqlib.algorithms.qhd import QHD
from nwqlib.evidence.sector import NumberSectorOptions, verify_number_sector

x = sp.Symbol("x", real=True)
problem = Optimization(objective=x*x, variables=(x,), bounds=((-1., 1.),))
sector_result = solve(problem, method=QHD(), shots=32, seed=7)
receipt, facts = verify_number_sector(
    sector_result,
    options=NumberSectorOptions(name="number", particles=1, tolerance=0.01),
)
value = facts[0].fact.value
print(value.numerator, value.denominator)
```

```text
0 1
```

No measured shot of this run left the one-particle sector.

The check supports QHD with the one-hot encoding and complete-register counts, which the solve above supplies. A binary-encoded QHD Result is refused, because every binary outcome encodes a grid point. Exact probabilities, Expectation parity bits and GCIM or Lanczos interferometer ancillas do not supply the data, and the check refuses them with a `ValueError`. Counts whose width differs from the complete register, or whose total differs from the returned shots, are rejected.

The check reports observed Hamming-weight leakage, not a chemistry interpretation independent of encoding. A correct mean particle number is insufficient. Equal weight on `N=0` and `N=2` has mean one and complete leakage outside `N=1`. Zero observed leakage does not prove pure-state membership or zero underlying leakage.

## QHD grid minimum {#qhd-grid-minimum}

`result.verify(checks=QHDVerification(comparisons=("grid_minimum",)))` evaluates the objective on the finite grid, including its constant term. The returned gap is the candidate's gap to the least of these freshly evaluated binary64 values. It is a diagnostic of that evaluated grid and does not by itself bound the gap for the exact mathematical objective. The verification record also reports `most_probable_gap`, the same gap for the most probable point, which the augmented-Lagrangian and refinement layers read by default ([QHD guide](algorithms/qhd.md#explicit-verification)).

The comparison can also report `minimum_success_mass`, the observed weight of grid points whose evaluated objective differs from the evaluated minimum by at most `minimum_tolerance`. With zero tolerance, this counts ties in the evaluated values. The mass is computed from a kept state (`keep_state=True`), exact probabilities from the backend or counts. A classical Result without a kept state reports it as unavailable.

## Cost of each check {#cost-of-each-check}

Checks run only when you call `result.verify` or one of the `verify_*` functions, and the cost depends on the check:

| Check | What each call computes |
| --- | --- |
| Projected, energy shift, number sector | Reads values the Results already store, so repeating it only scans those records. These checks do not measure data or run an eigensolver |
| ADAPT state checks | Rebuild the processed Ritz state, from saved vectors for a classical Result or by simulating the reference preparation and selected generators for a quantum-execution Result, then apply the residual or sector operators |
| LCHS `expm`, `ivp`, `closed_form`, `ivp_closed_form` | A new classical reference solution, using a matrix exponential, an IVP integration or both, within the size and RHS limits in their options |
| LCHS `selected_grid` | Evaluates the saved finite recipe again, except in the initial and zero reconstruction modes, where its reference is the initial vector |
| QLS | Solves the normalized original system once unless `spectral_domain` is its only comparison, and computes a missing spectrum only for `spectral_domain` |
| QPE | Diagonalizes the dense target unless the Run already holds its eigensystem |
| QHD `grid_minimum` | Evaluates the objective at every grid point |
| QHD fidelity comparison | Runs one restricted evolution |

Method-specific state, circuit, spectrum and reference checks have their own size limits in their options. Missing required data produces a limitation that names it, and no measurement is repeated to supply it.

All exact scalar comparisons use the finite `ExactArithmetic.max_integer_bits` guard. This is an arithmetic representation boundary, not an execution permission or a process-memory limit. Extra simulation, reference decompositions and large state operations run only for a check you select.
