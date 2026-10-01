# Verification

Verification is an explicit operation on an actual Result and its saved data. Running a method does not automatically run a reference computation. The selected check names the quantity, its mathematical domain, tolerance, access, computation and stored output.

The common scalar checks do not measure data or run an eigensolver. Method-specific state, circuit, spectrum and reference checks keep their own numerical owners and concrete size controls. Missing required data produces an actionable limitation rather than a hidden repeated measurement.

## Projected quantities

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.operators import ingest_pauli
from nwqlib.evidence.verification import ProjectedVerificationOptions, verify_projected

operator = ingest_pauli((("Z", 1.),), num_qubits=1)
method = FixedGCIM(basis=([1., 0.], [0., 1.]))
result = solve(Eigenproblem(A=operator), method=method, execution="classical")
projected_options = ProjectedVerificationOptions(
    name="projected",
    comparisons=("gram_hermiticity", "gram_psd_deficit", "projected_backward_error", "overlap_normalization"),
    tolerance=1e-10,
)
projected_receipt, projected_facts = verify_projected(result, options=projected_options)
```

The explicit solve above supplies the data. The producer's `projected_diagnostics()` supplies existing Gram coordinates, their raw pre-filter spectrum, normalization defect and projected backward error. The check preserves their original basis convention, including raw Chebyshev columns in Lanczos and normalized preparation columns in GCIM. `overlap_normalization` reads the stored coefficient normalization defect and does not normalize the basis again.

`gram_psd_deficit` scans the stored raw spectrum and returns `max(0,-min(spectrum))`. `gram_hermiticity` scans stored square coordinates for the maximum absolute real/imaginary component of `S-S†`. This does not establish agreement between independently measured opposite entries. The other checks read their existing scalars. No absent matrix or spectrum is reconstructed.

`projected_backward_error` is `||H c - E S c|| / ((||H||_F + |E| ||S||_F) ||c||)` for the lowest Ritz pair, with c its coefficient vector, but the two producers evaluate it on different pencils. Lanczos uses the normalized pencil (K, S) of its solve, with `K = (H - center S) / alpha` and the Ritz value `(E - center) / alpha`, where `center` and `alpha` are the midpoint and half-width of its spectral enclosure. Adding an identity term `c_I I` to a Pauli input leaves the value unchanged in exact arithmetic. For a matrix input it changes the value only through the roundoff budget of the Gershgorin frame, which grows with `|c_I|`. FixedGCIM and ADAPT use the physical pencil (H, S) in the Problem's energy unit, so adding `c_I I` changes `||H||_F` and `|E|` and therefore the value. Compare values of this criterion only within one family.

These criteria concern the projected problem. They do not provide a full-state residual, ground-state identity or total physical-error guarantee.

## Two-result energy shift

```python
from nwqlib.evidence.energy_shift import EnergyShiftOptions, verify_energy_shift

baseline_result = result  # The projected example above.
shifted_operator = ingest_pauli((("Z", 1.), ("I", .5)), num_qubits=1)
shifted_result = solve(Eigenproblem(A=shifted_operator), method=method, execution="classical")
options = EnergyShiftOptions.for_result(
    baseline_result, name="identity_shift", shift=0.5, tolerance=1e-10
)
receipt, facts = verify_energy_shift(shifted_result, options=options)
```

The two Results supply their actual operator tables, basis, preparation identities, subspace rule/order, sector, frame and scalar values. The exact table check establishes `H_shifted=H_baseline+cI` in the matched coordinates before comparing `abs(E_shifted-E_baseline-c)`. An explicitly asserted relation remains an unresolved premise. Numerical agreement does not prove that assertion. Neither workload is executed again.

The stored-table energy-shift check establishes `H_target − H_baseline = cI` for the recorded binary64 entries. Absent entries denote exact zeros. It compares the operators in the same basis and trial-space construction before reporting the energy discrepancy. A valid operator relation does not imply zero discrepancy between independently rounded or sampled Ritz estimates.

With the trial space held fixed, adding `cI` leaves the Gram matrix `S` unchanged and adds `cS` to the projected Hamiltonian, so every Ritz value shifts by exactly `c` in exact arithmetic. A Chebyshev trial space equals the power Krylov space of the operator (Kirby, Motta and Mezzacapo, arXiv 2208.00567v4, Eq. (14)), which the shift does not change. The table check requires matched preparations and subspace rule and order so that the two trial spaces are the same. The discrepancy is then zero in exact arithmetic. For exact or classical moments it measures floating-point departure, and for sampled moments it also contains the sampling difference between the two runs.

## Measured number-sector leakage

```python
import sympy as sp
from nwqlib import Optimization
from nwqlib.algorithms.qhd import QHD
from nwqlib.evidence.sector import NumberSectorOptions, verify_number_sector

x = sp.Symbol("x", real=True)
sector_result = solve(Optimization(objective=x*x, variables=(x,), bounds=((-1., 1.),)),
                      method=QHD(), shots=32, seed=7)
receipt, facts = verify_number_sector(
    sector_result, options=NumberSectorOptions(name="number", particles=1, tolerance=0.01)
)
```

The current supporting producer is QHD with the one-hot encoding and selected complete-register counts, which the explicit solve above supplies. A binary-encoded QHD Result is refused, because every binary outcome encodes a grid point. Exact probabilities, Expectation parity bits and GCIM/Lanczos interferometer ancillas do not supply this capability and receive an owned limitation. The check reports observed Hamming-weight leakage, not a chemistry interpretation independent of encoding. A correct mean particle number is insufficient: equal weight on `N=0` and `N=2` has mean one and complete leakage outside `N=1`. Invalid-width or mismatched-population data rejects. Zero observed leakage does not prove pure-state membership or zero underlying leakage.

## Evidence and receipts

Each explicit verification produces one `VerificationReceipt` with its actual `Plan`, Result and construction, selected options, invocation ID, reference and numerical applications. Raw facts precede this verification record, and returned facts may cite it as the completed source. Loading never creates an invocation or repeats a computation.

`result.verify(checks=options)` takes one options record and returns `(receipt, facts)` for every options type, and so do `verify_projected`, `verify_energy_shift` and `verify_number_sector`. `receipt` holds every raw fact of the computation. `facts` is a tuple of `FramedFact` records that cite `receipt`, and except for `LCHSRefinement` it holds the facts that answer the options' `verification_checks`.

| Options type | Methods | Returned `facts` |
| --- | --- | --- |
| `ProjectedVerificationOptions` | Lanczos, FixedGCIM, ADAPT | One fact per selected criterion |
| `EnergyShiftOptions` | Lanczos, FixedGCIM | The energy-shift discrepancy |
| `NumberSectorOptions` | QHD, one-hot encoding | The observed number-sector leakage |
| `AdaptVerificationOptions` | ADAPT | One fact per scalar of the selected checks |
| `LCHSVerification` | LCHS | The reference discrepancy, and for `ivp_closed_form` the reference consistency |
| `LCHSRefinement` | LCHS | Refined output-error components for `result.assess`, which answer no check |
| `QLSVerification` | QLS | The top-level fact of each selected comparison |
| `QPEVerification` | QPE | `component_error` and `overlap_deficit` |
| `QHDVerification` | QHD | `grid_minimum` and the infidelity of each selected fidelity comparison |

`result.verify(checks=QHDVerification(comparisons=("grid_minimum",)))` evaluates the objective on the finite grid, including its constant term. The returned gap is the candidate's gap to the least of these freshly evaluated binary64 values. It is a diagnostic of that evaluated grid and does not by itself bound the gap for the exact mathematical objective. The verification record also reports `most_probable_gap`, the same gap for the most probable point, which the augmented-Lagrangian and refinement layers read by default ([QHD guide](algorithms/qhd.md#explicit-verification)). The comparison can also report `minimum_success_mass`, the observed weight of grid points whose evaluated objective differs from the evaluated minimum by at most `minimum_tolerance`. With zero tolerance, this counts ties in the evaluated values. The mass is computed from a kept state (`keep_state=True`), native exact probabilities or counts, and a classical Result without a kept state reports it as unavailable.

A `Certificate` can attach completed verification facts to an existing `ClaimAssessment`. It does not rerun verification, update the experiment or turn numerical evidence into a theorem. Generic error and norm quantities reject negatives at their domain owner. A separately justified roundoff transformation must keep the raw value and disclose its use.

For the projected example, keep its selected options and returned facts together, then consume them with the actual Result context:

```python
from nwqlib.evidence import Certificate

certificate = Certificate(plan_id=result.plan_id, result_id=result.content_id,
    assessment=result.assess(absolute_tolerance=1e-10), checks=())
certificate = certificate.with_verification(result, options=projected_options,
                                             evidence=projected_facts)
print(tuple(check.status for check in certificate.checks))
```

A check PASS concerns its selected scalar threshold. The aggregate accuracy assessment can independently remain INCONCLUSIVE. See the [evidence API](api/evidence.md).

### Facts belong to the options that produced them

Each returned fact with a concrete value records, in `fact.evidence.options_id`, the content identity of the complete options record that produced it. A fact whose value is unavailable carries no evidence or identity and attaches as INCONCLUSIVE. `with_verification` compares the recorded identity with the options you pass and accepts the fact only when they are the same record. The identity covers every field and the record's revision history (`parent_id`), so keep the options record that you passed to `verify`, or its saved JSON. Options and facts saved as JSON and read back still attach. A record with the same values but another revision history, for example one made by `revise` instead of the constructor, has a different identity. Changing any field, including only the threshold or tolerance, also gives a new identity, and the existing facts then no longer answer the check. Attaching them raises `ValueError`. Its message reports an options identity mismatch, or a frame mismatch when the changed option is part of the check's frame, such as a number-sector particle count.

A fact that answers a selected check also records, in `fact.evidence.check_id`, the content identity of that exact `CheckSpec`. The direct `assemble_check(check, artifact_id=..., fact=...)` form compares it as well, so a `CheckSpec` revised after verification, for example with another threshold, raises `ValueError` there too. Output-error facts from `LCHSRefinement` answer no check and carry no check identity.

Attachment never runs verification, a reference solver or a backend to fill that gap. To use different options, run the verification again with them and attach the new facts:

```python
stricter = projected_options.revise(tolerance=1e-12)
# certificate.with_verification(result, options=stricter, evidence=projected_facts)
# raises ValueError: those facts were produced by projected_options.
stricter_receipt, stricter_facts = verify_projected(result, options=stricter)
certificate = certificate.with_verification(result, options=stricter,
                                             evidence=stricter_facts)
```

You decide whether that new verification is worth its cost, and the cost depends on the check. The projected, energy-shift and number-sector checks read values that the Results already store, so repeating them only scans those records. ADAPT state checks rebuild the processed Ritz state, from saved vectors for a classical Result or by simulating the reference preparation and selected generators for a quantum-execution Result, and then apply the residual or sector operators. The LCHS `expm`, `ivp`, `closed_form` and `ivp_closed_form` checks compute a new classical reference solution, using a matrix exponential, an IVP integration or both, within the size and RHS limits in their options. `selected_grid` evaluates the saved finite recipe again, except in the initial and zero reconstruction modes, where its reference is the initial vector. A QLS verification solves the normalized original system once unless `spectral_domain` is its only comparison, and computes a missing spectrum only for `spectral_domain`. The QPE check diagonalizes the dense target unless the Run already holds its eigensystem. The QHD `grid_minimum` comparison evaluates the objective at every grid point, and each QHD fidelity comparison runs one restricted evolution.

All exact scalar comparisons use the finite `ExactArithmetic.max_integer_bits` guard. This is an arithmetic representation boundary, not an execution permission or a process-memory limit. Extra simulation, reference decompositions and large state operations still require their explicit scientific selection and appropriate cost approval.
