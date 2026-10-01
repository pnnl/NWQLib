# Accuracy and verification

<a id="evidence-and-explicit-verification-api"></a>

Check a Result against an accuracy tolerance, run a Method's verification checks, and keep both together in a `Certificate`. The [Check accuracy and verify a result](../verification.md) guide explains each check and what it costs.

```python
from nwqlib.evidence import Certificate
from nwqlib.evidence.verification import ProjectedVerificationOptions
```

Both operations read what a Result already holds: its error bounds, `result.facts`, and its stored data. Neither runs the Method again, and only an explicit `verify` call computes new check values.

## Check a result against a tolerance

[`result.assess`][nwqlib.core.analysis.Result.assess] combines the Result's error bounds and returns a [`ClaimAssessment`][nwqlib.evidence.ClaimAssessment] whose `status` answers whether the error is within the tolerance:

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import Lanczos

problem = Eigenproblem(A=[[1.5, -1], [-1, 0.5]])
method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
result = solve(problem, method=method, seed=7)

total = result.assess(absolute_tolerance=1e-6)
print(total.status)
print(total.remaining)
sampling = result.assess(absolute_tolerance=1e-6, component="sampling")
print(sampling.status, sampling.covered_subtotal.numerator)
```

```text
INCONCLUSIVE
('native_preparation', 'native_simulation', 'projected_solve', 'ground_identification', 'physical_model')
PASS 0
```

The total-error criterion is INCONCLUSIVE because five of the error sources that Lanczos lists have no bound. INCONCLUSIVE does not mean the error exceeds the tolerance. With exact readout and no random draw, the sampling error is zero by definition, so the sampling component passes. A component criterion never covers total error. A relative criterion, `result.assess(relative_tolerance=0.05, reference=reference)`, needs a [`TargetReference`][nwqlib.evidence.TargetReference] that sets the scale.

Each entry of `result.facts` is a [`FramedFact`][nwqlib.evidence.FramedFact]: a value with its frame and its evidence. The frame, an [`ErrorFrame`][nwqlib.evidence.ErrorFrame], names the quantity, metric, unit, scope and conditions the value refers to. The kind of its [`Evidence`][nwqlib.evidence.Evidence] says whether the value is a proof, a certified bound, a numerical estimate, an observation or an assertion, and the value is witnessed when its evidence names the record of the computation that produced it. Only witnessed proofs and certified bounds without open assumptions can make an assessment pass.

## Run a verification check

`result.verify(checks=options)` runs the checks that one options record selects and returns `(receipt, facts)`. `receipt` is the [`VerificationReceipt`][nwqlib.execution.VerificationReceipt] of this computation, with its raw values and numerical calls, and `facts` holds one [`FramedFact`][nwqlib.evidence.FramedFact] per check that cites `receipt`. [`Result.verify`][nwqlib.core.analysis.Result.verify] accepts these options records:

| Options record | Methods | Returned `facts` |
| --- | --- | --- |
| [`ProjectedVerificationOptions`][nwqlib.evidence.verification.ProjectedVerificationOptions] | Lanczos, FixedGCIM, ADAPT | One per selected criterion |
| [`EnergyShiftOptions`][nwqlib.evidence.energy_shift.EnergyShiftOptions] | Lanczos, FixedGCIM | The energy-shift discrepancy |
| [`NumberSectorOptions`][nwqlib.evidence.sector.NumberSectorOptions] | QHD with the one-hot encoding | The observed number-sector leakage |
| [`AdaptVerificationOptions`][nwqlib.algorithms.gcim.adapt_verification.AdaptVerificationOptions] | ADAPT | One per scalar of the selected checks |
| [`LCHSVerification`][nwqlib.algorithms.lchs.verification.LCHSVerification] | LCHS | The reference discrepancy, and for `ivp_closed_form` the reference consistency |
| [`LCHSRefinement`][nwqlib.algorithms.lchs.refinement.LCHSRefinement] | LCHS | Refined output-error components for `result.assess`, which answer no check |
| [`QLSVerification`][nwqlib.algorithms.qls.verification.QLSVerification] | QLS | The top-level fact of each selected comparison |
| [`QPEVerification`][nwqlib.algorithms.qpe.records.QPEVerification] | QPE | `component_error` and `overlap_deficit` |
| [`QHDVerification`][nwqlib.algorithms.qhd.records.QHDVerification] | QHD | `grid_minimum` and the infidelity of each selected fidelity comparison |

The projected, energy-shift and number-sector checks read values the Results already store. The verification guide defines what each one computes and what it does not show: [projected quantities](../verification.md#projected-quantities), [two-result energy shift](../verification.md#two-result-energy-shift) and [measured number-sector leakage](../verification.md#measured-number-sector-leakage). The other checks can compute a reference solution, and the guide's [cost table](../verification.md#cost-of-each-check) states what each one costs.

## Keep an assessment and its checks together

A [`Certificate`][nwqlib.evidence.Certificate] holds one assessment and the checks attached to it with [`with_verification`][nwqlib.evidence.Certificate.with_verification]. Continuing the example above:

```python
from nwqlib.evidence import Certificate
from nwqlib.evidence.verification import ProjectedVerificationOptions

options = ProjectedVerificationOptions(
    name="projected",
    comparisons=("gram_hermiticity", "gram_psd_deficit"),
    tolerance=1e-10,
)
receipt, facts = result.verify(checks=options)
certificate = Certificate(plan_id=result.plan_id, result_id=result.content_id,
                          assessment=total, checks=())
certificate = certificate.with_verification(result, options=options,
                                            evidence=facts)
for check in certificate.checks:
    print(check.status, check.fact.fact.quantity)
print(certificate.assessment.status)
```

```text
PASS projected.gram_hermiticity
PASS projected.gram_psd_deficit
INCONCLUSIVE
```

Both checks pass their own threshold, and the assessment stays INCONCLUSIVE, because a check concerns its own scalar only. The returned `facts` attach only with the options object that produced them. To use another tolerance, run `result.verify` again with the new options.

Exact comparisons check the bit length of each numerator and denominator against `max_integer_bits`, default 4096, before every operation, and refuse rather than round ([engineering constants](../ENGINEERING_CONSTANTS.md#exact-evidence-integer-representation)). This limits the size of exact numbers, not time or memory.

## Verification options

::: nwqlib.evidence.verification.ProjectedVerificationOptions
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.verification.verify_projected
    options:
      heading_level: 3

::: nwqlib.evidence.verification.ProjectedDiagnostics
    options:
      heading_level: 3

::: nwqlib.evidence.energy_shift.EnergyShiftOptions
    options:
      heading_level: 3
      members:
        - for_result

::: nwqlib.evidence.energy_shift.verify_energy_shift
    options:
      heading_level: 3

::: nwqlib.evidence.sector.NumberSectorOptions
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.sector.verify_number_sector
    options:
      heading_level: 3

## Verification records

::: nwqlib.execution.VerificationReceipt
    options:
      heading_level: 3
      members: false

::: nwqlib.execution.KernelApplication
    options:
      heading_level: 3
      members: false

## Assessments and certificates

::: nwqlib.evidence.ClaimAssessment
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.TargetReference
    options:
      heading_level: 3

::: nwqlib.evidence.Certificate
    options:
      heading_level: 3

::: nwqlib.evidence.CheckAssessment
    options:
      heading_level: 3

## Error values

::: nwqlib.evidence.FramedFact
    options:
      heading_level: 3

::: nwqlib.evidence.ErrorFrame
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.Fact
    options:
      heading_level: 3

::: nwqlib.evidence.Evidence
    options:
      heading_level: 3

::: nwqlib.evidence.WorkProvenance
    options:
      heading_level: 3

## Error models and checks

A Method states its error sources in an `ErrorModel` and its checks as `CheckSpec` records. These entries serve readers who build or inspect them directly.

::: nwqlib.evidence.ErrorModel
    options:
      heading_level: 3

::: nwqlib.evidence.ErrorTerm
    options:
      heading_level: 3

::: nwqlib.evidence.CheckSpec
    options:
      heading_level: 3

::: nwqlib.evidence.CheckDomain
    options:
      heading_level: 3

::: nwqlib.evidence.assemble_check
    options:
      heading_level: 3

## Variance and sample size

::: nwqlib.evidence.linear_variance
    options:
      heading_level: 3

::: nwqlib.evidence.EstimatorContribution
    options:
      heading_level: 3

::: nwqlib.evidence.CrossCovariance
    options:
      heading_level: 3

::: nwqlib.evidence.IndependenceLaw
    options:
      heading_level: 3

::: nwqlib.evidence.VarianceAssessment
    options:
      heading_level: 3

::: nwqlib.evidence.resolve_scalar_bound
    options:
      heading_level: 3

::: nwqlib.evidence.ScalarBoundResult
    options:
      heading_level: 3
