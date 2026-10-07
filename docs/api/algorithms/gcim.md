# GCiM and ADAPT

<a id="gcim-and-adapt-api"></a>Use `Eigenproblem(A=...)` with `FixedGCIM(basis=...)` or `ADAPT(initial_state=..., pool=...)` to estimate the smallest eigenvalue of A by solving the projected problem `H f = E S f` in a basis of trial states. `FixedGCIM` uses a basis that you supply, and `ADAPT` grows one from a generator pool. Both results report projected Ritz values, which do not certify the full-space ground energy.

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import ADAPT, FixedGCIM
from nwqlib.algorithms.gcim import build_gcim_chemistry_problem
```

Every object on this page imports from `nwqlib.algorithms.gcim`, except `PairEstimate`, `GroupMoment` and `SampledPairVariance`, which import from `nwqlib.algorithms.gcim.fixed_basis`. The methods, their results, `ProjectedPencil`, `FixedGCIMBasis` and `AdaptVerificationOptions` also import from `nwqlib.algorithms`.

`FixedGCIM` solves the discretized Hill-Wheeler problem of Zheng et al., Phys. Rev. Research 5, 023200 (2023), arXiv:2212.09205v1, Eq. (13). `ADAPT` follows Zheng et al., npj Quantum Information 10, 127 (2024), arXiv:2312.07691v3. The [GCiM guide](../../algorithms/gcim.md) explains the matrix elements, their error bounds and the adaptive screening rule, and its [source and code map](../../algorithms/gcim.md#source-and-code-map) gives the paper location of each step.

## Solve in a fixed basis

::: nwqlib.algorithms.gcim.fixed_basis.FixedGCIM
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.gcim.fixed_basis.FixedGCIMResult
    options:
      heading_level: 3
      members:
        - projected_diagnostics

::: nwqlib.algorithms.gcim.fixed_basis.ProjectedPencil
    options:
      heading_level: 3
      members:
        - coefficients

::: nwqlib.algorithms.gcim.fixed_basis.ScalarEstimate
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.fixed_basis.PairEstimate
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.gcim.fixed_basis.GroupMoment
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.gcim.fixed_basis.SampledPairVariance
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.gcim.fixed_basis.FixedGCIMBasis
    options:
      heading_level: 3
      members:
        - reference

## Grow the basis adaptively

::: nwqlib.algorithms.gcim.adapt.ADAPT
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.gcim.adapt_records.ADAPTResult
    options:
      heading_level: 3
      members:
        - projected_diagnostics

::: nwqlib.algorithms.gcim.adapt_records.AdaptRound
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.adapt_verification.AdaptVerificationOptions
    options:
      heading_level: 3
      members: false

## Build a molecular problem

::: nwqlib.algorithms.gcim.chemistry.build_gcim_chemistry_problem
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.chemistry.GCIMChemistryProblemData
    options:
      heading_level: 3
      show_signature: false
      members:
        - eigenproblem
        - adapt_method
        - to_dict

::: nwqlib.algorithms.gcim.chemistry.closed_shell_reference_occupations
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.chemistry.qubit_operator_to_sparse_pauli
    options:
      heading_level: 3

## Compare with chemistry references and spin sectors

::: nwqlib.algorithms.gcim.chemistry.chemistry_reference_diagnostic
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.chemistry.chemistry_reference_report_section
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.chemistry.correlation_fraction
    options:
      heading_level: 3

::: nwqlib.algorithms.gcim.sector.sector_expectations
    options:
      heading_level: 3

`solve` calls `plan` and the other hooks of a Method. [Extending NWQLib](../extending.md) describes them.
