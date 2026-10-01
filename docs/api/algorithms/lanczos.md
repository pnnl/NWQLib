# Lanczos

<a id="lanczos-api"></a>Use `Eigenproblem(A=...)` with `Lanczos` to estimate the smallest eigenvalue of A from Chebyshev moments `<psi|T_k(K)|psi>` of the shifted and rescaled operator K. Configure the initial state and the trial dimension on `Lanczos`, and supply the operator through `Eigenproblem`. The result is a projected Ritz value, which does not establish the smallest full-space eigenvalue.

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import Lanczos, LanczosResult, SensitivitySampling
```

The construction follows Kirby, Motta and Mezzacapo, arXiv:2208.00567v4, Sections 2.1–3.1. The shift and rescaling follow Oumarou et al., arXiv:2603.15552v1, whose sensitivity allocation NWQLib adapts, and the thresholded pencil solve follows Epperly, Lin and Nakatsukasa, arXiv:2110.07492v2, Algorithm 1.1. The [Lanczos guide](../../algorithms/lanczos.md) explains the cutoff analysis and sensitivity sampling, and its [source and code map](../../algorithms/lanczos.md#source-and-code-map) gives the equation, page and implementing function of each step.

## Configure the method

::: nwqlib.algorithms.lanczos.method.Lanczos
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lanczos.method.SensitivitySampling
    options:
      heading_level: 3

## Read the result

::: nwqlib.algorithms.lanczos.records.LanczosResult
    options:
      heading_level: 3
      members:
        - projected_diagnostics

::: nwqlib.algorithms.lanczos.records.MomentStatistics
    options:
      heading_level: 3
      members: false

`solve` calls `plan` and the other hooks of a Method. [Extending NWQLib](../extending.md) describes them.
