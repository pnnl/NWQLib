# QPE

<a id="qpe-api"></a>Use `Eigenproblem(A=...)` with `QCELS(initial_state=...)` or with `SPE`, `RFE` or `RWPE`, which take `initial_state` too. A `SpectralEstimation` instead supplies its own Hamiltonian or unitary and state. Each method estimates an eigenvalue or eigenphase from the Hadamard-test signals `z_p = <psi|U^p|psi>` of the prepared state psi, with `U = exp(-i*tau*H)` for a Hamiltonian H. Each estimate concerns the eigenvalues present in the prepared state and does not identify the ground state.

```python
from nwqlib import Eigenproblem, SpectralEstimation, solve
from nwqlib.algorithms import QCELS, RFE, RWPE, SPE
from nwqlib.algorithms import QPEAnalysis, QPEVerification
```

| Method | Paper | Estimate | `result.interval` |
| --- | --- | --- | --- |
| `QCELS` | Ding and Lin, arXiv:2211.11973v2 | Single-mode complex least-squares fit | None |
| `SPE` | Wan, Berta and Campbell, arXiv:2110.12071v2 | First crossing of a Fourier-filtered CDF, the lowest value in the prepared spectrum | None |
| `RFE` | Kshirsagar, Katabarwa and Johnson, arXiv:2209.11322v3 | Largest sampled Fourier coefficient | None |
| `RWPE` | Granade and Wiebe, arXiv:2208.04526v1 | Mean of a Gaussian random walk, one bit per step | Nominal 95-percent Gaussian model interval |

The [QPE guide](../../algorithms/qpe.md) explains the meaning and units of the result, the choice of estimator and the cost of each.

## Estimators

::: nwqlib.algorithms.qpe.method.QCELS
    options:
      heading_level: 3
      show_bases: false
      members: false

::: nwqlib.algorithms.qpe.method.SPE
    options:
      heading_level: 3
      show_bases: false
      members: false

::: nwqlib.algorithms.qpe.method.RFE
    options:
      heading_level: 3
      show_bases: false
      members: false

::: nwqlib.algorithms.qpe.method.RWPE
    options:
      heading_level: 3
      show_bases: false
      members: false

## Settings shared by the four estimators

::: nwqlib.algorithms.qpe.method._QPEMethod
    options:
      heading_level: 3
      show_root_heading: false
      show_bases: false
      members: false

## Read the result

::: nwqlib.algorithms.qpe.records.QPEAnalysis
    options:
      heading_level: 3
      members:
        - value
        - eigenvalue
        - phase

::: nwqlib.algorithms.qpe.records.QPESample
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qpe.records.QPEInterval
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qpe.records.QCELSFit
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qpe.records.RWPEGaussian
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qpe.records.QPEExposure
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qpe.records.QPEVerification
    options:
      heading_level: 3
      members: false

`solve` calls `plan`, `analyze` and the other hooks of a Method. [Extending NWQLib](../extending.md) describes them.
