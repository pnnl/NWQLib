# QLS {#qls-api}

Use `LinearSystem(A=..., b=...)` with `QLS`. Its `solver` chooses the inverse polynomial or an explicitly configured shortcut model. The [QLS guide](../../algorithms/qls.md) explains physical outputs, normalization, spectral assumptions and shortcut restrictions.

```python
from nwqlib.algorithms.qls import QLS, QLSVerification
```

`QLSAnalysis` also imports from `nwqlib.algorithms.qls`. The result records `QLSSamples`, `QLSProjectedMoments` and `QLSGroupMoments` are defined in `nwqlib.algorithms.qls.primary_records`.

`solve(problem, method=QLS())` defaults to quantum execution of `qsvt_inverse` with `epsilon_inv=0.01`, an automatically chosen encoding normalization `alpha` and encoded inverse-gap parameter `kappa_be`, and the physical solution as output. This default runs no reference solve. The residual tolerance `epsilon_inv` sets the inverse polynomial and does not bound the total error of the output.

## Solve a linear system

::: nwqlib.algorithms.qls.method.QLS
    options:
      heading_level: 3
      members: false

## Read the result

::: nwqlib.algorithms.qls.primary_records.QLSAnalysis
    options:
      heading_level: 3
      members:
        - value
        - x
        - alpha
        - kappa

::: nwqlib.algorithms.qls.primary_records.QLSSamples
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qls.primary_records.QLSProjectedMoments
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.qls.primary_records.QLSGroupMoments
    options:
      heading_level: 3
      members: false

## Check a result

::: nwqlib.algorithms.qls.verification.QLSVerification
    options:
      heading_level: 3
      members: false

## Limits

- `epsilon_inv` is the residual tolerance of the inverse polynomial, not a bound on the total error of x. The rounding in circuit execution and the total propagated physical error remain unknown.
- The shortcut solvers give a unit direction modulo global phase. Their classical norm models do not supply the physical magnitude of x.
- Pauli A, and a non-dense A with a supplied encoding, need `kappa`, because QLS computes no singular values for them.
- Classical execution needs a dense A. QLS has no CSR or CSC route.
- The solution vector is a simulator output, not a hardware measurement.

[Limitations and open work](../../ROADMAP.md#qls) lists the open items. The [QLS source map](../../algorithms/qls.md#sources-and-code-map) gives the paper, equation and implementing function of each step. To add a method of your own, see [Extending NWQLib](../extending.md).
