# Pauli expectation

<a id="pauli-expectation-api"></a>Use `Expectation(state=..., observable=...)` with `ExpectationMethod` to compute the expectation value of a Hermitian observable O in a state psi: the normalized expectation `<psi|O|psi>/<psi|psi>` by default, or the quadratic form `<psi|O|psi>` with `output=QuadraticForm(observable=O)`.

```python
from nwqlib import Expectation, NormalizedExpectation, QuadraticForm, solve
from nwqlib.algorithms import ExpectationAnalysis, ExpectationMethod
from nwqlib.algorithms.expectation import ExpectationStatistics
from nwqlib.evidence.binary import (
    BinaryInferenceOptions,
    BinaryReadoutMitigation,
)
```

The shot count for an absolute sampling tolerance follows Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 2, with a union bound over the measured terms, and [Proposition 1](../../mathematics.md#r1) derives it. The fixed-time interval of `BinaryInferenceOptions(method="hoeffding")` follows Theorem 1, Eq. (2.3), p. 15, of the same paper. The [Expectation guide](../../algorithms/expectation.md) states the assumptions of each inference and of the readout calibration model, and its [source and code map](../../algorithms/expectation.md#source-and-code-map) links each step to its source and implementing function.

## Configure the method

::: nwqlib.algorithms.expectation.ExpectationMethod
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.binary.BinaryInferenceOptions
    options:
      heading_level: 3

::: nwqlib.evidence.binary.BinaryReadoutMitigation
    options:
      heading_level: 3

## Read the result

::: nwqlib.algorithms.expectation.ExpectationAnalysis
    options:
      heading_level: 3
      members:
        - provider_output_estimates

::: nwqlib.algorithms.expectation.ExpectationStatistics
    options:
      heading_level: 3

::: nwqlib.evidence.binary.BinaryEstimate
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.binary.BinaryCorrection
    options:
      heading_level: 3
      members: false

::: nwqlib.evidence.binary.BinaryInterval
    options:
      heading_level: 3
      members: false

`solve` calls `sampling_shots` and the other hooks of a Method. [Choose shots for a target accuracy](../../algorithms/expectation.md#selecting-absolute-sampling-accuracy) states the shot count that `sampling_shots` returns, and [Extending NWQLib](../extending.md) describes the hooks.
