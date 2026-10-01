# Pauli expectation API

Use `Expectation(state=..., observable=...)` with `ExpectationMethod`. The configured Method selects preparation, inference and optional calibration. Supply shots or a supported sampling-accuracy criterion through the [shared workflow](../workflow.md). The [Expectation guide](../../algorithms/expectation.md) explains exact readout, finite-shot inference and provider estimates.

The guide derives the accuracy-selected shot count from Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 1, Eq. (2.3), p. 15, and states the premises of each binary inference and of the readout calibration model. Its [source and code map](../../algorithms/expectation.md#source-and-code-map) links every step to its source and owning function.

::: nwqlib.algorithms.expectation.ExpectationMethod

::: nwqlib.algorithms.expectation.ExpectationAnalysis

The Result's `value` follows the selected `NormalizedExpectation` or `QuadraticForm` output. Classical execution keeps original dense/CSR/CSC/Pauli matvec access. Missing measurements leave a partial result. `statistics` keeps actual count populations and the selected inference premises, and a quadratic output propagates the physical scale through its point, variance and interval. Original provider records remain in `estimates`, and `provider_output_estimates` supplies their computed output-frame value and standard error. [Reading the Result](../../algorithms/expectation.md#reading-the-result) gives the meaning and frame of each scientific field, including those of the `BinaryEstimate` and `BinaryCorrection` records inside `statistics`.

::: nwqlib.algorithms.expectation.ExpectationStatistics

::: nwqlib.evidence.binary.BinaryInferenceOptions

::: nwqlib.evidence.binary.BinaryReadoutMitigation
