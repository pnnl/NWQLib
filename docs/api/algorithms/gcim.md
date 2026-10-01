# GCiM and ADAPT API

The `Eigenproblem` owns the Hermitian target. `FixedGCIM` owns normalized trial columns, and `ADAPT` owns the initial state, generator pool and stopping controls. Their Results report projected Ritz values. See the [GCiM guide](../../algorithms/gcim.md) for the complex H/S relation and the adaptive screening rule, and its [source and code map](../../algorithms/gcim.md#source-and-code-map) for the paper location behind each step.

::: nwqlib.algorithms.gcim.fixed_basis.FixedGCIM

::: nwqlib.algorithms.gcim.fixed_basis.FixedGCIMResult

::: nwqlib.algorithms.gcim.fixed_basis.ProjectedPencil

::: nwqlib.algorithms.gcim.adapt.ADAPT

::: nwqlib.algorithms.gcim.adapt_records.ADAPTResult

::: nwqlib.algorithms.gcim.adapt_records.AdaptRound

::: nwqlib.algorithms.gcim.adapt_verification.AdaptVerificationOptions

::: nwqlib.algorithms.gcim.chemistry.build_gcim_chemistry_problem
