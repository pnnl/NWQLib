# LCHS API

Use `LinearDynamics(A=..., initial_state=..., time=..., source=...)` with `LCHS`. Select the physical solution, state, norm or observable through the [output quantities](../workflow.md#scientific-inputs-and-output-quantities). The [LCHS guide](../../algorithms/lchs.md) explains finite quadrature, realization choices, scale recovery and approximation limits. Verification and refinement perform their explicitly selected work.

::: nwqlib.algorithms.lchs.method.LCHS

::: nwqlib.algorithms.lchs.primary_records.LCHSAnalysis

::: nwqlib.algorithms.lchs.verification.LCHSVerification

::: nwqlib.algorithms.lchs.refinement.LCHSRefinement

::: nwqlib.algorithms.lchs.providers.resolve_lchs_coefficient_plan
