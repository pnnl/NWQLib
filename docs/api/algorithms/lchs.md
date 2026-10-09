# LCHS {#lchs-api}

Use `LinearDynamics(A=..., initial_state=..., time=..., source=...)` with `LCHS` to approximate the solution of `du/dt = -A u + b` at the final time. Choose the physical solution, a state vector, a norm, an observable or samples through the [output quantities](../workflow.md#scientific-inputs-and-output-quantities). The [LCHS guide](../../algorithms/lchs.md) explains finite quadrature, the evolution backends, scale recovery and the error components.

```python
from nwqlib.algorithms.lchs import LCHS, LCHSRefinement, LCHSVerification
```

Every entry on this page imports from `nwqlib.algorithms.lchs`, except the result records `LCHSSamples`, `LCHSProjectedMoments` and `LCHSGroupMoments`, which are defined in `nwqlib.algorithms.lchs.primary_records`, and the coefficient-table records `LCHSQuadraturePlan` and `LCHSPairCompatibility` in `nwqlib.algorithms.lchs.providers`.

## Solve a linear ODE

::: nwqlib.algorithms.lchs.method.LCHS
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lchs.provider_config.ProviderConfig
    options:
      heading_level: 3
      show_bases: false

::: nwqlib.algorithms.lchs.provider_config.ProviderParameter
    options:
      heading_level: 3

## Read the result

::: nwqlib.algorithms.lchs.primary_records.LCHSAnalysis
    options:
      heading_level: 3
      members:
        - solution
        - state_vector

::: nwqlib.algorithms.lchs.primary_records.LCHSSamples
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lchs.primary_records.LCHSProjectedMoments
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lchs.primary_records.LCHSGroupMoments
    options:
      heading_level: 3
      members: false

## Check a result

`LCHSVerification` compares the physical solution with an independent reference, and `LCHSRefinement` evaluates error components that planning leaves unknown. Each runs only when passed to `result.verify(checks=...)`.

::: nwqlib.algorithms.lchs.verification.LCHSVerification
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lchs.refinement.LCHSRefinement
    options:
      heading_level: 3
      members: false

## Inspect the coefficient table

::: nwqlib.algorithms.lchs.providers.resolve_lchs_coefficient_plan
    options:
      heading_level: 3

::: nwqlib.algorithms.lchs.providers.LCHSCoefficientPlan
    options:
      heading_level: 3
      show_signature: false
      members:
        - prep_amplitudes
        - decompose_mps
        - record

::: nwqlib.algorithms.lchs.providers.LCHSQuadraturePlan
    options:
      heading_level: 3
      show_signature: false
      members: false

::: nwqlib.algorithms.lchs.providers.LCHSPairCompatibility
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lchs.provider_config.ResolvedProviderConfig
    options:
      heading_level: 3
      show_signature: false
      show_bases: false

## Lower-level functions

::: nwqlib.algorithms.lchs.time_independent_terms.cartesian_decomposition
    options:
      heading_level: 3

::: nwqlib.algorithms.lchs.inhomogeneous_theory.duhamel_quadrature
    options:
      heading_level: 3

## Limits

- A and the source are constant in time. General time-dependent A or source is not implemented.
- The Hermitian part L of A must be positive semidefinite, or the default `make_l_psd=True` shifts it and restores the growth factor.
- `approximation_tolerance` and the kernel-integral and k-quadrature bounds are component bounds. They do not bound the total error of the physical output, and floating-point, backend and model errors remain separate.
- The default `dense_exact` backend computes its branch matrices classically and accepts at most `max_dense_select_slots=4096` padded addresses, subject to separate work and byte checks.
- Reference checks need a dense A and a physical solution vector.

[Limitations and open work](../../ROADMAP.md#lchs) lists the open items. The [LCHS source map](../../algorithms/lchs.md#source-map) gives the paper, equation and implementing function of each step. To add a method of your own, see [Extending NWQLib](../extending.md).
