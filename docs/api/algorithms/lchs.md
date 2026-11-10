# LCHS {#lchs-api}

Use `LinearDynamics(A=..., initial_state=..., time=..., source=...)` with `LCHS` to approximate the solution of `du/dt = -A u + b` at the final time. Choose the physical solution, a state vector, a norm, an observable or samples through the [output quantities](../workflow.md#scientific-inputs-and-output-quantities). The [LCHS guide](../../algorithms/lchs.md) explains finite quadrature, the evolution backends, scale recovery and the error components, and [LCHS derivations](../../algorithms/lchs_derivations.md) gives the mathematical relations behind the sampled, joint-generator, analytic and Apel–Berntson options.

```python
from nwqlib.algorithms.lchs import LCHS, LCHSRefinement, LCHSVerification
```

Every entry on this page imports from `nwqlib.algorithms.lchs`, including the nine [route-option records](#route-options), except the result records `LCHSSamples`, `LCHSProjectedMoments`, `LCHSGroupMoments`, `LCHSPairEstimate` and `LCHSFixedShotPrecision`, which are defined in `nwqlib.algorithms.lchs.primary_records`, and the coefficient-table records `LCHSQuadraturePlan` and `LCHSPairCompatibility` in `nwqlib.algorithms.lchs.providers`.

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

## Choose a route and its options {#route-options}

Four fields of `LCHS` choose a route. Each takes the route's name, which stands for the route's record with default options, or the record with the options you set, as in `LCHS(ham_evo=Trotter(steps=4))`. Import the records from `nwqlib.algorithms.lchs`:

```python
from nwqlib.algorithms.lchs import (PTSC, Coherent, DenseExact, MPSCircuit, QSPBlockEncoding, Richardson,
                                    SampledPairs, Trotter, TrotterErrorBudgeted)
```

| Field | Records |
| --- | --- |
| `ham_evo` | [`DenseExact`][nwqlib.algorithms.lchs.route_options.DenseExact], [`Trotter`][nwqlib.algorithms.lchs.route_options.Trotter], [`TrotterErrorBudgeted`][nwqlib.algorithms.lchs.route_options.TrotterErrorBudgeted], [`QSPBlockEncoding`][nwqlib.algorithms.lchs.route_options.QSPBlockEncoding] |
| `init_prep`, `lcu_prep` | [`MPSCircuit`][nwqlib.algorithms.lchs.route_options.MPSCircuit] |
| `estimator` | [`Coherent`][nwqlib.algorithms.lchs.route_options.Coherent], [`SampledPairs`][nwqlib.algorithms.lchs.route_options.SampledPairs] |
| `SampledPairs.product_formula` | [`Richardson`][nwqlib.algorithms.lchs.route_options.Richardson], [`PTSC`][nwqlib.algorithms.lchs.route_options.PTSC] |

The routes `"auto"`, `"direct"`, `"analytic"`, `"plain"` and `"mpf"` have no options and stay names. The [LCHS guide](../../algorithms/lchs.md#route-options) lists the names each field accepts.

::: nwqlib.algorithms.lchs.route_options.DenseExact
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.Trotter
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.TrotterErrorBudgeted
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.QSPBlockEncoding
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.MPSCircuit
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.Coherent
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.SampledPairs
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.Richardson
    options:
      heading_level: 3
      members: false
      show_bases: false

::: nwqlib.algorithms.lchs.route_options.PTSC
    options:
      heading_level: 3
      members: false
      show_bases: false

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

With fixed `shots`, `plan.reconstruction.fixed_shot_precision` says what the chosen coherent route gives up in precision.

::: nwqlib.algorithms.lchs.primary_records.LCHSFixedShotPrecision
    options:
      heading_level: 3
      members: false

With `estimator="sampled_pairs"`, `result.pair_estimate` holds the raw estimates of the physical quadratic form and norm squared with their covering radii, and `sampled_pair_progress(run)` reads the rows that an unfinished Run has measured. [Sampled branch pairs](../../algorithms/lchs_derivations.md#sampled-branch-pairs) derives the scores, radii and bias bounds.

::: nwqlib.algorithms.lchs.primary_records.LCHSPairEstimate
    options:
      heading_level: 3
      members: false

::: nwqlib.algorithms.lchs.analysis.sampled_pair_progress
    options:
      heading_level: 3

::: nwqlib.algorithms.lchs.primary_records.LCHSPairProgress
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
- `approx_tol` and the kernel-integral and k-quadrature bounds are component bounds. They do not bound the total error of the physical output, and floating-point, backend and model errors remain separate. The fixed-step `trotter_pruning` and shared-route phase values that `print(result)` shows are also components ([Pauli pruning](../../algorithms/lchs.md#pauli-pruning), [phase targets](../../algorithms/lchs.md#shared-route-phase-target)).
- For a dense A, the default backend `"auto"` chooses `dense_exact`, which computes its branch matrices classically and, with quantum execution, accepts at most 4096 address slots by default (`DenseExact.max_select_slots`), subject to separate work and byte checks. For an A given as a Pauli sum, Cartesian parts, a band operator or a supplied block encoding it chooses `qsp_block_encoding`, which needs quantum execution and fits its phases at planning.
- `estimator="sampled_pairs"` estimates only `NormSquared`, `QuadraticForm` and `NormalizedExpectation`, with quantum execution and finite shots or a sampling `accuracy` request. Its covering radii hold for the complete planned set of rows.
- Reference checks need a dense A and a physical solution vector.

[Limitations and open work](../../ROADMAP.md#lchs) lists the open items. The [LCHS source map](../../algorithms/lchs.md#source-map) gives the paper, equation and implementing function of each step. To add a method of your own, see [Extending NWQLib](../extending.md).
