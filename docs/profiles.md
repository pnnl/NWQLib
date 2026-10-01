# Stored profiles and forecasts

`nwqlib.estimate(plan, profile=profile, allocation=allocation)` reads the existing selected Program and supplied device models. It returns a `PlanEstimate` with the whole logical resource fold, original `DeviceProfile` and `Allocation`, one UTC assessment time, and `ProfileAssessment` records for actual resolved points. `unpredicted` describes controller or range-axis points that have not been chosen. There is no native construction, simulation, pilot, fitting or provider refresh. Without a profile or allocation, `estimate(plan)` returns the logical `WorkloadEstimate` directly. An allocation alone is kept without inventing a physical prediction. Allocation metadata never configures ranks, threads or devices.

Both `estimate` and `compare` accept `assessed_at`, `facts`, `reference` and `max_assessments`. Supplied `facts` are immutable `FramedFact` records; `reference` is a `TargetReference`. Accuracy evidence requires a profile, the selected error model and an Accuracy criterion. Each actual point validates the supplied frame, subject and parameter restrictions; a fact for one point cannot be relabeled to cover another candidate. Comparison rows retain their supplied facts and reference. Inputs that have no assessment consumer are rejected: profile-free logical folding has no assessment time or accuracy evidence, and an allocation alone creates no accuracy assessment.

For one existing point, use the same assessment owner:

```python
from nwqlib.backends import assess

assessment = assess(
    plan, plan.resolve(plan.experiments[0].name),
    profile=profile, allocation=allocation,
    context=resource_context,
    assessed_at=historical_time,  # optional aware datetime for reproducibility
)
```

Omitting `assessed_at` records UTC now once per explicit invocation. An aggregate estimate or comparison reuses that time for every point. Expiry remains meaningful; no network refresh occurs. Optional `runtime=RuntimeOptions(...)` identifies an already known actual backend seed for a single point. Whole-Plan forecasts do not guess future seeds; only explicitly seed-independent models can apply there.

## Independent axes and original associations

| Field | Meaning |
| --- | --- |
| `applicability` | Explicit scoped method-premise evidence. A descriptor or successful planning alone supplies no affirmative scientific predicate. |
| `capability` | Supplied target support for the actual artifact, Program nodes, readout, instructions, kernel and control/adjoint subset. |
| `capacity` | Matching per-location stocks against simultaneous footprints and independent known lower requirements. |
| `time` | Named compatible model predictions and matching scoped consumption limits. |
| `accuracy` | The common `ErrorModel` assessment for this selected point and criterion; `error` keeps the complete `ClaimAssessment`. |

Each axis keeps individual facts and has status `feasible`, `infeasible`, `conditional` or `unknown`. A failure wins within its axis; otherwise conditions, unknowns and complete scoped support take that order. A memory failure does not erase a supported mathematical bound. Excessive sufficient error bounds remain `INCONCLUSIVE`, rather than proving that the actual error is excessive.

Absent or ambiguous applicability evidence remains unknown. Existing `Plan.requirements` and assumptions remain explicit conditions. A bare `scientific_applicability=True` flag encodes no independent mathematical predicate, and method admission does not produce one. The builtin Methods record no framed applicability statement, so their applicability axis is unknown, or conditional when the Plan lists assumptions or requirements. Explicit framed statements still require matching problem, output frame, method source, construction and parameter restrictions. Source identity does not strengthen asserted, empirical or numerical evidence.

The default assessment criterion is `plan.selection_accuracy` when present. Without one, accuracy stays unknown and `error` is `None`. Explicit low-level `accuracy=` changes this assessment only. In particular a selected sampling component and its confidence never become a total-error promise, and assessing a new criterion never changes the original shots or selection.

`AssessmentPoint` binds problem, Plan, Realization, base and selected construction, resource fold/context, profile, allocation, expected runtime/compiler sources, optional known runtime choices and assessment time. `ProfileAssessment` keeps the actual Realization and effective bindings. Its optional `ClaimAssessment` has no observation or result identity before acquisition. Supplied framed `facts` and `TargetReference` reach the common error model unchanged. One point performs one selection and fold, then consumes those facts without a second selection.

`ProfileAssessment.validate_context(plan, profile, allocation, runtime)`, which `assess` calls before returning, checks original identities, model order, evidence, uncertainty and accuracy-summary associations without reevaluation. Predictions from another point under the same profile cannot be exchanged. Changed scientific evidence cannot keep the former accuracy summary. These checks establish known-record association, not authenticity or calibration.

## Device and physical-model domains

`BackendTarget` is the portable target record used by adapters. `None` means an unknown supported subset; an empty tuple explicitly supports none. Instructions name a primitive or exact implementation source, transform and width. A coarse STATEVECTOR capability does not certify every instruction or readout. Builtin target declarations import no SDK and are not live device inventory.

`DeviceConfiguration` binds target, hardware, runtime, build, compiler, precision and representation. `DeviceProfile` adds source evidence, timestamps, advertised stocks and finite time models. `Allocation` separately names granted locations, topology and stocks. Its configuration ID must match before applying its stocks. ResourceContext capacities must refer to those same grants. No device stocks are pooled: two 8-GiB devices cannot satisfy a 10-GiB peak on one device.

Readout dimension uses all selected circuit quantum registers, independently of peak live width. Pauli label width, probability positions and actual classical counts layout are checked before predictions. Unresolved widths remain unknown; cardinality is bounded before forming a power. Native acquisition limits remain at their execution owner.

A complete declared 5-GiB serial workspace fits a matching 8-GiB stock in that scope. Parallel calls can require 10 GiB. Unknown extra workspace preserves known subtotals but prevents a complete-fit conclusion. An upper envelope above stock is conditional; an exact lower requirement above stock proves that failure. Always-resident inputs, analysis, I/O, materialization and stored bytes are simultaneous. A stored-byte stock is distinct from total memory. Five known stored bytes exceed a four-byte stored grant even if total workspace is unknown.

Variable per-acquisition peaks with `required_schedule="serial_acquisitions"` require the supplied serial schedule. Assessment does not change scheduling to make them fit. Resident-only invariant subtotals preserve their separate meaning. For an unpartitioned statevector, the body needs `16 * 2**q` bytes in complex128 or `8 * 2**q` in complex64; a density-matrix body replaces `2**q` by `4**q`. Bit-length comparison can prove an oversized requirement without forming the power or allocating a state. Unknown ancillas do not erase a known system lower bound. Body and workspace are not added without a disjoint-allocation relation. Distributed placement and tensor contraction need their actual structure; qubit count alone supplies neither model.

## Finite time models

The supported `acquisition_linear/1` model is

```text
seconds = c_invocation * invocations + c_shot * sampled_shots
        + c_exact * exact_evaluations + c_operation * logical_operations
```

Coefficient units are `s/invocation`, `s/shot`, `s/exact_evaluation` and `s/logical_operation`. Finite nonnegative coefficients are supplied explicitly; no persisted callable, parser, extrapolation or online fitting runs. Domain checks include configuration/allocation identity, basis, precision, synthesis, schedule, direct or batch acquisition, readout/population, effective bindings, width, operations, output size and primitive/kernel/transform mix. Separate bounds cover resets, measurements, classical work and adaptive rounds. Adding resets invalidates a model calibrated for zero resets even when gate counts match.

Direct non-batch readout means one invocation and one grouped exact evaluation or the requested sampled population, although the raw Program fold has zero terminal-batch counts. Several Pauli labels are returned statistics, not multiple exact evaluations. Selected batch counts come from its actual effective fold. Unknown orchestration does not become one job or a guessed trajectory count.

For example, a direct exact acquisition of the one-qubit HZH preparation has one invocation, one exact evaluation and three logical operations. Synthetic engineering coefficients of 0.25 s/invocation, 0.5 s/exact_evaluation and 0.125 s/logical_operation give `0.25 + 0.5 + 3 * 0.125 = 1.125` seconds. Synthetic calibration coefficients of 1, 2 and 0.5 in the same units, with residuals from -0.25 to 0.5 seconds, give 4.5 seconds with interval [4.25, 5]. These are not measured timings of any machine.

Expired, future-dated, foreign or unsupported models keep unavailable predictions with reasons; other axes keep their facts. Engineering coefficients produce conditional numbers under their assumptions. Calibration additionally keeps supplied training, validation and error sources and uncertainty. Neither case acquires calibration data.

Scopes are `selected_acquisition`, `acquisition_overhead`, and `native_call_wall`. The first two exclude planning, compilation, queue delay, analysis and whole-run cost. Native-call wall time covers the synchronous adapter call including waiting and result extraction. Scopes overlap and cannot be summed into a whole-run prediction, currency estimate or universal CPU cost. Matching limits preserve metric, unit and population.

Uncertainty keeps additive residuals in seconds and its original meaning: `future_run_prediction` concerns one future acquisition, `sample_mean_confidence` concerns a mean, and `model_error_envelope` claims no probabilistic coverage. Intersecting the interval with nonnegative elapsed time does not change its kind. No interval becomes an execution guarantee.

## Original forecasts and observed telemetry

Selected comparison rows carry their allocation and original estimate, including blocked rows. A selected row with a `PlanEstimate` passes that same forecast to `Run.forecast`; `Run.allocation` keeps the grant separately. These are preserved in `RunData` and archives. Before each actual submission intent the runtime looks up the existing exact-point assessment and validates its Plan, readout and construction. `ConsumptionEvent.assessment_id` records that original association. An unforeseen adaptive point has no forecast; resume does not reevaluate one.

```python
from nwqlib.backends import align_telemetry

ledger = align_telemetry(run)     # or the actual completed Result
```

Explicit `trace=`, `assessments=`, `receipts=`, `observations=` and optional `AttemptTiming` inputs are also supported. This projection never folds, executes, refreshes a provider, copies histogram values or fits models. It keeps every attempt, including failures and partial collections. Collected observation IDs and actual Result contribution IDs remain distinct.

Only a pre-attempt forecast plus a completed acquisition and exact timing of its matching scope yields the signed residual `observed_seconds - predicted_seconds`. A runtime/compiler mismatch keeps the original forecast with an unavailable residual. Missing scope, missing receipt, failed acquisition and right-censored timing also preserve their reasons. Right censoring is a lower bound for an explicit continuing target/cutoff, not an exact residual or survival model. Duplicate identical supplied timings deduplicate within one attempt; different attempts remain distinct. Timings from another run/attempt/receipt reject.

`PredictionLedger.validate_context(trace=..., assessments=...)` checks the saved projection without reevaluation. `Record.revise` makes a new data version; updated model coefficients never overwrite original predictions. No coverage claim follows from computing a residual.

Forecast validity is evaluated at the original `assessed_at`. Historical forecasts may still be compared with later actual timings. Their residual does not establish that device calibration was valid at execution time. Telemetry preserves this scope in the original prediction assumptions and reads no clock to reinterpret old model validity fields as a target-time domain.

## Action-owned bounds

`max_assessments=4096` bounds actual point-assessment plus model-prediction rows. The whole requested expansion is checked before any model evaluation; exceeding it reports required rows and the cap. It never silently marks requested forecasts unpredicted. The profile-free logical fold is independent of this cap. For `compare`, the cap covers all selected candidates together, including every actual point and each model prediction, rather than restarting for each row. `align_telemetry(..., max_comparisons=100000)` bounds actual forecast/timing pairs. Program graph/integer limits and common exact-evidence integer bounds remain at their mathematical owners. Archive encoding and native inspection have separate byte controls. These controls are not CPU or RSS estimates.
