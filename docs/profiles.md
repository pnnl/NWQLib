# Check device fit and run time

<a id="stored-profiles-and-forecasts"></a>`nwqlib.estimate(plan, profile=profile, allocation=allocation)` checks a planned algorithm against a machine you describe. It reports whether the state fits in the memory granted, whether the machine supports the circuit and its readout, and how long a time model you supply predicts the measurement takes. It runs nothing, builds no circuit, fits no model and queries no device.

Describe the machine as a `DeviceProfile` and the resources granted to this workload as an `Allocation`. The checks read the counts that [Estimate resources](resources.md) explains.

## Will it fit in memory? {#will-it-fit-in-memory}

A state-vector simulator stores `2**q` complex amplitudes for q qubits. An unpartitioned state vector therefore needs `16 * 2**q` bytes in complex128 and `8 * 2**q` bytes in complex64, and a density matrix replaces `2**q` by `4**q`.

| Qubits q | complex128 state vector | complex64 state vector |
| --- | --- | --- |
| 20 | 16 MiB | 8 MiB |
| 29 | 8 GiB | 4 GiB |
| 30 | 16 GiB | 8 GiB |
| 33 | 128 GiB | 64 GiB |
| 40 | 16 TiB | 8 TiB |
| 50 | 16 PiB | 8 PiB |

An 8 GiB grant therefore holds the state of at most 29 qubits in complex128 and 30 qubits in complex64. The capacity check of an assessment ([next section](#assess-a-plan-against-a-device)) applies this rule as follows:

- It counts the state alone, so it is a necessary requirement, not a sufficient one. Workspace, buffers, copies and readout storage come on top, NWQLib does not bound them, and it adds them to the state only when they are known to occupy separate memory.
- It applies to the exact total width of the circuit on a single-device allocation. When unknown ancillas leave the total width inexact, the exact width of the system register still gives a lower requirement. A width that is `conditional` on a serial schedule ([Read the labels](resources.md#meaning-of-each-quantity)) becomes exact with `context=ResourceContext(batch_schedule="serial")`, and without that context the check stays unknown.
- It compares bit lengths, so it proves that an oversized state does not fit without forming `2**q` or allocating a state.
- A state that fits makes the check `conditional`, because the other allocations are not bounded. A state that does not fit makes it `infeasible`.
- Distributed placement and tensor-network simulation need a description of their structure, and the qubit count alone gives no memory for them. A `tensor` representation leaves the check unknown.
- Memory is never pooled across devices, so two 8 GiB devices cannot satisfy a 10 GiB peak that one device needs.

This rule is NWQLib's and uses the stored representation and precision. The notebook [`examples/resource_estimation_at_scale.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/resource_estimation_at_scale.ipynb) applies the check in "Before committing" and Appendix C, where the state vectors of its QLS, LCHS and QPE plans at 100 system qubits are proved not to fit a declared one-pebibyte (`2**50` bytes) allocation.

NWQLib runs Aer's statevector target (`AER_STATEVECTOR_TARGET`) in double precision, complex128 ([Local Aer](aer.md#shots-seeds-and-limits)). Local execution also checks `ExecutionLimits`, whose defaults accept circuits of at most 20 qubits and allow the simulator 1024 MiB. A local run beyond either default needs `limits=ExecutionLimits(max_simulation_qubits=..., simulator_memory_mb=...)`.

## Assess a plan against a device {#assess-a-plan-against-a-device}

This example describes Aer's state-vector simulator with 8 GiB for the state and a linear time model, then assesses a one-qubit plan. The plan measures `<Z>` of `|1>`, prepared as H, Z, H, which is three logical operations (the default preparation is one X gate, see [blocks](blocks.md)).

```python
from datetime import datetime, timedelta, timezone

from nwqlib import Expectation, estimate, plan
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import (
    AER_STATEVECTOR_TARGET, Allocation, DeviceConfiguration, DeviceProfile,
    ModelDomain, TimeCoefficient, TimeModel)
from nwqlib.core import Limit, Source, Unit
from nwqlib.evidence import Evidence
from nwqlib.operators import ingest_pauli
from nwqlib.problems import ingest_occupation

problem = Expectation(state=ingest_occupation("1", num_qubits=1),
                      observable=ingest_pauli([("Z", 1.0)], num_qubits=1))
method = ExpectationMethod(preparation_choice="hzh")
selected = plan(problem, method=method, seed=7)

# The machine: Aer's exact state-vector simulator in complex128
start = datetime.now(timezone.utc)
until = start + timedelta(days=365)
mine = Source(name="my workstation", version="1",
              domain="declared by the user", reference="none")
declared = Evidence(kind="external_specification", source=mine)
machine = DeviceConfiguration(
    name="workstation Aer", version="1", target=AER_STATEVECTOR_TARGET,
    hardware=mine, build=mine, compiler=mine,
    runtime=Source(name=AER_STATEVECTOR_TARGET.name, version="1",
                   domain="Aer state-vector simulator", reference="none"),
    precision="complex128", representation="statevector")

# The grant: 8 GiB of memory on one device
allocation = Allocation(
    name="8 GiB", configuration_id=machine.content_id,
    locations=("logical_device",), topology="single_device",
    limits=(Limit(stage="execution", metric="memory", kind="capacity_stock",
                  unit=Unit(symbol="byte", dimension="bytes"),
                  value=8 * 2**30, scope="logical_device"),),
    recorded_at=start, valid_until=until, evidence=declared)

# Time: 0.25 s per call, 0.5 s per exact evaluation and 0.125 s per
# logical operation, valid only for workloads inside this domain
domain = ModelDomain(
    configuration_id=machine.content_id, allocation_id=allocation.content_id,
    basis="selected_logical", batch_schedule="unspecified",
    runtime="seed_independent", acquisition="direct_observation",
    population="unconditional", readouts=("pauli_expectation",),
    selection_ids=(), primitive_gates=("h", "z"), min_qubits=1, max_qubits=20,
    min_operations=1, max_operations=10_000, max_shots=0,
    max_exact_evaluations=1, max_readout_items=1, max_resets=0,
    max_measurements=0, max_classical_work=0, max_adaptive_rounds=0)
time_model = TimeModel(
    name="linear", kind="engineering", scope="selected_acquisition",
    coefficients=(
        TimeCoefficient(feature="invocations", seconds_per_unit=0.25,
                        unit="s/invocation"),
        TimeCoefficient(feature="exact_evaluations", seconds_per_unit=0.5,
                        unit="s/exact_evaluation"),
        TimeCoefficient(feature="logical_operations", seconds_per_unit=0.125,
                        unit="s/logical_operation")),
    domain=domain, recorded_at=start, valid_until=until,
    evidence=Evidence(kind="numerical_estimate", source=mine),
    assumptions=("synthetic coefficients, not measured timings",))
profile = DeviceProfile(configuration=machine, models=(time_model,),
                        recorded_at=start, valid_until=until,
                        evidence=declared)

forecast = estimate(selected, profile=profile, allocation=allocation)
assessment = forecast.assessments[0]
for axis in ("applicability", "capability", "capacity", "time", "accuracy"):
    print(f"{axis:14}{getattr(assessment, axis).status}")
state = next(detail for detail in assessment.capacity.details
             if detail.quantity == "native state body")
print(state.fact.value.numerator, "bytes of state,", state.status)
print(assessment.predictions[0].seconds.value, "s predicted")
```

```text
applicability unknown
capability    unknown
capacity      conditional
time          conditional
accuracy      unknown
32 bytes of state, conditional
1.125 s predicted
```

- **Capacity.** The one-qubit state needs `16 * 2**1 = 32` bytes, which fits in 8 GiB. The check stays conditional because the simulator's other allocations are not bounded.
- **Time.** The model predicts `0.25 + 0.5 + 3 * 0.125 = 1.125` seconds for one call with one exact evaluation and three logical operations. A prediction rests on supplied coefficients, so it stays conditional. These coefficients are synthetic, not measured timings of any machine.
- **Capability.** `AER_STATEVECTOR_TARGET` declares its `readouts` but not its `artifacts`, `instructions` and `program_nodes` (the `Program` steps it supports), so this check is unknown. A target that declares them, for example `AER_STATEVECTOR_TARGET.revise(artifacts=..., instructions=..., program_nodes=...)`, lets the check be decided.
- **Applicability.** Built-in methods record no statement that their mathematical assumptions hold, so this check is unknown.
- **Accuracy.** The plan has no accuracy criterion, so nothing is assessed. Pass `accuracy=` to `plan` to set one.

Replace the plan, the grant and the coefficients with your own. For a plan with sampled shots, use `AER_COUNTS_TARGET`, because `AER_STATEVECTOR_TARGET` declares only exact readouts (Pauli expectations, probabilities, amplitudes and trajectories).

`estimate` returns a `PlanEstimate`. It holds the resource estimate of the whole plan (`resources`), the `DeviceProfile` and `Allocation` as supplied, one assessment time in UTC and one `ProfileAssessment` per experiment whose point is fixed (`assessments`). `unpredicted` lists points not chosen yet, such as later rounds of an adaptive method or points of a parameter range, and these get no forecast. Without a profile and an allocation, `estimate(plan)` returns the resource estimate directly. An allocation without a profile is kept, and no physical prediction is made from it. Allocation data never configures ranks, threads or devices.

## Read the five results {#independent-axes-and-original-associations}

Each `ProfileAssessment` answers five independent questions:

| Field | Question |
| --- | --- |
| `applicability` | Is there explicit evidence that the method's mathematical assumptions hold for this problem? A method name or successful planning alone is no such evidence. |
| `capability` | Does the target support this circuit, its `Program` steps, readout, instructions, host kernel and controlled or adjoint operations? |
| `capacity` | Do the simultaneous memory peaks, and known lower requirements such as the state itself, fit what the allocation grants at each location? |
| `time` | What do the supplied time models predict, and do the predictions meet matching time limits? |
| `accuracy` | Does the plan's error model meet the accuracy criterion? `error` holds the complete `ClaimAssessment`. |

Each result has the status `feasible`, `infeasible`, `conditional` or `unknown`, and keeps its individual checks in `details`, each with a status, a reason and, where one applies, the value and the limit it was compared with. One infeasible check makes its result infeasible. Otherwise any conditional check makes it conditional, then any unknown check makes it unknown, and only complete support gives feasible. The five results do not affect each other, so a memory failure does not hide a supported error bound. An error bound that is sufficient but larger than the criterion gives `INCONCLUSIVE`, not proof that the error is too large.

Applicability without evidence, or with ambiguous evidence, is unknown. The plan's `requirements` and assumptions stay explicit conditions. A bare `scientific_applicability=True` flag states no mathematical condition, and passing the method's input checks does not produce one. The built-in methods record no applicability statement, so their applicability is unknown, or conditional when the plan lists assumptions or requirements. An explicit statement (`FramedFact`) counts only when its problem, output frame (the quantity and unit the statement refers to), method source, construction and parameter restrictions all match. The identity of a source does not strengthen asserted, empirical or numerical evidence.

The accuracy criterion is `plan.selection_accuracy` when the plan has one. Without one, accuracy is unknown and `error` is `None`. `assess(..., accuracy=...)` changes only that assessment. In particular, a sampling component and its confidence never become a promise about total error, and assessing a new criterion never changes the plan's shots or choices.

## Describe the machine and the grant {#device-and-physical-model-domains}

- `BackendTarget` declares what a backend supports: `artifacts`, `readouts`, `program_nodes` (`Program` steps) and `instructions`. Each instruction is named by a primitive gate or an exact implementation source, with its transformation and width. `None` means an unknown subset, and an empty tuple supports nothing. A coarse state-vector capability does not certify every instruction or readout. The built-in targets `AER_STATEVECTOR_TARGET` and `AER_COUNTS_TARGET` import no SDK and are not a live device inventory.
- `DeviceConfiguration` fixes the target, hardware, runtime, build, compiler, precision and representation. Its `runtime` source names the target. A time model needs the precision, representation and compiler to be stated.
- `DeviceProfile` adds the specification evidence, the validity window (`recorded_at` to `valid_until`), advertised limits and named time models. New calibration makes a new profile and never rewrites an earlier assessment.
- `Allocation` names the granted locations, their topology (`single_device`, `independent_devices` or `distributed`) and limits. Its `configuration_id` must match the configuration before its limits apply. Each memory limit (`Limit(kind="capacity_stock", ...)`) names one location, and `ResourceContext(capacities=...)` must refer to the same grants.

The capacity check compares memory location by location:

- A complete declared 5 GiB workspace used in sequence fits an 8 GiB grant at that location, while the same calls in parallel can need 10 GiB.
- Unknown extra workspace keeps the known subtotals but prevents a complete-fit conclusion.
- An upper bound above the grant is conditional, while an exact lower requirement above it proves failure.
- Data that stay resident (inputs, analysis, I/O, materialization and stored bytes) count at the same time as everything else.
- A stored-byte limit is distinct from total memory, so five known stored bytes exceed a four-byte stored grant even when total workspace is unknown.
- A peak that needs a serial schedule (`required_schedule="serial_acquisitions"`) fits only under the supplied serial schedule, and the assessment does not change the schedule to make it fit. Subtotals made only of resident data do not need that schedule.

The readout size uses all quantum registers of the circuit, whatever its peak width. Pauli label widths, probability positions and the classical layout of counts are checked before any prediction. An unresolved width stays unknown, and the readout size is bounded before any power is formed. Limits on the measurement itself are checked at execution.

## Finite time models {#finite-time-models}

The supported time model, `acquisition_linear/1`, is

```text
seconds = c_invocation * invocations + c_shot * sampled_shots
        + c_exact * exact_evaluations + c_operation * logical_operations
```

The coefficient units are `s/invocation`, `s/shot`, `s/exact_evaluation` and `s/logical_operation`. Coefficients are finite, nonnegative and supplied explicitly. NWQLib stores no callable, parses no expression, extrapolates nothing and fits nothing. The model is NWQLib's own (`backends.assessment._predict_time`).

A model applies only inside its `ModelDomain`. The domain fixes the configuration and allocation, the gate basis, precision, synthesis and schedule, direct or batch measurement, readout and population, parameter values, width, operations, output size and the mix of gates, kernels and transformations. Separate maxima cover resets, measurements, classical work and adaptive rounds, so a model calibrated for zero resets does not apply once resets are added, even when the gate counts match. The width and the logical operation count must be exact. A plan whose operation count is unavailable, such as the QLS example of [Estimate resources](resources.md), gets an unavailable prediction with that reason.

The counts come from the resource estimate. A plan without a measurement batch counts one invocation and one grouped exact evaluation, or the requested shots, even though its estimate reports zero batch counts. Several Pauli labels returned together are statistics of one evaluation, not several evaluations. A batch takes its counts from its estimate, and unknown orchestration is not taken as one job or a guessed number of trajectories.

For example, a direct exact measurement (no batch) of the one-qubit HZH preparation has one invocation, one exact evaluation and three logical operations. Synthetic engineering coefficients of 0.25 s/invocation, 0.5 s/exact_evaluation and 0.125 s/logical_operation give `0.25 + 0.5 + 3 * 0.125 = 1.125` seconds. Synthetic calibration coefficients of 1, 2 and 0.5 in the same units, with residuals from -0.25 to 0.5 seconds, give 4.5 seconds with interval [4.25, 5]. These are not measured timings of any machine.

`kind="engineering"` coefficients give conditional numbers under their stated assumptions and carry numerical-estimate evidence. `kind="calibrated"` coefficients also keep the supplied training, validation and error sources (`CalibrationReference`) and an uncertainty, with empirical-prediction evidence. Neither kind collects calibration data. An expired, future-dated, foreign or unsupported model leaves its prediction unavailable with reasons, and the other results keep theirs.

The `scope` of a model is `selected_acquisition`, `acquisition_overhead` or `native_call_wall`. The first two exclude planning, compilation, queue delay, analysis and the cost of the whole run. `native_call_wall` covers the synchronous backend call, including waiting and result extraction, and excludes preparation, saving results and the cost of the whole run. Scopes overlap and cannot be summed into a whole-run prediction, a cost in currency or a CPU time that holds everywhere. A time limit is compared only with a prediction of the same metric, unit and population.

`ModelUncertainty` adds residuals in seconds to the prediction and keeps their meaning. `future_run_prediction` concerns one future measurement, `sample_mean_confidence` concerns a mean and does not bound one run, and `model_error_envelope` claims no probabilistic coverage. The interval is `[max(0, s + lower), s + upper]` around the prediction s, and cutting it at zero does not change its kind. No interval becomes a guarantee about execution.

## Assess one experiment and supply error statements {#assess-one-experiment-and-supply-error-statements}

`assess` gives the `ProfileAssessment` of one experiment. Continuing the example above:

```python
from nwqlib.backends import assess

one = assess(selected, selected.resolve(selected.experiments[0].name),
             profile=profile, allocation=allocation)
print(one.time.status)
```

```text
conditional
```

`assess` also takes `context=` (a `ResourceContext`) and `assessed_at=`, an aware `datetime` that makes the record reproducible. Without `assessed_at`, each call records the current UTC time once, and an estimate or comparison of several experiments uses that one time for all of them. The validity windows of the profile, allocation and models are checked at that time, and nothing is refreshed over the network. `runtime=RuntimeOptions(...)` (from `nwqlib.core.planning`) names a backend seed that is already known for one experiment. A forecast for the whole plan does not guess future seeds, so only models with `runtime="seed_independent"` apply there.

`estimate` and `compare` accept `assessed_at`, `facts`, `reference` and `max_assessments`. `facts` is a tuple of `FramedFact` records, each a supplied error statement, such as a bound, with the quantity, metric and unit it refers to and the parameter values it holds for. `reference` is a `TargetReference`, an exact target value or a bound on its magnitude. Accuracy evidence needs a profile, the plan's error model and an `Accuracy` criterion. Each experiment checks the frame, subject and parameter restrictions of the facts, so a fact for one experiment cannot be relabeled to cover another candidate. Comparison rows keep their facts and reference. Inputs that no check would use are rejected. A resource estimate without a profile has no assessment time or accuracy evidence, and an allocation alone creates no accuracy assessment.

## What an assessment records {#what-an-assessment-records}

`assessment.point`, an `AssessmentPoint`, records the content hashes of what the assessment used: problem, plan, `Realization` (the experiment and its parameter values), construction, resource estimate and its context, profile, allocation, the expected runtime and compiler sources, the known runtime choices if any, and the assessment time. `ProfileAssessment` also keeps the `Realization` with its parameter values. Its optional `ClaimAssessment` refers to no observation or result, because nothing has run. Supplied `facts` and `reference` reach the error model unchanged. One experiment is planned and estimated once, and the facts are then used without planning again.

`ProfileAssessment.validate_context(plan, profile, allocation, runtime)`, which `assess` calls before returning, checks that the recorded hashes, model order, evidence, uncertainty and accuracy summary belong together, without recomputing anything. A prediction from another experiment under the same profile cannot be swapped in, and changed scientific evidence cannot keep the former accuracy summary. These checks establish that the records belong together, not that they are authentic or calibrated.

## Compare forecasts with measured times {#original-forecasts-and-observed-telemetry}

Each row of `compare(..., profile=..., allocation=...)` carries the allocation, including a row whose method refused the problem, and each row with a plan also carries its forecast. Preparing a row passes that same forecast to `Run.forecast`, and `Run.allocation` keeps the grant. Both are saved in `RunData` and in archives. Before each submission, the Run looks up the forecast of that exact experiment and checks its plan, readout and construction, and `ConsumptionEvent.assessment_id` records the link. An adaptive point that was not foreseen has no forecast, and resuming does not create one.

For a finished Run prepared from such a row, or its `Result`:

```python
from nwqlib.backends import align_telemetry

timing = align_telemetry(run)
```

`align_telemetry` returns a `PredictionLedger` of every attempt, including failures and partial collections. It also accepts explicit `trace=`, `assessments=`, `receipts=` (preparation records), `observations=` and `AttemptTiming` inputs. It estimates nothing, runs nothing, refreshes no provider, copies no histogram and fits no model. The IDs of collected observations and of the observations a `Result` used stay distinct.

A residual `observed_seconds - predicted_seconds` needs all of the following, and otherwise the comparison keeps its reason and no residual:

- a forecast attached before the attempt started, and a completed measurement,
- an exact timing of the forecast's scope (Aer records `native_call_wall` time for each call),
- the runtime and compiler of the forecast equal to those in the preparation record.

Missing scope, a missing preparation record, a failed measurement and right-censored timing each keep their reason. Right censoring is a lower bound for an explicit continuing target or cutoff, not an exact residual or a survival model. Identical supplied timings are counted once within one attempt, different attempts stay distinct, and timings from another run, attempt or preparation record are rejected.

`PredictionLedger.validate_context(trace=..., assessments=...)` checks a saved `PredictionLedger` without recomputing. `Record.revise` makes a new version of the data, and updated coefficients never overwrite the original predictions. A computed residual makes no claim about coverage.

A forecast's validity is judged at its original `assessed_at`. A historical forecast can still be compared with later timings, but the residual does not establish that the device calibration was valid at execution time. The original prediction keeps this scope in its assumptions, and no clock is read to reinterpret old validity fields.

## Limits on how much work an assessment does {#action-owned-bounds}

`max_assessments=4096` bounds the experiment assessments plus model predictions. The whole requested expansion is checked before any model is evaluated, and exceeding it reports the rows needed and the cap. It never marks requested forecasts as unpredicted without saying so. A resource estimate without a profile does not count against this cap. For `compare`, the cap covers all candidates together, every experiment and every model prediction, rather than restarting for each row. `align_telemetry(..., max_comparisons=100000)` bounds the forecast and timing pairs. Limits on `Program` size and integers, and on the integers of exact error statements, stay with the code that applies them. Archive encoding and circuit inspection have separate byte limits. None of these limits estimates CPU time or memory.
