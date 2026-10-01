# Backends, profiles and export

This page covers the backends that run a Plan's circuits, the device profiles that forecast time and memory before a run, and the functions that export circuits as OpenQASM.

```python
from nwqlib.backends import AerBackend, Allocation, DeviceProfile, assess
from nwqlib.io import export_qasm, write_qasm3_file
```

Pass a backend as `backend=` to `solve` or `prepare`. Without one, a quantum Plan runs on `AerBackend()`. The expectation of Z in the state |0> is exactly 1:

```python
from nwqlib import Expectation, solve
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import AerBackend

problem = Expectation(state=[1.0, 0.0], observable=[[1.0, 0.0], [0.0, -1.0]])
result = solve(problem, method=ExpectationMethod(), backend=AerBackend(),
               seed=7)
print(result.value)  # 1.0
```

## Choose a backend

| Backend | Class | Readouts | Needs | Guide |
| --- | --- | --- | --- | --- |
| Qiskit Aer, local | `AerBackend` | Counts, and exact Pauli expectations, probabilities and amplitudes | `nwqlib[aer]` | [Aer](../aer.md) |
| NWQ-Sim, local | `NWQSimBackend` | Counts. CPU statevector also gives exact Pauli expectations, probabilities and amplitudes | A runner built for one backend and method | [NWQ-Sim](../nwqsim.md) |
| NWQ-Sim on a Slurm cluster | `NWQSimSlurmBackend` with `SlurmProfile` | As NWQ-Sim | A runner built at the site and a Slurm account | [Slurm](../slurm.md) |
| IBM Quantum | `IBMRuntimeBackend` | Counts and provider expectation estimates | `nwqlib[ibm]` and a saved IBM account or a token in `NWQLIB_IBM_RUNTIME_TOKEN` | [IBM Runtime](../ibm.md) |
| IonQ QPU | `IonQBackend` | Raw counts | `nwqlib[ionq]` and an API key in `NWQLIB_IONQ_API_KEY` | [IonQ](../ionq.md) |
| Quantinuum H2 through Nexus | `NexusBackend` | Counts | `nwqlib[nexus]` and an existing qnexus login | [Nexus](../nexus.md) |

Every backend except Aer runs jobs that continue after Python exits. Save the Run to a directory and reopen it with `load_run`, as [Continue an interrupted run](../run_archives.md) describes. The cloud and Slurm backends have offline checks only, and each Run states its qualification once. [Choose a backend](../backends.md) compares them, and [Backend adapter contract](../development/execution.md#backend-adapter-contract) states the rules every backend implements.

::: nwqlib.backends.connection.AerBackend
    options:
      heading_level: 3
      members:
        - from_noise_model

::: nwqlib.backends.nwqsim.NWQSimBackend
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.slurm.NWQSimSlurmBackend
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.slurm.SlurmProfile
    options:
      heading_level: 3
      members:
        - script

::: nwqlib.backends.ibm_runtime.IBMRuntimeBackend
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.ionq.IonQBackend
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.nexus.NexusBackend
    options:
      heading_level: 3
      members: false

## Forecast cost and feasibility

A `DeviceProfile` describes a machine and its time models, and an `Allocation` the resources granted to a workload. `nwqlib.estimate(plan, profile=profile, allocation=allocation)` forecasts every point of the Plan and returns a `PlanEstimate`. `assess` forecasts one point. Nothing runs, and the numbers below are synthetic, not measurements of a machine. The one-qubit Plan prepares `|1>` as H, Z, H and reads `<Z>` exactly, which counts one invocation, one exact evaluation and three logical operations, so the engineering model predicts 0.25 + 0.5 + 3 × 0.125 = 1.125 seconds:

```python
from datetime import datetime, timezone

from nwqlib import Expectation, plan
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import (
    AER_STATEVECTOR_TARGET, Allocation, DeviceConfiguration, DeviceProfile,
    ModelDomain, TimeCoefficient, TimeModel, assess,
)
from nwqlib.core import Limit, Source, Unit
from nwqlib.evidence import Evidence
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import ingest_occupation

problem = Expectation(state=ingest_occupation("1", num_qubits=1),
                      observable=ingest_pauli((("Z", 1.0),), num_qubits=1))
selected = plan(problem, method=ExpectationMethod(preparation_choice="hzh"),
                seed=7)

source = Source(name="example profile", version="1",
                domain="synthetic numbers, not a measured machine",
                reference="api/backends.md")
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
end = datetime(2027, 1, 1, tzinfo=timezone.utc)
target = AER_STATEVECTOR_TARGET  # NWQLib's declared Aer statevector target
machine = DeviceConfiguration(
    name="example machine", version="1", target=target, hardware=source,
    runtime=Source(name=target.name, version="1", domain="example runtime",
                   reference="api/backends.md"),
    build=source, compiler=source, precision="complex128",
    representation="statevector")
spec = Evidence(kind="external_specification", source=source)
grant = Allocation(
    name="example grant", configuration_id=machine.content_id,
    locations=("logical_device",), topology="single_device",
    limits=(Limit(stage="execution", metric="memory",
                  unit=Unit(symbol="byte", dimension="bytes"),
                  kind="capacity_stock", value=2**30,
                  scope="logical_device"),),
    recorded_at=start, valid_until=end, evidence=spec)
domain = ModelDomain(
    configuration_id=machine.content_id, allocation_id=grant.content_id,
    basis="selected_logical", batch_schedule="unspecified",
    acquisition="direct_observation", runtime="seed_independent",
    population="unconditional", readouts=("pauli_expectation",),
    selection_ids=(), primitive_gates=("h", "z"),
    min_qubits=1, max_qubits=1, min_operations=0, max_operations=10,
    max_shots=0, max_exact_evaluations=1, max_readout_items=1,
    max_resets=0, max_measurements=0, max_classical_work=0,
    max_adaptive_rounds=0)
model = TimeModel(
    name="example timing", kind="engineering",
    scope="selected_acquisition",
    coefficients=(
        TimeCoefficient(feature="invocations", seconds_per_unit=0.25,
                        unit="s/invocation"),
        TimeCoefficient(feature="exact_evaluations", seconds_per_unit=0.5,
                        unit="s/exact_evaluation"),
        TimeCoefficient(feature="logical_operations", seconds_per_unit=0.125,
                        unit="s/logical_operation")),
    domain=domain, recorded_at=start, valid_until=end,
    evidence=Evidence(kind="numerical_estimate", source=source),
    assumptions=("synthetic coefficients for this example",))
profile = DeviceProfile(configuration=machine, recorded_at=start,
                        valid_until=end, evidence=spec, models=(model,))

forecast = assess(selected, selected.resolve("expectation"),
                  profile=profile, allocation=grant,
                  assessed_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
print(forecast.time.status)                  # conditional
print(forecast.predictions[0].seconds.value)  # 1.125
```

The machine's `target` is a [`BackendTarget`][nwqlib.backends.capabilities.BackendTarget], here the built-in [`AER_STATEVECTOR_TARGET`][nwqlib.backends.targets.AER_STATEVECTOR_TARGET] for exact readouts. An engineering model's prediction is conditional on its stated assumptions. The [profiles guide](../profiles.md) defines the five axes, the memory comparison and the time models.

::: nwqlib.backends.profiles.DeviceProfile
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.profiles.DeviceConfiguration
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.capabilities.BackendTarget
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.targets.AER_STATEVECTOR_TARGET
    options:
      heading_level: 3

::: nwqlib.backends.targets.AER_COUNTS_TARGET
    options:
      heading_level: 3

::: nwqlib.backends.profiles.TimeModel
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.profiles.TimeCoefficient
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.profiles.ModelDomain
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.profiles.ModelUncertainty
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.profiles.CalibrationReference
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.profiles.Allocation
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.assessment.assess
    options:
      heading_level: 3

::: nwqlib.backends.assessment.PlanEstimate
    options:
      heading_level: 3
      members:
        - assessment_for

::: nwqlib.backends.assessment.ProfileAssessment
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.assessment.AxisAssessment
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.assessment.AssessmentDetail
    options:
      heading_level: 3
      members: false

::: nwqlib.backends.assessment.TimePrediction
    options:
      heading_level: 3
      members: false

## Compare forecasts with observed timings

After a Run prepared with a `PlanEstimate` finishes, `align_telemetry(run)` pairs each attempt's forecast with its observed timing.

::: nwqlib.backends.telemetry.align_telemetry
    options:
      heading_level: 3

::: nwqlib.backends.telemetry.PredictionLedger
    options:
      heading_level: 3
      members:
        - validate_context

::: nwqlib.backends.telemetry.AttemptTiming
    options:
      heading_level: 3
      members: false

## Export OpenQASM

`export_qasm` converts any Qiskit circuit to OpenQASM 3 or 2. `write_qasm3_file` and `write_qasm3` write a Plan's circuit (`plan.construction`) as OpenQASM 3 without a quantum SDK, for the constructions listed in [Export OpenQASM](../qasm-streaming.md#writer-subset), and `materialize_qasm3_file` imports such a file back into Qiskit after checking its bytes.

::: nwqlib.io.qasm.export_qasm
    options:
      heading_level: 3

::: nwqlib.io.streaming.write_qasm3_file
    options:
      heading_level: 3

::: nwqlib.io.streaming.write_qasm3
    options:
      heading_level: 3

::: nwqlib.io.streaming.QasmWriteBudget
    options:
      heading_level: 3
      members: false

::: nwqlib.io.streaming.QasmWriteReceipt
    options:
      heading_level: 3
      members: false

::: nwqlib.io.streaming.QasmWriteError
    options:
      heading_level: 3
      members: false

::: nwqlib.io.streaming.QasmPrefix
    options:
      heading_level: 3
      members: false

::: nwqlib.io.materialization.materialize_qasm3_file
    options:
      heading_level: 3

::: nwqlib.io.materialization.QasmMaterializationBudget
    options:
      heading_level: 3
      members: false

::: nwqlib.io.materialization.QasmMaterialization
    options:
      heading_level: 3
      members: false
