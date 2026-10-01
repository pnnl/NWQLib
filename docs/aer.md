# Local Aer

<a id="local-aer-execution"></a>Aer runs NWQLib's circuits on your machine with Qiskit Aer. Install `nwqlib[aer]` and pass `AerBackend()` to `solve`:

```python
import nwqlib
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import AerBackend

problem = nwqlib.Expectation(state=[1., 0.],
                             observable=[[1., 0.], [0., -1.]])
result = nwqlib.solve(problem, method=ExpectationMethod(), shots=128,
                      seed=7, backend=AerBackend())
print(result.value)  # 1.0: |0> is an eigenstate of Z with eigenvalue 1
```

`AerBackend()` is also the default backend when `solve` or `prepare` executes a quantum Plan, so `backend=` can be omitted. A classical Plan runs its host kernels and rejects a quantum backend. To save a run and reopen it later, use the four steps in [Run on any backend](backends.md#run-on-any-backend).

## Shots, seeds and limits

A Plan from `nwqlib.plan` fixes the output, shot count and randomness, and passing it to `solve` executes that same choice. Changing them requires a new Plan. `shots` counts samples per elementary measurement setting, while the run adds up the circuits, shots and attempts it executes. Exact readouts use `shots=None` where the method supports them. Sampling and adaptive methods apply their own documented defaults for how many circuits and shots they run.

The method and the backend draw from separate random streams of the Plan. `seed=None` records the entropy it resolved, and an integer seed records a reproducible choice. Fixed Aer seeds can repeat the same samples, so they do not establish independent samples. Reanalysis and reporting draw no new seed.

`ExecutionLimits` applies to local execution. Its default simulator width is 20 qubits and its default simulator memory setting is 1024 MiB. These limits do not constrain planning and are not a bound on process memory. A statevector simulation needs exponential state storage even when the output is one scalar or a probability marginal. NWQLib runs the statevector target in double precision, so each amplitude is a 16-byte complex128 number, and leaves the counts target at Aer's default precision. See [Run on a backend](prepared_execution.md) for cumulative limits and continuation.

## Run your own circuit

Aer executes the circuits that a method builds. A unitary `QuantumCircuit` that you supply enters a workflow as the state it prepares from the all-zero input. For example, pass it as the `state` of an `Expectation` problem. `ExpectationMethod` then builds readout circuits for the requested observable, and `solve` executes them on `AerBackend()`. See [inputs](inputs.md) and [Finite Pauli expectation](algorithms/expectation.md).

## Readouts {#readout-and-population}

The method chooses counts, Pauli expectations, probability marginals or amplitudes to match the requested scientific output. A scalar or marginal readout does not export a full statevector. QLS and LCHS physical-vector outputs use their amplitude reconstruction, including the recovery scale and the original coordinate order. Counts do not recover a complex physical vector.

An exact probability, Pauli-expectation or amplitude readout is checked against a binary64 roundoff window that grows with the circuit's native operation count and width, with a per-instruction constant derived for qiskit-aer 0.17.2 ([derivation](ENGINEERING_CONSTANTS.md#numerical-guards-and-tolerances)). Without control flow, the count is the number of native evolution operations that execute, save instructions excluded. With control flow, it is the full instruction count, saves included. The preparation record's `probability_window_exclusions` lists what the derivation does not bound, such as supplied matrix, state or channel instructions, measurement or reset, control flow and the masses that NWQLib derives from amplitudes. An installed Aer release other than 0.17.2 adds "unchecked qiskit-aer version". An exclusion does not change the numeric window.

Pauli readout positions refer to the original instruction sequence. An observation at its length follows those instructions, including measurements. Earlier observations are not moved across measurements. The common amplitude route requires an unconditional trajectory before the final measurements. A circuit with control flow can have unknown measurement and trajectory conditioning, and successful execution does not establish an unconditioned state.

## Trajectory readout on Aer

Aer reads all points of a [trajectory](backends.md#trajectory-readout) in one simulation, and amplitude outputs keep the declared phase convention. It inserts each point's save instructions at the point before building the Aer circuit, and no operation after the last point executes. A Pauli or probability readout applies its tail, saves the values and, when a later point follows, applies the exact inverse of that tail.

An amplitude point, or a reduction whose reducer is not registered as phase invariant, receives its saved state with the phase of its own circuit prefix, not the phase of the whole executed circuit. The preparation stores each correction factor, and the preparation record stores the upper bound of the roundoff of that multiplication (`statevector_roundoff`), which every code path that uses the corrected state adds to its state-error budget.

With a noise model, Aer supports sampled counts only, while trajectory readout requires the noiseless pure-state route. After the circuit is built, a trajectory requires Aer's statevector simulation method. Before preparation, the run checks the live state, every saved result and the temporary marginal workspace against `simulator_memory_mb`. The workspace is counted for the thread limit given to the simulator (`max_parallel_threads`), which the preparation records and a reopened run applies again. When a local result returns but the run refuses to save it, the run marks the submission failed and releases its reserved output bytes, and the measurement itself stays uncertain.

## Noise {#noise}

Noise is explicit and applies to sampled counts:

```python
from qiskit_aer.noise import NoiseModel, ReadoutError

noise = NoiseModel()
noise.add_all_qubit_readout_error(
    ReadoutError([[0.95, 0.05], [0.10, 0.90]]))
backend = AerBackend.from_noise_model(noise)
plan = nwqlib.plan(problem, method=ExpectationMethod(), shots=128, seed=7)
noisy = nwqlib.solve(plan, backend=backend)
print(noisy.value)  # 0.921875
```

Without noise the same Plan gives 1.0.

The backend uses your `NoiseModel` object itself, without copying, serializing or hashing it during preparation. Keep the model unchanged while you reuse this backend or its prepared circuits. The backend's UUID identifies this binding, not the model contents. For counts, preparation translates the circuit to the noise model's effective native gate basis, and the preparation record notes that translation. Raw noisy counts do not bound the physical bias. Exact expectations, probabilities and amplitudes reject a noisy backend before preparation.

Configuration JSON alone holds no executable noise model. `run.save(path)` stores the bound model, also an unused one, as public `NoiseModel.to_dict()` data with the model's basis gates and separate NPY arrays. `load_run(path, backend=...)` starts from those basis gates, so later preparations translate to the original Aer target. It rebuilds the quantum and readout errors from the saved data with Qiskit gate constructors and the saved parameters, and it keeps the original binding identity and cache state. `load_run` binds the rebuilt model to the reopened run's own copy of the backend configuration (`run.backend`) and leaves the backend passed to `load_run` unchanged. A later run on that backend therefore uses the model the backend holds, and a configuration rebuilt from JSON stays unbound. Keep the original backend configuration when reopening. A new `from_noise_model(...)` call creates a different binding and does not replace the connection of a saved run. Array dtype, shape and values stay unchanged. [Continue an interrupted run](run_archives.md) also describes how a run saved without basis gates reopens.

## Inspect prepared circuits {#inspect-actual-prepared-circuits}

`nwqlib.estimate(plan)` adds up the resource formulas of the Plan without compiling a circuit. After preparation, `prepared.inspect_resources(index=0)` counts the operations of one prepared circuit without copying, compiling or executing it:

```python
from nwqlib.execution import ExecutionLimits

problem = nwqlib.Expectation(state=[1., 1.],
                             observable=[[0., 1.], [1., 0.]])
plan = nwqlib.plan(problem, method=ExpectationMethod(), shots=128, seed=7)
limits = ExecutionLimits(max_simulation_qubits=2, simulator_memory_mb=256,
                         max_total_circuits=1, max_total_shots=128)
prepared = nwqlib.prepare(plan, backend=AerBackend(), limits=limits)
with prepared.run as run:
    native = prepared.inspect_resources(index=0)
    compiled = prepared.inspect_resources(
        index=0, transpile_options={"basis_gates": ["u", "cx"],
                                    "optimization_level": 0,
                                    "seed_transpiler": 7})
    print(native["operations"])    # {'h': 2, 'measure': 1}
    print(compiled["operations"])  # {'u': 2, 'measure': 1}
    nwqlib.submit(prepared)
    print(run.wait().value)        # 1.0: |+> is an eigenstate of X
```

The returned mapping names the counted circuit and its native basis. Its `operations` entry maps each top-level operation name to its count, including measurements, barriers, simulator saves, `Clifford` objects and user-defined gates. A composite gate or control-flow operation counts once under its own name, and its definition or body is not expanded. `total_operations`, `num_qubits`, `num_clbits` and `depth` describe the same circuit.

A nonempty `transpile_options` requests an extra compilation of a copy of the prepared circuit, from which simulator saves are removed before compilation. Its `compiler` entry records the Qiskit version and the effective options, and the copy does not replace the circuit that runs. `max_operations` bounds the input and returned instruction counts, and `max_bytes` bounds known inspection collections and options. Compiler expansion, vendor workspace and total process memory remain unknown. These are inspection controls and do not run the circuit. `prepared.circuits` returns copies of all prepared circuits for display, and `prepared.circuit(index)` one copy. Changing a copy cannot change what runs, and both inspection operations require an open run.

The run's preparation records and trace log preparation, submission and completed work separately. A resource estimate or an operation count alone proves none of these later events. See [Estimate resources](resources.md) and [Check device fit and run time](profiles.md) for the difference between formulas, counts of prepared circuits and device predictions.

## Circuit instructions and caching {#native-instruction-semantics}

Aer follows the operation classes and definitions of the circuit. It applies a dense `UnitaryGate` as its matrix. With a noise model whose gate basis omits that instruction, each dense `UnitaryGate` outside control-flow blocks is first replaced by its exact synthesis, once per distinct matrix in a run ([Exact synthesis of dense matrices](backends.md#exact-synthesis-of-dense-matrices)), and the preparation record's compiler entry names that step. Later preparations of the same construction also reuse the translated gate. A custom gate named `measure`, `barrier` or another native gate keeps its own definition. Supported nested definitions are translated within the existing finite decomposition limit, including control-flow blocks. An operation that cannot be resolved is rejected before submission instead of being run under a misleading name. Multiplexers with omitted diagonals or shortened control tables use their own defining circuit.

Removal of final readouts respects measurements needed by later classical instructions. Static native measurement counts describe the translated circuit, separately from shots and simulator trajectories. For control flow, branch and loop execution counts are not inferred from the circuit text. A possible measurement need not execute, so a counts request can still fail with the backend's missing-readout error. Such unknown counts remain unavailable and do not become zero.

Within one run, preparation can reuse definitions it has already translated, keeping their association with the source. Later submissions do not copy every expanded circuit into a permanent cache. To keep translated native data and caches without a new synthesis, save and reopen the run ([Continue an interrupted run](run_archives.md)). Reopening a completed run returns its saved Result, and continuing unfinished work uses that run's saved progress.
