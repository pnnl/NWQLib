# Local Aer execution

Install `nwqlib[aer]` for local quantum execution. `AerBackend()` is the default backend when `prepare` or `solve` executes a quantum Plan. A classical Plan uses its selected host kernels and rejects a quantum backend.

```python
import nwqlib
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import AerBackend
from nwqlib.execution import ExecutionLimits

problem = nwqlib.Expectation(state=[1., 0.], observable=[[1., 0.], [0., -1.]])
selected = nwqlib.plan(problem, method=ExpectationMethod(), shots=128, seed=7)
prepared = nwqlib.prepare(selected, backend=AerBackend(),
    limits=ExecutionLimits(max_simulation_qubits=2, simulator_memory_mb=256,
                           max_total_circuits=1, max_total_shots=128))
inventory = prepared.inspect_resources(index=0)
with prepared.run as run:
    nwqlib.submit(prepared)
    result = run.wait()
    print(result.value)
```

The selected Plan fixes the output, shot count and randomness. Passing it to `solve` executes that same selection. Changing those choices requires a new Plan. `shots` counts samples per elementary measurement setting, while the Run accumulates actual circuits, shots and attempts. Exact readouts use `shots=None` where the Method supports them. Samples and adaptive counts methods apply their own documented acquisition defaults.

`ExecutionLimits` applies to actual local work. Its default simulator width is 20 qubits and its default simulator memory setting is 1024 MiB. These limits do not constrain symbolic planning or establish a process-memory bound. A statevector simulation still needs exponential state storage even when the output is one scalar or a probability marginal. See [prepared execution](prepared_execution.md) for cumulative limits and continuation.

## Readout and population

The Method selects counts, Pauli expectations, probability marginals or amplitudes to match its requested scientific output. A scalar or marginal readout does not export a full statevector. QLS and LCHS physical-vector outputs use their selected amplitude reconstruction, including the actual recovery scale and original coordinate order. Counts do not recover a complex physical vector.

An exact probability, Pauli-expectation or amplitude readout is checked against the binary64 roundoff window of its preparation receipt, which grows with the receipt's native operation count and circuit width and uses Aer's own per-instruction constant. For a body without control flow, the count includes every native evolution operation that executes and excludes save instructions, while a body with control flow uses its full instruction inventory, including saves. The comment above the constants in `_validation.py` derives that constant from the fusion pass, kernels and gate matrices of qiskit-aer 0.17.2, and [Engineering constants](ENGINEERING_CONSTANTS.md) summarizes the derivation. The receipt's `probability_window_exclusions` names what the derivation does not bound, such as supplied matrix, state or channel instructions, measurement or reset instructions, control flow and the masses that NWQLib derives from amplitudes. An installed Aer release other than 0.17.2 adds "unchecked qiskit-aer version". An exclusion does not change the numeric window.

The Plan's method and backend random streams are separate. `seed=None` records resolved entropy, while a supplied integer records an explicit reproducible selection. Fixed Aer seeds identify a possible repeated population and do not establish independent samples. Reanalysis and reporting do not draw another seed.

Aer executes the circuits that a selected Method constructs. A supplied unitary `QuantumCircuit` enters a workflow as the state it prepares from the all-zero input. For example, pass it as the `state` of an `Expectation` problem. `ExpectationMethod` then selects readout circuits for the requested observable, and `solve` executes them on `AerBackend()`. See [inputs](inputs.md) and [expectation and quadratic form](algorithms/expectation.md).

Pauli readout positions refer to the original instruction sequence. An observation at its length follows those instructions, including measurements. Earlier observations are not moved across measurements. The common amplitude route requires an unconditional pre-final-measurement trajectory. A native control-flow circuit can have unknown measurement and trajectory conditioning, and successful execution does not establish an unconditioned state.

A trajectory schedule observes several points of one coherent state in one simulation. Aer inserts the saves of each point at its boundary before lowering, and no body operation after the last point executes. A readout view of a Pauli or probability point applies its selected tail, saves the view's values and, when a later observation follows, applies the selected exact inverse of that tail. An amplitude point, or a reduction whose reducer is not registered as phase invariant, receives its saved state with the phase of its own logical prefix, not the phase of the whole executed circuit. The preparation stores each correction factor, and the receipt records the envelope of that multiplication (`statevector_roundoff`), which every consumer of the corrected state adds to its state-error budget. Before native preparation the Run admits the live state, every saved result and the transient marginal workspace against `simulator_memory_mb`. The workspace is charged for the thread cap that the simulator is given (`max_parallel_threads`), which the preparation records and restoration reapplies. When a local result returns and its publication is then refused, the Run records the submission as failed and releases its output reservation. The acquisition event stays uncertain.

Trajectory readout evaluates the declared points of one deterministic, noiseless coherent evolution. Probabilities and Pauli expectations are functions of the pure state at each selected boundary, and amplitude outputs also preserve the declared phase convention. An intermediate readout view is reversed before evolution continues. The selected body and views must not introduce measurement-conditioned evolution, resets, postselection, or non-unitary channels. A registered reducer must support the selected state representation and backend. Aer configurations with an explicit noise model support sampled counts, while trajectory readout requires its noiseless pure-state route. Before lowering, Aer refuses a view, or the part of the body up to the last point, that contains a measurement, a reset or a control-flow operation, also inside the definition of a composite instruction such as `Initialize`, or an opaque instruction without a definition. Operations after the last point are not executed and are not checked. After lowering, Aer requires the statevector simulation method.

## Noise

Noise is explicit and applies to sampled counts:

```python
from qiskit_aer.noise import NoiseModel, ReadoutError

noise = NoiseModel()
noise.add_all_qubit_readout_error(ReadoutError([[0.95, 0.05], [0.10, 0.90]]))
backend = AerBackend.from_noise_model(noise)
noisy_result = nwqlib.solve(selected, backend=backend)
```

The backend binds the caller's actual SDK object without copying, serializing or hashing it during preparation. Keep the model unchanged while reusing this backend or its prepared handles. Its UUID identifies the binding rather than the model contents. Counts preparation translates the selected circuit to the noise model's effective native gate basis, and the prepared receipt records that transformation. Raw noisy counts do not supply a bound on physical bias. Exact expectations, probabilities and amplitudes reject a noisy backend before native preparation.

Configuration JSON alone has no executable noise payload. `run.save(path)` stores the bound model, including an unused binding, with public `NoiseModel.to_dict()` data, the model's basis gates and separate NPY arrays. `load_run(path, backend=...)` starts from those basis gates, so later preparations lower to the original Aer target. It rebuilds the quantum and readout errors from that data with Qiskit gate constructors and the saved parameters (`_run_archive._load_noise_model`), and it preserves the original binding identity and cache state. `load_run` binds the rebuilt model to the reopened Run's own copy of the backend configuration (`run.backend`) and leaves the backend passed to `load_run` unchanged. A later Run on that backend therefore uses the model the backend holds, and a configuration rebuilt from JSON stays unbound. Keep the original backend configuration when reopening. A new `from_noise_model(...)` call creates a different binding and does not replace the connection for a saved run. Array dtype, shape and values stay unchanged. See [Run archives](run_archives.md), which also describes how a Run saved without basis gates reopens.

## Inspect actual prepared circuits

`nwqlib.estimate(selected)` folds the selected Program and its resource laws without compiling a circuit. After preparation, `prepared.inspect_resources(index=0)` inventories one existing native circuit without copying it, compiling it again or executing it. The returned mapping names the counted circuit and its native basis. Its `operations` entry maps each top-level operation name to its count, including measurements, barriers, simulator saves, `Clifford` objects and user-defined gates. A composite gate or control-flow operation counts once under its own name, and its definition or body is not expanded. `total_operations`, `num_qubits`, `num_clbits` and `depth` describe the same circuit.

An explicit nonempty `transpile_options` requests an auxiliary compilation of that existing circuit:

```python
prepared = nwqlib.prepare(selected)
with prepared.run:
    native = prepared.inspect_resources(index=0)
    auxiliary = prepared.inspect_resources(index=0,
        transpile_options={"basis_gates": ["u", "cx"], "optimization_level": 0,
                           "seed_transpiler": 7})
```

The auxiliary inventory counts the transpiled copy, from which simulator saves are removed before compilation. Its `compiler` entry records the Qiskit version and the effective options, and the copy does not replace the executable circuit. `max_operations` bounds input and returned instruction counts, and `max_bytes` bounds known inspection collections and options. Compiler expansion, vendor workspace and total process memory remain unknown. These are inspection controls, not authorization to run the circuit. `prepared.circuits` provides detached circuit copies for display, and `prepared.circuit(index)` one such copy. Modifying a copy cannot alter the prepared acquisition, and both inspection operations require an open Run.

The Run's receipts and trace separately record preparation, submission and completed work. A resource estimate or circuit inventory alone proves none of those later events. See [resources](resources.md) and [profiles](profiles.md) for the distinction between logical laws, native inventory and device predictions.

## Native instruction semantics

Aer lowering follows actual operation classes and definitions. Aer applies a dense `UnitaryGate` as its matrix. With a noise model whose gate basis omits that instruction, lowering first replaces each dense `UnitaryGate` outside control-flow blocks by its exact synthesis (`qiskit_compat.exact_dense_unitaries`), and the receipt's compiler names that step. The Run synthesizes each distinct dense matrix once per Run object and keeps its circuit for later preparations (`Run._exact_dense_unitaries`). It reserves each new synthesis against its `max_synthesis_work`, a total over all its preparations, before the synthesis starts, and a reopened Run synthesizes and reserves a matrix again. Later preparations of the same construction also reuse the lowered gate. A custom gate named `measure`, `barrier` or another native gate keeps its own definition. Supported nested definitions are lowered within the existing finite decomposition limit, including control-flow blocks. An unresolved operation rejects before submission instead of being dispatched under a misleading name. Multiplexers with omitted diagonals or shortened control tables use their actual defining circuit.

Final-readout removal respects measurements needed by later classical instructions. Static native measurement counts describe the lowered circuit, separately from shots and simulator trajectories. For control flow, branch and loop execution counts are not inferred from syntax. A possible measurement need not execute, so a counts request can still fail with the backend's missing-readout error. Such unknown counts remain unavailable rather than becoming zero.

Within one Run, preparation can reuse already lowered definitions while keeping their source association. Later submissions do not copy every expanded circuit into a permanent cache. Save and reopen through the common [Run archive](run_archives.md) path to keep selected native data and reached caches without a new synthesis. Reopening a completed Run returns its saved Result, while continuation of unfinished work uses that Run's existing frontier.
