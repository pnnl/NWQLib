# Run your own circuit

A Method that builds a circuit from your input, submits it to the selected backend and returns the counts needs a `descriptor` and two hooks, `plan` and `analyze`. The shared lifecycle behind `plan`, `prepare`, `submit` and `Run.wait` lowers the circuit, executes it, charges the work against the Run's limits and attaches the observations to the Result. `nwqlib.estimate` folds the selected blocks without a Method hook, and reports the operations of a supplied circuit as unavailable. Error models, explicit verification and shot selection from an accuracy request have their own optional hooks, `error_model`, `verify` and `sampling_shots`, and registration and `check-method` need a `Registration` record and a `case()` factory. A Method that offers none of these leaves them out. The archive hooks, `save_archive`, `load_archive` and `result_type`, become necessary when a Run is durable or when a Result or Run is saved or loaded, as described [below](#when-the-archive-hooks-are-required). [Add a Method](algorithm_protocol.md) states the complete extension contract.

The Method below prepares a GHZ state on the requested number of qubits, measures every qubit and returns the counts. Its input and its output kind are two Records of its own, and its Result holds the counts.

```python
from typing import ClassVar

from qiskit import QuantumCircuit

from nwqlib import plan, prepare, submit
from nwqlib.algorithms import AlgorithmDescriptor, ApplicabilityError, Method
from nwqlib.blocks import SelectedConstruction, select_preparation
from nwqlib.core import Record
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Experiment, ObservationSpec, Plan
from nwqlib.ir import (
    Allocate, BlockCall, ClassicalValue, Definition, Measure, PortMap, Program, Register, Sequence,
)
from nwqlib.problems.inputs import state_input


# The Problem is the Method's input, here the number of qubits.
class GHZProblem(Record):
    qubits: int = 2

    # nwqlib.plan calls this when the caller gives no output.
    def default_output(self):
        return GHZCounts()


# The output kind, a Record without fields.
class GHZCounts(Record):
    pass


# The Result holds the counts as (bits, count) pairs.
class GHZResult(Result):
    counts: tuple[tuple[str, int], ...]
```

The state preparation is an ordinary Qiskit circuit without classical bits or measurements.

```python
def ghz_circuit(qubits):
    # A Hadamard on qubit 0 and a CNOT from qubit 0 to each other qubit prepare the GHZ state.
    circuit = QuantumCircuit(qubits)
    circuit.h(0)
    for target in range(1, qubits):
        circuit.cx(0, target)
    return circuit
```

The Program places that circuit, as the selected block `body` that `plan` makes from it, between the allocation of a register and its measurement.

```python
def ghz_program(qubits, body):
    # The Program's nodes, keyed by their ids.
    nodes = {
        # Allocate the quantum register q.
        "allocate": Allocate(wire="q"),
        # Call the block on q through the block's one quantum port, "system".
        "body": BlockCall(signature="body", ports=(PortMap(port="system", wire="q"),)),
        # Measure all of q into the classical bits c, qubit i into bit i.
        "measure": Measure(wire="q", result="c"),
        # Run the three nodes in this order.
        "root": Sequence(children=("allocate", "body", "measure")),
    }
    return Program(
        # The Sequence is the Program's root.
        root="root",
        # The register q, the signature of the block the Program calls, and the bits c.
        registers=(Register(name="q", width=qubits),),
        signatures=(body.record.signature,),
        classical=(ClassicalValue(name="c", dtype="bits", width=qubits),),
        definitions=tuple(Definition(id=name, node=node) for name, node in nodes.items()),
    )
```

The Method's `plan` turns the circuit into the block `body`, builds the Program and records both in a Plan, and `analyze` turns the observed counts into the Result.

```python
class GHZMethod(Method):
    # The Method's name and version.
    descriptor: ClassVar[AlgorithmDescriptor] = AlgorithmDescriptor(method="ghz_counts", version="1")

    def plan(self, problem, *, output, execution, shots, rng):
        # ApplicabilityError refuses a request this Method cannot serve.
        if execution != "quantum" or shots is None:
            raise ApplicabilityError("ghz_counts executes its circuit and needs a shot count")
        circuit = ghz_circuit(problem.qubits)
        # state_input admits the circuit as the unitary on the all-zero state, and
        # select_preparation turns it into the selected block "body" that the Program calls.
        body = select_preparation("body", state_input(circuit))
        program = ghz_program(problem.qubits, body)
        # The Plan records what was selected. _bind, at its end, attaches the live block,
        # while the Plan's portable record carries only the block's definition.
        return Plan(
            # The Problem, Method and output objects this hook received.
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            # The root seed and the stream states after planning.
            randomness=rng.snapshot(),
            # The Program and the definition of the block it calls.
            construction=SelectedConstruction(program=program, selections=(body.record,)),
            # One experiment, which asks for counts with the requested shots.
            experiments=(
                Experiment(
                    name="ghz",
                    setting="computational",
                    observation=ObservationSpec(kind="counts", shots=shots),
                ),
            ),
        )._bind(blocks=(body,))

    def analyze(self, plan, data, *, settings):
        # data.observations.chunks holds one chunk per completed experiment, so one here.
        # Its histogram gives the outcome indices, their counts and the readout width. Bit 0
        # of an index is classical bit 0, the rightmost character of a Qiskit bit string.
        (chunk,) = data.observations.chunks
        histogram = chunk.histogram()
        # The identity fields name the Plan, its construction, the observation population
        # and the chunks this reduction used.
        return GHZResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=(chunk.content_id,),
            counts=tuple((format(index, f"0{histogram.width}b"), count)
                         for index, count in zip(histogram.index_list(), histogram.weights.tolist())),
        )
```

`plan`, `prepare` and `submit` then run the Method.

```python
# Choose the Plan, prepare a Run on the default backend, submit it and wait for the Result.
selected = plan(GHZProblem(qubits=3), method=GHZMethod(), shots=100, seed=7)
prepared = prepare(selected)
with prepared.run as run:
    submit(prepared)
    result = run.wait()
print(result.counts)
```

On the default local Aer backend this prints the counts of `000` and `111`, which together account for the 100 shots.

## What each part does

The Problem and the output kind are ordinary Records. `nwqlib.plan` checks that the returned Plan keeps the same Problem, Method and output objects and the stream snapshot taken after planning, so the Method cannot substitute inputs or randomness. A built-in Problem such as `Expectation` or `Eigenproblem` serves the same role when its fields describe your input.

`state_input` admits only a circuit without classical bits, measurements or free parameters. Measurement belongs to the Program's `Measure` node. Lowering inlines the circuit's gates. A supplied circuit carries no cost law and no approximation bound, so the [selected block](blocks.md#selected-action-and-identity) records unknown synthesis cost and error. `state_input` and `select_preparation` each give a supplied circuit a fresh identity, so planning the same circuit twice gives two Plans with different identities. The alternative, a block bound to your own constructor through `SelectedBlock.bind`, requires a complete `SelectedDefinition`, and the archive writer stores only blocks made by the library's own factories.

`Program` and `SelectedConstruction` check the Program's structure when they are built, before any circuit exists, as [Programs](ir.md) describes. Every field of the Plan's portable record enters the Plan's identity, so a different shot count is a different Plan.

`analyze` receives the Run's data after every experiment has an observation. The lifecycle refuses a Result whose Plan or observation identity differs from the Run, or that names a chunk the Run did not acquire. `_summary_lines` on the Result supplies the first lines of `print(result)`.

The Run uses `AerBackend()` when `prepare` receives no backend. Pass `backend=` for another connection, as [Choose a backend](backends.md) describes. `limits=ExecutionLimits(...)` from `nwqlib.execution` changes the Run's circuit, shot and data allowances, and its `max_simulation_qubits`, 20 by default, bounds the width that Aer admits.

## When the archive hooks are required

A Run on local Aer without a `directory` keeps its state in memory and writes no folder, so the Method above runs to its Result without archive hooks. The hooks are called in four situations. A durable Run writes its selection when it is created, and that write calls `save_archive`. A Run is durable when `prepare` receives a `directory`, or when the backend launches work that outlives the process, which IBM Runtime, IonQ, Nexus, Slurm and NWQ-Sim do, and such a Run without a `directory` uses a folder under `~/.nwqlib/runs`. `result.save` calls `save_archive` and needs `result_type`, and `run.save` calls `save_archive`. `load_result(path, method=GHZMethod)` and `load_run(path, backend=..., method=GHZMethod)` call `load_archive`, and reading a saved Result needs `result_type`. `check-method` saves and reloads a temporary Result, so it needs all three. Without the hooks, a durable `prepare` raises `TypeError` before any submission, and `result.save` and `run.save` raise `TypeError` before writing anything. A failed durable `prepare` keeps its folder, so remove it before retrying. Add the three to the class:

```python
class GHZMethod(Method):
    # descriptor, plan and analyze as above

    # The Result class that result.save and a saved Result's reader need.
    result_type: ClassVar[type[Result]] = GHZResult

    def save_archive(self, plan, files):
        from nwqlib.blocks._archive import write_blocks

        # Store the Plan's portable record, and the supplied circuit as a QPY file beside it.
        return dict(plan=files.write_plan(plan), blocks=write_blocks(plan.blocks, files))

    @classmethod
    def load_archive(cls, saved, files):
        from nwqlib.blocks._archive import read_blocks

        fields = saved["plan"]
        # Rebuild this Method's own Records from their stored fields.
        plan = files.read_plan(
            fields,
            problem=GHZProblem.model_validate(fields["problem"]["fields"]),
            method=cls.model_validate(fields["method"]["fields"]),
            output=GHZCounts.model_validate(fields["output"]["fields"]),
        )
        # Attach the live block read back from the stored circuit.
        return plan._bind(blocks=read_blocks(saved["blocks"], plan.construction.selections, files))
```

`files.write_plan` returns the Plan's portable record, in which the Problem, Method and output appear as a type name and their fields, and the archive stores it. `files.read_problem` and `files.read_output` rebuild only the built-in Problem and output kinds, so a Method with its own Records rebuilds them from the stored fields with `model_validate`, as above. A Problem that holds an admitted operator or state stores it with `files.write_operator` or `files.write_state` and reads it back with the matching reader, which is what the built-in writer does. With these hooks the same Method runs on a durable Run, and its Result and Run can be saved and reopened. Each target path must not exist yet:

```python
from nwqlib import load_result, load_run
from nwqlib.backends import AerBackend

# A directory makes the Run durable, and prepare writes its selection through save_archive.
selected = plan(GHZProblem(qubits=3), method=GHZMethod(), shots=100, seed=7)
prepared = prepare(selected, directory="ghz-run")
with prepared.run as run:
    submit(prepared)
    result = run.wait()
# result.save calls save_archive, and load_result calls load_archive.
result.save("ghz-result")
restored = load_result("ghz-result", method=GHZMethod)
```

The saved Run reopens from its folder with a backend and the Method.

```python
# load_run calls load_archive to rebuild the Plan.
with load_run("ghz-run", backend=AerBackend(), method=GHZMethod) as reopened:
    reopened.resume()
    print(reopened.exposure["completed"])
print(restored.counts == result.counts)
```

Resuming the reopened Run submits nothing, because its one experiment is already complete, so the printed completed exposure is still one job of 100 shots, and the restored counts equal the original ones.

## Beyond counts

`check-method` runs an explicit `MethodCase` with an independent oracle against the Method. `error_model`, `verify`, `sampling_shots` and a `Registration` make error sources, explicit checks, accuracy-driven shot selection and discovery available. [Add a Method](algorithm_protocol.md) describes each of these, and its reference Hadamard Method shows the error model, `_summary_lines`, the author case and the registration.
