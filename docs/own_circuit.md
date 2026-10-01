# Run your own circuit

To measure an observable on a circuit you built in Qiskit, pass the circuit as the state of a built-in Problem and run a built-in Method. To get another output, such as the raw counts, or to run your own Program, write a Method.

## Measure an observable on your circuit {#measure-an-observable-on-your-circuit}

Pass the circuit as the `state` of `Expectation` and solve it with the built-in `ExpectationMethod`. The circuit is read as the unitary applied to the all-zero state.

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from nwqlib import Expectation, solve
from nwqlib.algorithms import ExpectationMethod

# Your circuit prepares the Bell state (|00> + |11>)/sqrt(2).
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
H = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])

problem = Expectation(state=qc, observable=H)
result = solve(problem, method=ExpectationMethod(), seed=7)
print(result.value)  # 1.5
```

The Bell state has `<ZZ> = 1` and `<XX> = 1`, so the expectation of `ZZ + 0.5 XX` is 1.5. Without `shots`, `ExpectationMethod` evaluates the Pauli expectations exactly. `solve(problem, method=ExpectationMethod(), shots=2000, seed=7)` measures them and also prints 1.5, because both terms are deterministic for this state.

The circuit must have no classical bits, measurements or free parameters. Other inputs that take a state accept a circuit the same way, for example `Lanczos(initial_state=qc)` and the basis states of `FixedGCIM` ([accepted input forms](inputs.md)).

## Write a Method {#write-a-method}

A Method needs a `descriptor` and two hooks. `plan` builds a [Program](ir.md), NWQLib's description of a circuit as named steps, from your input and returns it in a Plan with the experiments to run, and `analyze` turns the observations into a Result.

NWQLib's shared steps `plan`, `prepare`, `submit` and `Run.wait` then build the Qiskit circuit, run it, count the work against the Run's limits and attach the observations to the Result. The Method below prepares a GHZ state on the requested number of qubits, measures every qubit and returns the counts. Its input and its output kind are two Records of its own, and its Result holds the counts.

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
    Allocate, BlockCall, ClassicalValue, Definition, Measure, PortMap,
    Program, Register, Sequence,
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
    # A Hadamard on qubit 0 and a CNOT from qubit 0 to each other qubit
    # prepare the GHZ state.
    circuit = QuantumCircuit(qubits)
    circuit.h(0)
    for target in range(1, qubits):
        circuit.cx(0, target)
    return circuit
```

The Program places that circuit, as the block `body` that `plan` makes from it, between the allocation of a register and its measurement. [Describe a circuit as a Program](ir.md) lists these nodes.

```python
def ghz_program(qubits, body):
    # The Program's nodes, keyed by their ids.
    nodes = {
        # Allocate the quantum register q.
        "allocate": Allocate(wire="q"),
        # Call the block on q through the block's one quantum port, "system".
        "body": BlockCall(signature="body",
                          ports=(PortMap(port="system", wire="q"),)),
        # Measure all of q into the classical bits c, qubit i into bit i.
        "measure": Measure(wire="q", result="c"),
        # Run the three nodes in this order.
        "root": Sequence(children=("allocate", "body", "measure")),
    }
    return Program(
        # The Sequence is the Program's root.
        root="root",
        # The register q, the signature of the block the Program calls,
        # and the bits c.
        registers=(Register(name="q", width=qubits),),
        signatures=(body.record.signature,),
        classical=(ClassicalValue(name="c", dtype="bits", width=qubits),),
        definitions=tuple(Definition(id=name, node=node)
                          for name, node in nodes.items()),
    )
```

The Method's `plan` turns the circuit into the block `body`, builds the Program and records both in a Plan, and `analyze` turns the observed counts into the Result.

```python
class GHZMethod(Method):
    # The Method's name and version.
    descriptor: ClassVar[AlgorithmDescriptor] = AlgorithmDescriptor(
        method="ghz_counts", version="1")

    def plan(self, problem, *, output, execution, shots, rng):
        # ApplicabilityError refuses a request this Method cannot serve.
        if execution != "quantum" or shots is None:
            raise ApplicabilityError(
                "ghz_counts executes its circuit and needs a shot count")
        circuit = ghz_circuit(problem.qubits)
        # state_input accepts the circuit as the unitary on the all-zero
        # state, and select_preparation turns it into the block "body"
        # that the Program calls.
        body = select_preparation("body", state_input(circuit))
        program = ghz_program(problem.qubits, body)
        # The Plan records the Method's choices. _bind, at its end, attaches
        # the live block, while the Plan's portable record carries only the
        # block's definition.
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
            construction=SelectedConstruction(
                program=program, selections=(body.record,)),
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
        # data.observations.chunks holds one chunk per completed experiment,
        # so one here. Its histogram gives the outcome indices, their counts
        # and the readout width. Bit 0 of an index is classical bit 0, the
        # rightmost character of a Qiskit bit string.
        (chunk,) = data.observations.chunks
        histogram = chunk.histogram()
        outcomes = zip(histogram.index_list(), histogram.weights.tolist())
        # The identity fields name the Plan, its construction, the
        # observations and the chunks this reduction used.
        return GHZResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=(chunk.content_id,),
            counts=tuple((format(index, f"0{histogram.width}b"), count)
                         for index, count in outcomes),
        )
```

`Plan._bind` starts with an underscore but is a supported hook for Method authors ([supported protected extension hooks](algorithm_protocol.md#supported-protected-extension-hooks)). `plan`, `prepare` and `submit` then run the Method.

```python
# Choose the Plan, prepare a Run on the default backend, submit it and wait
# for the Result.
selected = plan(GHZProblem(qubits=3), method=GHZMethod(), shots=100, seed=7)
prepared = prepare(selected)
with prepared.run as run:
    submit(prepared)
    result = run.wait()
print(result.counts)
```

On the default local Aer backend this prints the counts of `000` and `111`, which together account for the 100 shots.

## What each part does

The Problem and the output kind are ordinary Records. `nwqlib.plan` checks that the returned Plan keeps the same Problem, Method and output objects and the stream snapshot taken after planning, so the Method cannot substitute inputs or randomness. It does not check that the Method computes the requested output, so a Method whose Problem allows several outputs refuses, with `ApplicabilityError`, every output it does not compute, as the [Hadamard-test Method](#compare-with-a-built-in-method) does. A built-in Problem such as `Expectation` or `Eigenproblem` serves the same role when its fields describe your input.

`state_input` accepts only a circuit without classical bits, measurements or free parameters, so measurement belongs to the Program's `Measure` node. Building the Qiskit circuit inlines your circuit's gates. A supplied circuit carries no cost formula and no approximation bound, so its [block](blocks.md#supplied-circuits) records unknown synthesis cost and error. `nwqlib.estimate` adds up the resource counts of the Plan's blocks without any Method hook, and it reports the operations of a supplied circuit as unavailable. `state_input` and `select_preparation` each give a supplied circuit a fresh content hash, so planning the same circuit twice gives two Plans with different content hashes. A block bound to your own constructor through `SelectedBlock.bind` is the alternative. It requires a complete `SelectedDefinition`, and the archive writer stores only blocks made by the library's own factories.

`Program` and `SelectedConstruction` check the Program's structure when they are built, before any circuit exists, as [Describe a circuit as a Program](ir.md#check-bind-and-save-a-program) describes. Every field of the Plan's portable record enters the Plan's content hash, so a different shot count is a different Plan.

`analyze` receives the Run's data after every experiment has an observation. NWQLib refuses a Result whose Plan or observation content hash differs from the Run's, or that names a chunk (a part of the observations) the Run did not collect. `_summary_lines` on the Result supplies the first lines of `print(result)`.

The Run uses `AerBackend()` when `prepare` receives no backend. Pass `backend=` for another connection, as [Choose a backend](backends.md) describes. `limits=ExecutionLimits(...)` from `nwqlib.execution` changes the Run's circuit, shot and data allowances, and its `max_simulation_qubits`, 20 by default, bounds the width that Aer accepts.

<a id="beyond-counts"></a>

## Optional hooks {#optional-hooks}

Leave out every hook your Method does not need.

| Hook | When you need it | Where it is documented |
| --- | --- | --- |
| `result_type`, `save_archive`, `load_archive` | A Run saved to a folder, saving or loading a Result or Run, and `check-method` | [When the archive hooks are required](#when-the-archive-hooks-are-required) |
| `error_model(plan)` | Your Method reports its known and unavailable error sources | [Add a method](algorithm_protocol.md#optional-hooks) |
| `verify` | Your Method supports explicitly requested verification checks | [Add a method](algorithm_protocol.md#optional-hooks) |
| `sampling_shots` | `plan(..., accuracy=...)` chooses the shot count from an accuracy request | [Add a method](algorithm_protocol.md#optional-hooks) |
| `prepare_all_refusal` | An adaptive Method, whose later settings depend on earlier outcomes, used with `prepare(plan, settings="all")` | [Add a method](algorithm_protocol.md#optional-hooks) |
| `Result._summary_lines` | Your own first lines of `print(result)` | [Supported protected extension hooks](algorithm_protocol.md#supported-protected-extension-hooks) |
| A `Registration` record | Listing and resolving your Method through a registry | [Register a Method](algorithm_protocol.md#explicit-trusted-registration) |
| A `case()` factory returning a `MethodCase` | `python -m nwqlib check-method`, which runs the case with its independent expected answer against your Method | [Check a Method](algorithm_protocol.md#author-cases-and-their-limits) |
| `reduction_allowance` and a registered reducer | A Plan that reduces readout data while the circuits run | [Add a method](algorithm_protocol.md#reduction-hooks) |

The test suite's reference Hadamard Method, a fuller version of the one in the next section, [`tests/_hadamard_method.py`](https://github.com/pnnl/NWQLib/blob/main/tests/_hadamard_method.py) with its registration in [`tests/_hadamard_method_metadata.py`](https://github.com/pnnl/NWQLib/blob/main/tests/_hadamard_method_metadata.py), shows the error model, `_summary_lines`, the author case and the registration. These files are in the source repository and are not installed with the package.

## Compare your Method with a built-in one {#compare-with-a-built-in-method}

A comparison is meaningful when both Methods compute the same output for the same Problem. The GHZ Method returns raw counts, which no built-in Method returns, so it has no built-in counterpart. The Method below takes an `Expectation` Problem whose observable is one real Pauli term `c P` and estimates its default output, the normalized expectation, with the Hadamard test, as `ExpectationMethod` does by direct measurement. It refuses any other output. An ancilla in `|+>` controls `sign(c) P` on the prepared state, and after a second Hadamard the mean of the ancilla's Z outcome is `sign(c) Re<psi|P|psi>`, which `analyze` multiplies by `|c|`. `compare` plans both Methods for one Problem, and `solve` runs each row.

```python
from typing import ClassVar

from nwqlib import Expectation, compare, solve
from nwqlib.algorithms import (
    AlgorithmDescriptor, ApplicabilityError, ExpectationMethod, Method,
)
from nwqlib.blocks import (
    SelectedConstruction, select_preparation, select_signed_pauli,
    transform_block,
)
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Experiment, ObservationSpec, Plan
from nwqlib.ir import (
    Allocate, BlockCall, ClassicalValue, Definition, Measure, PortMap,
    Program, Register, Sequence,
)
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import ingest_vector


# The Result holds the estimate of <psi|P|psi>, scaled by |c|.
class HadamardResult(Result):
    value: float


class HadamardExpectation(Method):
    descriptor: ClassVar[AlgorithmDescriptor] = AlgorithmDescriptor(
        method="hadamard_expectation", version="1")

    def plan(self, problem, *, output, execution, shots, rng):
        # Only the default output, the normalized expectation of the
        # Problem's own observable, is what this circuit computes.
        if (not isinstance(problem, Expectation)
                or output != problem.default_output()
                or execution != "quantum" or shots is None):
            raise ApplicabilityError(
                "hadamard_expectation measures the normalized expectation "
                "of an Expectation Problem with shots")
        terms = tuple(problem.observable.pauli_terms().labels())
        if len(terms) != 1 or terms[0][1].imag != 0:
            raise ApplicabilityError(
                "hadamard_expectation measures one real Pauli term c P")
        # Prepare psi, put the ancilla in |+>, apply sign(c) P controlled
        # by the ancilla and undo the Hadamard. The ancilla's <Z> is then
        # sign(c) Re<psi|P|psi>.
        h = select_preparation("h", ingest_vector((1.0, 1.0)))
        blocks = (
            select_preparation("prep", problem.state),
            h,
            transform_block(
                "controlled_pauli",
                select_signed_pauli("pauli", problem.observable),
                control=True),
            transform_block("inverse_h", h, adjoint=True),
        )
        on_ancilla = (PortMap(port="system", wire="ancilla"),)
        nodes = {
            "allocate_ancilla": Allocate(wire="ancilla"),
            "allocate_system": Allocate(wire="system"),
            "prep": BlockCall(signature="prep",
                              ports=(PortMap(port="system", wire="system"),)),
            "h": BlockCall(signature="h", ports=on_ancilla),
            "controlled_pauli": BlockCall(
                signature="controlled_pauli",
                ports=(PortMap(port="control", wire="ancilla"),
                       PortMap(port="system", wire="system"))),
            "inverse_h": BlockCall(signature="inverse_h", ports=on_ancilla),
            "measure": Measure(wire="ancilla", result="z"),
        }
        nodes["root"] = Sequence(children=tuple(nodes))
        program = Program(
            root="root",
            registers=(Register(name="ancilla", width=1),
                       Register(name="system", width=len(terms[0][0]))),
            signatures=tuple(block.record.signature for block in blocks),
            classical=(ClassicalValue(name="z", dtype="bits", width=1),),
            definitions=tuple(Definition(id=name, node=node)
                              for name, node in nodes.items()),
        )
        return Plan(
            problem=problem, method=self, output=output,
            execution=execution, shots=shots, randomness=rng.snapshot(),
            construction=SelectedConstruction(
                program=program,
                selections=tuple(block.record for block in blocks)),
            experiments=(Experiment(
                name="hadamard", setting="ancilla_z",
                observation=ObservationSpec(kind="counts", shots=shots)),),
        )._bind(blocks=blocks)

    def analyze(self, plan, data, *, settings):
        (chunk,) = data.observations.chunks
        histogram = chunk.histogram()
        # Outcome 0 has Z = +1 and outcome 1 has Z = -1.
        mean = sum((1 - 2 * index) * count for index, count in zip(
            histogram.index_list(), histogram.weights.tolist())
        ) / chunk.returned_shots
        ((_, c),) = plan.problem.observable.pauli_terms().labels()
        return HadamardResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=(chunk.content_id,),
            value=abs(c.real) * mean,
        )


# <Y> of the normalized state (|0> + i|1>)/sqrt(2) is 1.
problem = Expectation(
    state=(1., 1j),
    observable=ingest_pauli((("Y", 1.),), num_qubits=1),
)
comparison = compare(
    problem, methods=(HadamardExpectation(), ExpectationMethod()),
    shots=2000, seed=7,
)
for row in comparison.rows:
    result = solve(row)
    print(row.method.descriptor.method, result.value)
```

```text
hadamard_expectation 1.0
finite_pauli_expectation 1.0
```

Both rows print 1.0 because the measured outcomes are deterministic for this state. `compare` plans every candidate without measuring anything, keeps your order and chooses no winner ([Plan, compare and solve](scientist.md)).

## When the archive hooks are required

A Run on local Aer without a `directory` keeps its state in memory and writes no folder, so the GHZ Method above runs to its Result without archive hooks. The hooks are called in four situations.

- A durable Run, which keeps its state in a folder on disk, writes its Plan when it is created, and that write calls `save_archive`. A Run is durable when `prepare` receives a `directory`, or when the backend launches work that outlives the process, which IBM Runtime, IonQ, Nexus, Slurm and NWQ-Sim do. Such a Run without a `directory` uses a folder under `~/.nwqlib/runs`.
- `result.save` calls `save_archive` and needs `result_type`, and `run.save` calls `save_archive`.
- `load_result(path, method=GHZMethod)` and `load_run(path, backend=..., method=GHZMethod)` call `load_archive`, and reading a saved Result needs `result_type`.
- `check-method` saves and reloads a temporary Result, so it needs all three.

Without the hooks, a durable `prepare` raises `TypeError` before any submission, and `result.save` and `run.save` raise `TypeError` before writing anything. A failed durable `prepare` keeps its folder, so remove it before retrying.

`save_archive` and `load_archive` store and restore the block with `write_blocks` and `read_blocks` from `nwqlib.blocks._archive`. This module starts with an underscore but is a supported hook for Method authors ([supported protected extension hooks](algorithm_protocol.md#supported-protected-extension-hooks)). Add the three in a subclass that keeps the name `GHZMethod`, so the examples below use it:

```python
# Inherits descriptor, plan and analyze from the GHZMethod above.
class GHZMethod(GHZMethod):

    # The Result class that result.save and a saved Result's reader need.
    result_type: ClassVar[type[Result]] = GHZResult

    def save_archive(self, plan, files):
        from nwqlib.blocks._archive import write_blocks

        # Store the Plan's portable record, and the supplied circuit as a
        # QPY file beside it.
        return dict(plan=files.write_plan(plan),
                    blocks=write_blocks(plan.blocks, files))

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
        blocks = read_blocks(saved["blocks"], plan.construction.selections,
                             files)
        return plan._bind(blocks=blocks)
```

`files.write_plan` returns the Plan's portable record, in which the Problem, Method and output appear as a type name and their fields, and the archive stores it. `files.read_problem` and `files.read_output` rebuild only the built-in Problem and output kinds, so a Method with its own Records rebuilds them from the stored fields with `model_validate`, as above. A Problem that holds a converted operator or state stores it with `files.write_operator` or `files.write_state` and reads it back with the matching reader, as the built-in writer does.

With these hooks the same Method runs on a durable Run, and its Result and Run can be saved and reopened. Each target path must not exist yet:

```python
from nwqlib import load_result, load_run
from nwqlib.backends import AerBackend

# A directory makes the Run durable, and prepare writes its Plan through
# save_archive.
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

`run.exposure` is the work submitted so far, and its `completed` entry counts the finished jobs, circuits and shots. Resuming the reopened Run submits nothing, because its one experiment is already complete, so the printed completed work is still one job of 100 shots, and the restored counts equal the original ones.
