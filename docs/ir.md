# Describe a circuit as a Program

<a id="shared-programs-and-structural-checks"></a>A `Program` from `nwqlib.ir` describes a circuit as named steps, such as allocating a register, calling a block on it and measuring it. A Method's `plan` returns one inside its Plan, and NWQLib checks the Program's structure when you build it, before any circuit exists. [Run your own circuit](own_circuit.md) runs the Program below, and [Compose blocks](blocks.md) describes the blocks a Program calls.

## Example: a GHZ Program {#example-a-ghz-program}

This Program allocates a three-qubit register `q`, calls the block `body`, which prepares the GHZ state from your circuit, and measures `q` into the classical bits `c`.

```python
from qiskit import QuantumCircuit

from nwqlib.blocks import SelectedConstruction, lower_qiskit, select_preparation
from nwqlib.ir import (
    Allocate, BlockCall, ClassicalValue, Definition, Measure, PortMap,
    Program, Register, Sequence,
)
from nwqlib.problems.inputs import state_input

# Your circuit: H on qubit 0, then a CNOT from qubit 0 to each other qubit.
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 2)
# The block "body" prepares the circuit's state on its one port, "system".
body = select_preparation("body", state_input(circuit))

# The Program's nodes, keyed by their ids.
nodes = {
    "allocate": Allocate(wire="q"),
    "body": BlockCall(signature="body",
                      ports=(PortMap(port="system", wire="q"),)),
    "measure": Measure(wire="q", result="c"),
    "root": Sequence(children=("allocate", "body", "measure")),
}
program = Program(
    root="root",
    registers=(Register(name="q", width=3),),
    signatures=(body.record.signature,),
    classical=(ClassicalValue(name="c", dtype="bits", width=3),),
    definitions=tuple(Definition(id=k, node=n) for k, n in nodes.items()),
)
print(program.check_readiness().ready)  # True

# Build the Qiskit circuit from the Program and its block.
construction = SelectedConstruction(program=program,
                                    selections=(body.record,))
print(lower_qiskit(construction, blocks=(body,)).circuit)
```

The last line prints the built circuit, with the block as one gate:

```text
     ┌───────┐┌─┐
q_0: ┤0      ├┤M├──────
     │       │└╥┘┌─┐
q_1: ┤1 body ├─╫─┤M├───
     │       │ ║ └╥┘┌─┐
q_2: ┤2      ├─╫──╫─┤M├
     └───────┘ ║  ║ └╥┘
c: 3/══════════╩══╩══╩═
               0  1  2
```

A wire is the name of a register. `state_input` accepts only a circuit without classical bits, measurements or free parameters, so measurement belongs to the Program's `Measure` node. The Program and its block records are immutable, and their JSON form, `program.model_dump_json()`, reads back with `Program.model_validate_json` to an equal Program.

## Nodes

Each step of a Program is a node, stored with its id in a `Definition`. A node refers to other nodes by id, so one definition can be used in several places.

| Node | What it does | Minimal constructor |
| --- | --- | --- |
| `Allocate` | Starts a register in the zero state | `Allocate(wire="q")` |
| `BlockCall` | Calls a block, mapping each of its ports to a whole register | `BlockCall(signature="body", ports=(PortMap(port="system", wire="q"),))` |
| `Measure` | Measures a register in the computational basis into classical bits, qubit i into bit i. The register stays allocated | `Measure(wire="q", result="c")` |
| `Sequence` | Runs its children in order | `Sequence(children=("allocate", "body", "measure"))` |
| `Reset` | Returns an allocated register to zero | `Reset(wire="q")` |
| `Release` | Discards a register. Using it again needs another `Allocate` | `Release(wire="q")` |
| `Repeat` | Runs a body a given number of times | `Repeat(body="step", count=3)` |

The Program holds the nodes and the declarations they refer to.

| Field of `Program` | What it holds | Minimal constructor |
| --- | --- | --- |
| `root` | The id of the node that runs first | `root="root"` |
| `definitions` | Every node with its id | `Definition(id="root", node=Sequence(children=(...)))` |
| `registers` | The quantum registers, each with a name and width | `Register(name="q", width=3)` |
| `classical` | The classical values that `Measure` and classical stages write | `ClassicalValue(name="c", dtype="bits", width=3)` |
| `signatures` | The ports and parameters of each block the Program calls | `body.record.signature` of the block `body` |

A Program can also use `MeasurementBatch` for several settings of one body, `Branch`, `ClassicalStage`, `AdaptiveLoop`, `CoherentRegion` and `Parallel`, and parameters and expressions for symbolic widths and counts. [Program checks](development/program_checks.md) states their rules, with a symbolic example.

## Check, bind and save a Program {#check-bind-and-save-a-program}

`Program(...)` and `SelectedConstruction(...)` run the structural check when they are built and raise a `ValueError` for an invalid Program. A requirement that cannot be decided yet, such as a width or `Repeat` count given by an unbound parameter, is not an error. It is reported as a readiness blocker.

| Operation | What it does |
| --- | --- |
| `program.check_readiness()` | Rejects known-invalid values and returns the blockers for unresolved requirements in `.blockers`, with `.ready` true when there are none |
| `program.check_readiness().require_ready()` | Raises if any blocker remains |
| `program.bind(n=...)` | Returns a new, checked Program with the parameter bound. The original is unchanged |
| `program.model_dump_json()`, `Program.model_validate_json(text)` | Save and reload the Program as JSON. Reloading checks the stored content hashes |

The parameters, bindings, widths, port mappings, metadata references, `premises` and `limits` of a Program all enter its content hash, so changing any of them gives a different Program.

The check rejects a Program, or records a blocker, when

- a register is used before `Allocate` or after `Release`, or is still allocated when a host `ClassicalStage` runs,
- a `BlockCall` does not map each port of the block exactly once, maps two ports to the same register, or maps a port to a register of a different known width,
- a width, count or constraint is unbound, which gives a blocker and is never read as zero or false,
- a promise that a register is coherent, made before a measurement, is relied on after it, on the measured register or on any register that a joint or undeclared block call may have entangled with it.

## Limits {#limits}

The structural check bounds its own work and memory, so it refuses a very large or deeply nested Program before that work or memory grows past these limits. The limits are the fields of `AdmissionLimits`, which you pass as `Program(..., limits=AdmissionLimits(...))` (`from nwqlib.ir import AdmissionLimits`). The two `Admission...Exceeded` errors are defined in `nwqlib.ir.validation`. A refusal raised while a Program is built arrives as a Pydantic `ValidationError`, a subclass of `ValueError`, whose message names the limit and the count.

| Limit | Default | What it counts | Error raised | Method field that raises it |
| --- | --- | --- | --- | --- |
| `max_definitions` | 1000000 | Definitions, expressions, parameters, registers, classical values and signatures, the complete count | `AdmissionDefinitionsExceeded` | None. Build a smaller construction |
| `max_depth` | 128, also the largest allowed value | Longest chain of references between definitions | `ValueError`, "definition depth exceeds admission limit" | None |
| `max_steps` | 100000 | Two counts, each against this limit: the fields stored in the Program (stage `"stored field inventory"`), counted completely, and the planning work of one check (stage `"admission work"`), counted until the check stops | `AdmissionStepsExceeded`, naming the stage | `max_admission_steps`, default 1,000,000 for `ExpectationMethod`, `LCHS` and `QLS`, and 10,000,000 for `FixedGCIM` |
| `max_integer_bits` | 4096 | Bit length of every integer value and computed integer | `ValueError`, "integer growth exceeds max_integer_bits" | None |

A Method with a `max_admission_steps` field reports a refusal as a `ValueError` that names the field, the refused stage and its count. [Planning work limit](development/program_checks.md#planning-work-limit) explains what the count means and how far to raise the field. The limits count NWQLib's checking work, not quantum operations, and `Repeat` counts are not expanded, so binding a `Repeat` count to one trillion costs no more to check than binding it to one. Raising a limit lets the check accept a larger Program and runs nothing.

## Rules of the structural check {#rules-of-the-structural-check}

The complete rules are on the contributor page [Program checks](development/program_checks.md).

- <a id="construction-and-identity"></a>Construction, references and content hashes: [Construction and identity](development/program_checks.md#construction-and-identity).
- <a id="bounded-symbolic-subset"></a>Parameter domains, expressions and constraints: [Parameters and expressions](development/program_checks.md#bounded-symbolic-subset).
- <a id="declared-quantum-and-classical-lifecycle"></a>Register states, coherence epochs, correlation groups, branches and loops: [Declared quantum and classical lifecycle](development/program_checks.md#declared-quantum-and-classical-lifecycle).
- <a id="generic-independent-measurement-experiments"></a>`MeasurementBatch`, settings and range axes: [Generic independent measurement experiments](development/program_checks.md#generic-independent-measurement-experiments).
- <a id="explicit-parallel-resource-composition"></a>`Parallel` and resource declarations: [Explicit parallel resource composition](development/program_checks.md#explicit-parallel-resource-composition).
- <a id="rules-and-code-owners"></a>Each rule, the failure it prevents and its code: [Rules and where they are implemented](development/program_checks.md#rules-and-code-owners).
