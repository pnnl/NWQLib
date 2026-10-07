# Compose blocks

<a id="selected-semantic-blocks"></a>A block is one subroutine of your circuit, such as a state preparation or a Pauli SELECT, that a `Program` calls. Selecting a block with one of the `select_*` functions of `nwqlib.blocks` fixes its exact construction and records its cost formulas and approximation error where they are known. The resource estimate and the Qiskit circuit then describe that same construction. Building the Qiskit circuit from a Program and its blocks is called lowering, and only `lower_qiskit` does it.

Selecting a block imports neither Qiskit nor a backend provider. The input's `max_bytes` limit is checked before the input is converted. Reading a block's record does not scan its coefficients and does not make the block executable.

## Example: a Pauli encoding {#compose-and-lower-a-pauli-encoding}

The example block-encodes `A = -2X + Y + 3I` on one system qubit. The normalization is `alpha = sum(abs(c_j)) = 6`, so the projected block is `A/6`. Two index qubits address the three terms and one zero-amplitude padding slot. The encoding is PREP on the index qubits, SELECT on the index and system qubits, and the inverse of PREP. The first two selections show two recipes for preparing `|1>`, an X gate and H, Z, H.

```python
from nwqlib.blocks import (
    PauliEncoding, SelectedConstruction, lower_qiskit,
    select_pauli_preparation, select_preparation, select_signed_pauli,
    transform_block,
)
from nwqlib.ir import (
    Allocate, BlockCall, Definition, PortMap, Program, Register, Sequence,
)
from nwqlib.operators import ingest_pauli
from nwqlib.problems import ingest_occupation

state = ingest_occupation("1", num_qubits=1)
x_prep = select_preparation("prep", state)                  # recipe x
hzh_prep = select_preparation("prep", state, choice="hzh")  # recipe h, z, h

terms = ingest_pauli([("X", -2), ("Y", 1), ("I", 3)], num_qubits=1)
select = select_signed_pauli("select", terms)
prep = select_pauli_preparation("prep", select)
blocks = (prep, select, transform_block("unprep", prep, adjoint=True))


def call(name, **ports):
    maps = tuple(PortMap(port=p, wire=w) for p, w in ports.items())
    return BlockCall(signature=name, ports=maps)


nodes = {
    "index": Allocate(wire="index"),
    "system": Allocate(wire="system"),
    "prep": call("prep", system="index"),
    "select": call("select", index="index", system="system"),
    "unprep": call("unprep", system="index"),
    "encoding": Sequence(children=("prep", "select", "unprep")),
    "root": Sequence(children=("index", "system", "encoding")),
}
program = Program(
    root="root",
    registers=(
        Register(name="index", width=2), Register(name="system", width=1),
    ),
    signatures=tuple(block.record.signature for block in blocks),
    definitions=tuple(Definition(id=k, node=n) for k, n in nodes.items()),
)
construction = SelectedConstruction(
    program=program,
    selections=tuple(block.record for block in blocks),
    encodings=(PauliEncoding(root="encoding", select="select",
                             prepare="prep", unprepare="unprep"),),
)
print(construction.encoding_semantics("encoding").alpha)  # 6.0
saved = construction.model_dump_json()
assert SelectedConstruction.model_validate_json(saved) == construction
logical = lower_qiskit(construction, blocks=blocks, max_operations=1024,
                       max_qubits=3, max_clbits=0)
print(logical.circuit.num_qubits)  # 3
```

The port maps connect the PREP `system` port to the index wire and the SELECT ports to the index and system wires. The JSON round trip keeps the record and its content hash, while the executable blocks stay in `blocks`. Only `lower_qiskit` builds a Qiskit circuit. `nwqlib.resources.estimate(construction)` adds up the resource counts of the same blocks without building the circuit.

## What each selection does {#selected-action-and-identity}

`select_preparation` takes a `StateInput` and keeps its content hash, positive norm, computational ordering and physical global phase. The block promises `U|0> = input / norm`. The chosen unitary extension determines `U` on other inputs, its inverse and its controlled form. The X and HZH recipes agree as full operators, while arbitrary preparations of the same state need not share that stronger relation. Occupation strings are q0-first, so `"10"` prepares the printed ket `|01>` on two qubits.

`select_signed_pauli` takes Hermitian Pauli terms with coefficients `c_j` and does not center them, prune coefficients or expand them to a dense matrix. The normalization is `alpha = sum(abs(c_j))`. Label qubits precede system qubits, with q0 rightmost within each register. The block applies `SELECT = sum_j |j><j| tensor sign(c_j) P_j`, written with the label register first as the papers do. Because the label qubits are the low-order qubits, a Qiskit dense matrix of the block, whose qubit 0 is the least significant index, reads `sum_j sign(c_j) P_j tensor |j><j|` ([block-encoding conventions](conventions.md#block-encoding-and-qsp-conventions)). Unused labels act as the identity.

`select_pauli_preparation` takes the SELECT block and selects the preparation of its coefficient amplitudes. The coefficient vector is converted once into a `StateInput`, within that input's size limits, and selection does not synthesize the PREP circuit. Padding amplitudes are zero. Input with a single term has no label register and no PREP. Identity-only input is valid. Zero input is accepted as an input but cannot define this encoding, whose `alpha` must be positive.

`PauliEncoding` names a `Sequence` of the coefficient PREP, SELECT and the inverse of PREP. Validation checks that the three calls refer to these blocks, in this order, with matching index-register ports. `encoding_semantics(root)` derives `(A, alpha, index-zero projector, epsilon)` from that same SELECT. The index register is **not** promised to return to zero unconditionally. Projecting it onto zero succeeds with probability `||A psi||² / alpha²` for normalized `psi`. `select_zero_reflection` has the Lanczos sign `2|0><0|-I`, including identity on an empty register.

`transform_block` adds one coherent control, takes the adjoint, or both. The control is a whole register, the transformed block's first port. The phase-preserving control and inverse act on the whole circuit of the chosen extension, including its global phase. A transformed block cannot be transformed again. A Method's own trusted constructors can use this operation when their block's definition permits control or inverse. The transformed block keeps the base block's semantics separately as `base_semantics`, and its full-operator approximation error is unknown. A state error defined on the zero input, or a projected block error, is not carried over as a bound on the inverse or controlled operator.

`select_pauli_readout` reuses the signed SELECT input for the label-controlled basis rotation `V`: H for X, H S† for Y, identity for Z, I and padding. The classical outcome is the coefficient sign times the non-I parity on used labels, and zero on padding, without postselection. These outcomes are the diagonal entries of `D` in the computational basis, and `Pi_used` projects onto the used label addresses and acts as the identity on the system register. Thus `V† D V = Pi_used SELECT Pi_used`. The identity extension of the unitary SELECT does not make a padding outcome count +1. The algorithm that uses the readout decodes those outcomes.

## Bind your own constructor

`SelectedBlock` holds a block's record and the code that builds its circuit, and it cannot be modified. A Method author uses `SelectedBlock.bind(record, payload=inputs, constructor=build)` to bind trusted code to the block's exact definition and its inputs. The constructor receives `(block, arguments, method_context)` and returns a circuit. Lowering checks the block's content hash, the argument domains and the operation and width limits before calling it, and each expensive constructor applies its own numerical and synthesis limits. Adding a Method needs no shared registry of implementations. Serializing or revising the record changes no existing binding and does not make the record executable again.

The constructor must implement its versioned definition, including its scientific dependencies and its work formula. The third argument, `method_context`, is a getter. `method_context(factory)` returns the Method's context and creates it with `factory()` on first use. Constructors must not keep the getter or the Run. Control and adjoint bind the exact live base block through `transform_block`, so the base gate as built, with its full phase, is the one that is controlled or inverted. A block with a readiness blocker may have no constructor, but it cannot run. `SelectedBlock.bind` imports no SDK and builds no circuit.

## Supplied circuits

`select_preparation` binds a supplied Qiskit circuit (input kind `qiskit.supplied`) as its own implementation, `preparation.supplied`. Each such selection records a new random identifier, so its record gets a new content hash, and a matching input name or portable record cannot attach another circuit to it. The supplied choice has no decomposition, CX cost formula or resource laws, and its approximation error is unknown. Control and inverse keep that uncertainty rather than inheriting the cost formula of a direct state preparation. The bytes of the stored circuit data are bounded separately from the unknown synthesis and allocation costs of the circuit. Stored numerical profiles must cover the block's exact content hash. Whether a backend can run the block is checked separately against its declared implementation and primitives.

A supplied preparation and a supplied `BlockEncoding` read their circuit the same way. `AnnotatedOperation` gates keep their ordered control, inverse and finite real power modifiers, including fractional, negative and zero powers. Reading copies the stored base gate and its modifiers, sharing repeated objects and rejecting cycles, and it neither expands powers nor synthesizes control, inverse or power. The modifier lists and control-bit storage are checked against their limits before allocation, and arrays stay within the finite-data and cumulative copy limits. Qiskit's high-level synthesis (HLS) of these gates runs at the shared preparation step before execution, and its unknown expansion cost is separate from the gate count of the stored circuit.

A block whose implementation or cost formula is unknown is still a valid record, and its missing executable binding is a readiness blocker. Saved references never import arbitrary modules or run a callable, so a saved block becomes executable again only when an archive reader restores its circuit, as `load_archive` does in [Run your own circuit](own_circuit.md#when-the-archive-hooks-are-required).

## Costs a selection records {#one-graph-and-scoped-costs}

`SelectedConstruction.program` is the same `Program` graph that the rest of NWQLib reads. Sequences, shared calls and compact `Repeat` nodes keep order and multiplicity, and the resource estimate and lowering both read this one graph. Exact basis, occupation and full-uniform preparations list their gates in order. Other blocks point to their numerical or circuit cost formula. Generic vector, contiguous-prefix uniform and product preparation are distinct choices. The parameters of the product formula mean q applications of a **one-qubit** formula, and they do not price a full `2**q` vector. A controlled block switches the relevant formula and keeps its parameters.

The exact gate list of a recipe, the native CX slot bound, the construction size formula of a block and the physical hardware cost are different quantities. The native CX slot bound is an upper bound on the number of CX gates in the selected native construction before gate cancellation. Control includes single-qubit operations and phase and projector work. The resource estimate carries each formula's context and every unknown cost, and a missing formula is never counted as zero. The preparation formulas are the same functions that the numerical circuit builders use, and they import no SDK. They are bounds where declared, not counts of compiled circuits. Neither selection nor looking up a formula synthesizes or simulates a circuit.

`epsilon=0` for the supported exact recipes means no algorithmic approximation in the ideal-gate relation. The record does not bound floating-point synthesis roundoff, and it implies no hardware fidelity or global algorithm accuracy.

## Resource estimates {#resource-consumer}

[Resource estimates](resources.md) read these same records. A `SelectedDefinition` can declare a `ResourceLaw` for each metric and a `Workspace` for each invocation. A fault-tolerant block that has a cost formula and no executable circuit is still a valid conditional estimate. The arity of primitives and concrete target bounds are checked when the record is built. `lower_qiskit` rejects `Parallel`, so a `Parallel` node enters estimates but cannot be built into a circuit.

## Build the Qiskit circuit {#explicit-logical-lowering}

`lower_qiskit` builds the logical Qiskit circuit of a `SelectedConstruction`. It supports `Sequence`, `Repeat` with a concrete or bound count, coherent regions, `BlockCall` of a known block with valid concrete arguments, allocation and release, computational-basis measurement and reset. Whole registers appear in declaration order, the circuit keeps every declared register, and the returned `LogicalCircuit` gives both the quantum and the classical layout. `Program.select_experiment` selects one root `MeasurementBatch` setting and its bindings, and that Program lowers directly. Batches with several settings, nested batches, dynamic branches, classical stages, adaptive loops and reallocation are not supported here and raise an error.

`lower_qiskit` checks its first four limits before it imports Qiskit or emits an instruction. A Run passes its own `ExecutionLimits` values in place of the defaults where the table says so.

| Argument | Default | What it bounds |
| --- | --- | --- |
| `max_operations` | 100_000 | Dynamic IR visits, with `Repeat` multiplicity included. Positive and inclusive |
| `max_qubits` | 4096 | Total declared quantum width. Nonnegative |
| `max_clbits` | 4096 | Total declared classical width. Nonnegative |
| `max_direct_amplitudes` | 65_536 | Amplitude count of each reachable block that declares a direct state preparation. A Run passes `ExecutionLimits.max_direct_amplitudes` |
| `max_synthesis_work` | 1_000_000_000 | Work of the exact syntheses, and of Qiskit's control of them, for the controlled transformed blocks. It is added up before each base is controlled. A Run counts this work against `ExecutionLimits.max_synthesis_work` instead ([limit checks of the exact synthesis](development/dense_synthesis.md#admission-of-the-exact-synthesis)) |

Unused definitions and zero-count bodies are not built. Very large `Repeat` counts are checked arithmetically, without expanding the loop, while empty bodies still count traversal visits. Built block circuits are cached by the block's content hash and, for parameterized calls, by the arguments in formal-parameter order. Static forward and inverse uses share the built base circuit, and parameterized transforms need a supported compact constructor. `construction_work` reports the sum of the blocks' size formulas, which is neither a limit on total work nor a measurement. Each numerical or circuit constructor applies its own cost limits. `defined_selections` lists the block definitions used, and the Program keeps the arguments of each call. A coherent circuit is never split across jobs.

The output is a logical Qiskit circuit that is not compiled, run or exported. QASM export is a separate step ([Export OpenQASM](qasm-streaming.md)).

## Exact dense synthesis

How NWQLib synthesizes and controls dense unitary matrices, and how each construction counts that work against its limits, is on the contributor page [Exact dense synthesis](development/dense_synthesis.md).

- <a id="controlled-dense-unitaries"></a>Exact synthesis of dense and controlled dense unitaries: [Controlled dense unitaries](development/dense_synthesis.md#controlled-dense-unitaries).
- <a id="dense-control-route"></a>The `dense_control_route` option of LCHS, QLS and `build_control_diagonal_generator_encoding`: [Dense control route](development/dense_synthesis.md#dense-control-route).
- <a id="admission-of-the-exact-synthesis"></a>Work and byte formulas of the synthesis, and the limit each construction checks: [Limit checks of the exact synthesis](development/dense_synthesis.md#admission-of-the-exact-synthesis).
- <a id="adjoint-multiplexers"></a>Inverting an uncontrolled multiplexer by its adjoint table: [Adjoint multiplexers](development/dense_synthesis.md#adjoint-multiplexers).

## What the tests check {#bounded-evidence}

The block tests use at most three qubits. Amplitudes computed independently check the controlled `-X` phase, the order of complex preparation and inverse, the reflection sign, the projected action of negative, Y and identity terms, and identity padding. Hand-counted gate lists of the X and HZH recipes tell the two recipes apart. The generic controlled preparation is checked with an inequality against the native CX slot bound. A fresh-process check confirms that `plan_block_encoding` handles the tested banded, Pauli-LCU and dense-dilation inputs without importing Qiskit, and each check that rejects an oversized input has a passing low-work counterpart. These checks establish no large-system performance or hardware claim.

## Source map

Page numbers refer to the listed arXiv versions. The CX laws attached to these selections are mapped in the [resource guide](resources.md#source-map). The sources of the exact dense synthesis, from the KAK decomposition to the CX counts and work formulas, are mapped in [Exact dense synthesis](development/dense_synthesis.md#source-map).

| Relation | Source | Location | Code |
| --- | --- | --- | --- |
| Signed Pauli encoding: PREP amplitudes sqrt(abs(c_j)/alpha), SELECT applies sign(c_j) P_j, projected block A/alpha | An, Childs and Lin, arXiv:2312.03916v2, with coefficient signs moved from both preparation oracles into SELECT | Appendix A.3, Lemma 24, Eq. (178), p. 36 | `blocks.selection.select_signed_pauli`, `blocks.records.SelectedConstruction.encoding_semantics` |
| Exact block encoding with alpha at least the operator norm | An, Childs and Lin, arXiv:2312.03916v2 | Appendix A.2, Definition 23, p. 36 | `blocks.records.SelectedConstruction.encoding_semantics` |
| The same signed form with nonnegative weights of unit L1 norm (their alpha_i = abs(c_j)/alpha, P_i = sign(c_j) P_j and H = A/alpha) and each Pauli carrying its sign, so SELECT squares to the identity | Kirby, Motta and Mezzacapo, arXiv:2208.00567v4 | Sec. 2.1, Eqs. (2)-(4), p. 3, and Sec. 2.2, Eq. (5), p. 4 | `blocks.selection.select_signed_pauli` |
| Zero reflection 2P_0 - I with P_0 the all-zero projector, which PREP conjugation turns into the walk reflection R about the PREP state, so that (RU)^k block-encodes T_k(A/alpha) | Kirby, Motta and Mezzacapo, arXiv:2208.00567v4 | Sec. 2.2, Lemma 1, Eqs. (6)-(7), p. 4 | `blocks.selection.select_zero_reflection` |
| (alpha, a, epsilon) block encoding of a supplied or planned encoding | Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1 | Sec. 4.1, Definition 43, p. 41 | `blocks.encoding.select_block_encoding` |
| Label-controlled readout basis change V with V† D V equal to the used part of SELECT | NWQLib construction | This guide | `blocks.selection.select_pauli_readout` |
| `SelectedBlock.bind` attaches trusted code to one exact block definition, and control and inverse reuse the live base block | NWQLib design rule | This guide | `blocks.selection.SelectedBlock`, `blocks.selection.transform_block`, `blocks.lowering.lower_qiskit` |
