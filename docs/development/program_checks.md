# Program checks

This page is for contributors. It states the rules that the structural check of a `Program` applies, the limits on that check's own work, and where each rule is implemented. Method authors who build a Program start from [Describe a circuit as a Program](../ir.md), which lists the nodes and the limits a refused Program names.

Two terms recur. Admission is the structural check that accepts or rejects a Program, run when the Program is built and again when a Run prepares or lowers it (builds its Qiskit circuit). Kept definitions and kept fields are the ones stored in the Program record.

## What the check covers {#what-the-check-covers}

The public `nwqlib.ir` package constructs immutable Program records, persists shared graphs and checks bounded bindings and declared lifecycles without an SDK. It imports Pydantic and the standard library through the core record module. These operations do not import algorithms, construct circuits, prove block actions, calculate resource totals, lower instructions or execute classical policies. Structural readiness is one prerequisite for a later consumer, and it is neither execution approval nor semantic conformance.

## A symbolic example {#a-symbolic-example}

The following Program declares a one-qubit rotation contract and a measured experiment body. It has seven kept definitions and four expressions, with two parameter points sharing the label Z and the same experiment body. The body contains a Repeat and another reference to the same BlockCall:

```python
from nwqlib.core import Float64, InputRef, Source
from nwqlib.ir import (
    Allocate, Argument, Binary, Binding, BlockCall, BlockSignature,
    ClassicalValue, Constant, Definition, ExprRef, Expression, Measure,
    MeasurementBatch, MetadataRef, Parameter, ParameterRef, PortMap, Program,
    QuantumPort, Register, Release, Repeat, Sequence, Setting,
)

rotation = Source(name="example rotation", version="1",
                  domain="declared unitary", reference="example:rotation")
readout = MetadataRef(
    format=Source(name="example readout", version="1",
                  domain="setting metadata",
                  reference="example:readout-schema"),
    data=InputRef(identity="example:readout-z/1",
                  representation="example readout v1", source=rotation),
)
theta_argument = Argument(parameter="theta",
                          value=ExprRef(expression="theta"))
settings = tuple(
    Setting(label="Z",
            bindings=(Binding(parameter="theta",
                              value=Float64(value=theta)),),
            metadata=readout)
    for theta in (0.0, 0.5)
)
nodes = {
    "allocate": Allocate(wire="q"),
    "call": BlockCall(signature="rotation",
                      ports=(PortMap(port="target", wire="q"),),
                      arguments=(theta_argument,)),
    "repeat": Repeat(body="call", count=ExprRef(expression="n")),
    "measure": Measure(wire="q", result="bits"),
    "release": Release(wire="q"),
    "experiment": Sequence(
        children=("allocate", "repeat", "call", "measure", "release")),
    "batch": MeasurementBatch(body="experiment", settings=settings),
}
symbolic = Program(
    root="batch",
    parameters=(Parameter(name="n", domain="integer", lower=0),
                Parameter(name="theta", domain="real")),
    expressions=(
        Expression(id="n", value=ParameterRef(parameter="n")),
        Expression(id="theta", value=ParameterRef(parameter="theta")),
        Expression(id="zero", value=Constant(value=0)),
        # 0 <= n
        Expression(id="valid",
                   value=Binary(op="le", left="zero", right="n")),
    ),
    constraints=(ExprRef(expression="valid"),),
    registers=(Register(name="q", width=1),),
    classical=(ClassicalValue(name="bits", dtype="bits", width=1),),
    signatures=(BlockSignature(
        name="rotation", target=rotation,
        quantum=(QuantumPort(name="target", width=1, requires="coherent"),),
        parameters=(Parameter(name="theta", domain="real"),),
        obligations=("The selected implementation must implement its "
                     "declared rotation.",),
    ),),
    definitions=tuple(Definition(id=k, node=n) for k, n in nodes.items()),
)
# The unbound constraint and repeat count:
print(symbolic.check_readiness().blockers)
```

Binding the repeat count to one trillion, as in the public operations below, leaves those inventories unchanged. `symbolic.bind(n=-1)` rejects the negative count.

## Construction and identity

Program holds ordered finite tables: definitions, expressions, parameters, registers, classical values and block signatures. Definition holds a local ID and a discriminated node. Child and body fields contain local IDs, never nested copies of another body. Expression similarly has a local ID and references other expressions. Duplicate IDs, missing references, cycles, unsupported kinds and excessive graph depth reject during construction, including unused definitions. Reusing a definition does not imply reusing a quantum state.

Program and all its parts inherit the core Record identity (a content hash), immutability, detached export and validated revision behavior. JSON keeps reference strings and checks supplied nested identities on reload. Parameters, bindings, widths, mappings, metadata format and data references, premises and the structural limits (`AdmissionLimits`) all enter the identity. Local IDs are reference names, and the enclosing content identity binds their definitions.

The public operations are:

```python
assert not symbolic.check_readiness().ready
bound = symbolic.bind(n=10**12)  # validated revision; original unchanged
restored = Program.model_validate_json(bound.model_dump_json())
restored.check_readiness().require_ready()
assert len(tuple(restored.iter_definitions())) == 7
```

`iter_definitions()` visits kept definitions once in declaration order. It is not a dynamic instruction iterator or a resource estimator. `check_readiness()` rejects known-invalid values and returns explicit blockers for unresolved requirements. `require_ready()` refuses the Program as structurally executable if any blocker remains. The result also reports expression evaluations and the other checking work in `lifecycle_steps` for that one check. This is checking work, not quantum events or algorithm costs.

## Parameters and expressions {#bounded-symbolic-subset}

Parameter declares either exact integer or finite binary64 real values, with optional inclusive bounds. Bind integers as Python integers and reals as core Float64 records. Booleans, fractional integer bindings, nonfinite reals and out-of-domain values reject. Quantum widths and Repeat counts are nonnegative integers, and measured classical bit widths are positive. Zero-width quantum registers can represent an absent index register, but do not have a positive-width measurement result.

The expression tree admits Constant, ParameterRef and Binary. Binary operations are add, multiply, exact ceiling division, min, max, equality, less-than and less-or-equal. Operands have the same numeric domain, comparisons return bool, and ceiling division requires integers and a positive denominator. Exact arithmetic is performed directly on bounded Python integers, so this subset needs no symbolic simplification kernel. There is no source-string evaluation, arbitrary SymPy input, exponentiation, expansion, simplification or callable payload.

Each kept expression and declared width evaluates once per distinct binding context, including contexts with unresolved parameters. Repeated setting labels or metadata do not repeat a width scan when their binding context is unchanged. Missing parameters produce unknown values, never zero or false. Constraints reference bool expressions and apply at the root context and each batch binding context. A false constraint rejects, and an unknown constraint blocks readiness. Thus a parameter needed by a global constraint must have a root binding even when individual settings also override it. A width, count or argument used by a consumer must resolve before readiness. Block formal parameter domains are checked against the typed expressions and the concrete argument bounds.

## Admission limits {#admission-limits}

The limits table, with defaults and the error each refusal raises, is in [Describe a circuit as a Program](../ir.md#limits). This section records how the check counts.

`AdmissionLimits` caps top-level definitions and declarations (`max_definitions`, default 16384), graph depth (`max_depth`, default and maximum 128), the kept record and tuple field inventory and the context evaluation and lifecycle work (each capped by `max_steps`, default 100000), and integer result bit length (`max_integer_bits`, default 4096). Multiplication checks prospective integer growth before constructing an oversized result. Work depends on kept structure and bounded distinct contexts, not on Repeat multiplicity. A heavily shared graph with many different lifecycle contexts can reach `max_steps` and reject explicitly.

A `max_definitions` refusal raises `AdmissionDefinitionsExceeded` with the complete count and the ceiling. Both `max_steps` refusals raise `AdmissionStepsExceeded`, a ValueError subclass in `ir/validation.py` that carries the refused stage, its count and the ceiling. Raised inside Program construction, each reaches the caller wrapped in Pydantic's ValidationError, whose `errors()` entry keeps it as `ctx["error"]`. The kept record and tuple field inventory is counted to its end before any table is built, so its count is complete and a `max_steps` of that value passes the inventory. The admission-work count is the one reached when the check stopped, a lower bound. Later admission, preparation and lowering can need more than either count. A Method whose `max_admission_steps` field sets the ceiling (QLS, FixedGCIM, ExpectationMethod, LCHS) refuses naming that field, as the [planning work limit](#planning-work-limit) describes.

Material collection work is charged (counted against `max_steps`) before building context keys, copying or keeping cached states or processing correlation members. Cached keys and outputs keep only collections whose creation has consumed this work budget. Correlation components share immutable member sets interned within each check, so equal state-cache keys reuse the same component object instead of comparing its full member set for every wire. These are bounded units of checking work, not CPU cycles or measured byte totals. Raising one of these limits does not authorize execution.

## Planning work limit {#planning-work-limit}

`ExpectationMethod`, `LCHS`, `FixedGCIM` and `QLS` have a field `max_admission_steps`, default 1,000,000, which sets `AdmissionLimits.max_steps` of each Program the Method builds. It bounds two counts of planning work, which is NWQLib's bookkeeping while it checks a Program's structure, not quantum operations, CPU time or bytes:

- the kept field slots of the Program, and
- the admission work of one check of that Program (its expression evaluations and lifecycle steps).

Program validation, preparation and native lowering each apply the work ceiling to their own check. Passing one stage does not establish that a later stage fits. The resource estimate of a Program may use up to 24 times the field's value. The default is ten times the shared `AdmissionLimits` default of 100,000. In sampled QLS Programs, checking work grows with the number of distinct measured registers.

A larger Program is refused with a ValueError that names the field, the refused stage and its count. Preparation, restoring a saved Run and loading saved evidence check the Program again and name the same field when they refuse.

| Refused stage | What the count means | How to raise the field |
| --- | --- | --- |
| Kept field slots (`"stored field inventory"`) | The complete count | Raising the field to the count passes the slot count |
| Admission work (`"admission work"`) | The count reached when the check stopped, a lower bound. The complete count was not measured | Raise the field to at least the count. That value is not promised to pass |

Later admission, preparation and lowering can need more than the reported count. Raising `max_admission_steps` changes the planning-work allowance and does not change the selected quantum operations, or, for QLS, the selected polynomial. Pass the new value when you configure the Method, for example `ExpectationMethod(max_admission_steps=2_000_000)`. The shared `max_definitions` limit has no Method field, and a refusal against it asks for a smaller construction.

## Declared quantum and classical lifecycle

All references are whole registers, ordered by declaration and signature. Subregister slicing and parallel scheduling are not part of this check. A BlockCall maps every exclusive formal port exactly once, in signature order, and aliased wires and known width mismatches reject. BlockSignature holds a versioned Source target, ordered QuantumPorts, formal parameters, interface availability, coupling and semantic obligations. A declaration is a promise that the semantic check of the block must verify against the selected implementation.

The structural check follows three facts per register through the Program.

- The state says which promise a later port requirement may rely on. It is zero, coherent, measured (the observed computational basis state, which a later unitary call makes coherent again unless a jointly coupled mixed or unknown register enters the same call), mixed, unknown or released.
- The coherence epoch numbers the coherent generations of the register. The first allocation starts it at 0. Each measurement, reset, release, reallocation or port that promises a newly prepared zero or coherent output adds one to the register it acts on and to every register correlated with it. A StateClaim names a register and an epoch, so a coherence promise written before a measurement no longer matches after it.
- The correlation group is the set of registers that a joint or undeclared BlockCall may have entangled with this one. A nonunitary effect on one member makes the other members mixed, because measuring or discarding part of an entangled state leaves the rest without a coherent-state guarantee.

The class `_Admission` in `ir/validation.py` documents the complete model.

| Node or effect | Meaning for the structural check |
| --- | --- |
| Allocate | Begin a fresh zero state in the current experiment. Double allocation rejects. |
| Release | Explicitly discard a live wire. Later use rejects until another Allocate. |
| Measure | Define a typed bits value and break the old coherence epoch. The wire remains live. |
| Reset | Prepare zero on a live wire and begin a new coherence epoch. |
| QuantumPort unitary | Clear known zero. Keep coherence epochs only when the declared input and coupling support that guarantee. |
| QuantumPort preserve | Promise exact input-state restoration, including a known-zero promise. |
| QuantumPort zero or coherent | Promise a new preparation with a new epoch. |
| Unknown interface or effect | Keep explicit readiness blockers. Never invent a valid implementation. |
| CoherentRegion | Check named current coherent epochs and forbid breaking them in its body. |
| ClassicalStage in_job | Read available classical inputs and define typed outputs. Unrelated live wires remain usable. |
| ClassicalStage host | Require all quantum wires released before crossing the boundary. Classical values may persist. |

Epochs advance on measurement, reset, discard and new preparations. Allocate after Release advances the existing wire's generation rather than restoring an old epoch merely because the name matches. Root starts with no allocated wires or available classical values. Root may finish with live wires for a future quantum-output consumer, and this does not transport them to another job.

### Correlation groups {#correlation-groups}

A joint multiport BlockCall conservatively connects its wires' correlation groups. Measurement, reset or discard invalidates old coherence claims on all connected partners, including partners not named by that operation. The affected partner state becomes mixed or unknown for subsequent requirements. Explicit independent coupling promises that the call creates no cross-port correlation, and that promise also needs semantic conformance. Independent local operations do not erase correlations already present. This is a conservative declared-effect model, not a simulation or proof of separability or of clean-ancilla restoration.

An unrestricted joint unitary can transfer a mixed input to another output (SWAP is one allowed example). Accordingly, affected unitary outputs and their correlated partners cannot keep a coherent promise just because their own input was fresh. Such calls remain valid for live-only consumers, but conflict with a protected coherent promise. Entirely coherent joint inputs and genuinely independent action preserve their appropriate coherence guarantees. Unknown effects instead make state and epoch obligations unresolved, so an unknown block inside a CoherentRegion remains representable with readiness blockers. Resolving it to a valid declared unitary can make the Program ready, and a known Measure, Reset or incompatible preparation inside that protected region still rejects.

### Branches and loops {#branches-and-loops}

Measurement followed by in-job classical processing, a bool Branch, and Reset followed by another unitary are valid. Measuring one wire while an unrelated wire stays coherent is valid. Moving live state across a host boundary is not. A host workflow discards old wires explicitly, processes available classical values, then uses Allocate for a new experiment.

Branch checks both arms regardless of an eventual runtime condition. Later classical availability is their intersection, and both arms must agree on live wire names. Correlation groups are conservatively joined, and differing epochs become unknown. A value produced on only one arm cannot be read unconditionally. Identical correlation partitions are reused. Different partitions are joined by traversing their finite group and member incidences, keeping the transitive closure without reconnecting the same component once per wire.

Repeat checks at most the first two body effects and requires a stable iteration boundary for multiple or symbolic repetitions, then composes epoch increments arithmetically. A zero count exports no body effects, but its body is still checked for invalid declared uses. A symbolic count that may be zero cannot guarantee body-only classical outputs. The supported direct integer ParameterRef case with a declared lower bound of at least one excludes that zero path, so stable-loop outputs are available, while the unresolved count still blocks executable readiness. Other symbolic expressions receive no general positivity proof.

AdaptiveLoop represents zero to `max_rounds` iterations. It checks the body and then the versioned ClassicalStage policy, including policy reads of measurements from that body. The termination rule remains a declaration, with no scheduler or callable execution. Nonstationary state or correlation effects and changing live-wire sets are explicitly unsupported loop compositions. Independent experiments should use MeasurementBatch.

## Generic independent measurement experiments

MeasurementBatch references one shared body, kept settings and optional compact RangeAxes. Setting identity covers its label, bindings and MetadataRef. `repetitions` declares independent body invocations, uniformly for terminal and outer batches.

Terminal `observation_kind` uses the shared SDK-free `ObservationKind`. `counts` requests sampled shots. `pauli_expectation`, `probabilities` and `trajectory` request exact-statistic evaluations with zero statistical shots. A `trajectory` evaluates the selected body once, and its ordered observation points belong to the selected Experiment's readout details (`ReadoutDetails.positions`), never to settings. An outer batch has kind `None` and repeats descendants. The structural check rejects a nonterminal kind. Unknown kind or repetitions remain valid planning metadata.

The logical lowerer materializes one selected body regardless of these measurement declarations. IR Repeat keeps its in-circuit meaning. Native measurement must separately reject missing requirements before preparation. MetadataRef carries both a versioned format Source and immutable declared data InputRef. An analyzer can therefore distinguish Lanczos degree, parity and readout formats from GCiM matrix, pair, Pauli and quadrature formats without the IR importing either algorithm. Metadata payloads are not decoded or fetched here.

Each setting or axis point is an independent experiment starting with empty wire and classical scope. Its body must explicitly release all wires, and no state or classical result escapes by name. A later observation step associates measured results with the point or setting identity. Repeated references to the batch keep fresh logical scope and do not carry a coherent state between experiments. Nesting an independent batch while outer quantum wires remain live is explicitly unsupported, including inside a CoherentRegion. Fresh local scope cannot hide an outer lifetime obligation from a host boundary. Root batches and batches entered after explicit release are valid, and ordinary in-job ClassicalStage processing with unrelated live wires remains valid in its existing scope.

A RangeAxis keeps start, exclusive stop and positive step as three integers. It validates its domain endpoints without generating any Cartesian product. Axis-dependent count, width, argument and constraint requirements remain unresolved until a point is selected. Axis-independent bodies can be structurally ready without enumerating points.

Select one point with `program.select_experiment(batch_id, setting_index, k=3)`, giving exactly one valid integer for each axis. This creates a validated revision rooted at that batch, containing one setting and no axes, with the selected values bound globally. The selected graph keeps only reachable definitions and their signature and expression dependencies. All global constraints, their dependencies, complete quantum and classical register layouts, effective bindings and premises remain. Definitions use deterministic root-first traversal order. The parent identifies the entire original Program, and no observation IDs or experiments are created. `plan(...)` reuses the immutable source index and source identities across experiments. After this one-time source work, selected construction and validation visit the selected closure plus kept global metadata. Large global metadata still costs work at every selected validation.

The fresh-process check in `docs/scripts/check_core_records.py` exercises Program construction, JSON and schema persistence, bounded binding and readiness while blocking SDK and algorithm import attempts. Its caught-import negative control verifies that swallowing ImportError cannot conceal an attempted import.

## Explicit parallel resource composition

Parallel joins direct declared unitary or preserve BlockCalls on disjoint whole registers. A shared wire is an explicit dependency and rejects this composition, and Sequence remains serial. The node admits no allocation, classical stage or nonunitary child. Register location and system, clean or dirty role describe logical resource placement, independently of compiled wires. MeasurementBatch can keep an explicit nonnegative repetitions expression, and AdaptiveLoop can keep a uniform resource envelope (`resource_envelope`) over history. These declarations are read by the [resource estimate](../resources.md), and they enable no new executable lowering and approve no computation.

The Program and resource entry walkers set work aside before copying child references. An over-budget kept collection is rejected by its length before its children are pushed on the traversal stack. Symbolic Repeat still keeps one body regardless of its integer count. For measurement resource accounting, terminal batches keep their sampled-shot, exact-evaluation and setting populations through nested batch and Repeat composition, and outer root declarations and repetitions are separate metrics. See the resource guide for the explicit serial scheduling condition on variable batch peaks.

## Rules and where they are implemented {#rules-and-code-owners}

These structural rules are NWQLib's own contract and have no paper source. Each one rejects an invalid or oversized Program, or records a readiness blocker for an unresolved requirement, before any circuit, resource estimate or native preparation exists.

| Rule | Failure it prevents | Code |
| --- | --- | --- |
| Construction runs the complete structural check | A consumer receiving an unchecked Program | `ir.records.Program`, `ir.validation.check_program` |
| Linear register lifetime: Allocate, use, Release, and no live register at a host stage | Use after release, or quantum state crossing a host boundary | `ir.validation._Admission.visit` |
| Coherence epochs advance on measurement, reset, release and new preparation | A coherent-state promise made before a measurement being honored after it | `ir.validation._Admission.visit`, `ir.validation._Admission.break_epoch` |
| Correlation groups and partner invalidation | A measurement on one register leaving an entangled partner marked coherent | `ir.validation._Admission.connect`, `ir.validation._Admission.invalidate_coherence` |
| Branch join and loop stability | Paths or iterations that disagree on live registers or their states | `ir.validation._Admission.join`, `ir.validation._Admission.visit` |
| Independent batch scope | State or classical results leaking between experiments | `ir.validation._Admission.visit` |
| Unknown values become blockers and false constraints reject | An unbound width, count or constraint read as zero or false | `ir.validation._Admission.expressions`, `ir.validation._Admission.integer` |
| Finite inventory, depth, work and integer limits | Unbounded checking work or memory from caller metadata | `ir.expressions.AdmissionLimits`, `ir.validation._Admission.tick` |
| Point selection keeps global constraints and register layouts | A selected experiment that drops a constraint or a register | `ir.selection.select_experiment` |
