"""Independent tiny event, basis, symbolic and per-location resource witnesses."""

from collections import Counter

import pytest

from nwqlib.blocks import (
    BlockSemantics, SelectedConstruction, SelectedDefinition, lower_qiskit,
    select_preparation, select_zero_reflection, transform_block,
)
from nwqlib.core import Basis, Float64, InputRef, Limit, Source, Unit
from nwqlib.evidence import Evidence
from nwqlib.ir import (
    AdaptiveLoop, Allocate, Argument, Binding, BlockCall, BlockSignature, Branch,
    ClassicalStage, ClassicalValue, Definition, ExprRef, Expression, MeasurementBatch,
    MetadataRef, Parallel, Parameter, ParameterRef, PortMap, Program, QuantumPort,
    Register, Release, Repeat, Sequence, Setting,
)
from nwqlib.resources import ResourceContext, ResourceLaw, WorkloadEstimate, Workspace, estimate

SOURCE = Source(name="hand inventory", version="1", domain="declared test recipe", reference="three serial T gates; no native binding")
EVIDENCE = Evidence(kind="user_assertion", source=SOURCE)
LOWERING = dict(max_operations=1000, max_qubits=4, max_clbits=4)


def preparation(choice="native"):
    """Actual selected X or HZH extension, with an independent hand inventory."""
    from nwqlib.problems import ingest_occupation
    block=select_preparation("prep",ingest_occupation("1",num_qubits=1),choice=choice)
    program=Program(root="root",registers=(Register(name="system",width=1),),signatures=(block.record.signature,),
        definitions=(Definition(id="allocate",node=Allocate(wire="system")),
            Definition(id="call",node=BlockCall(signature="prep",ports=(PortMap(port="system",wire="system"),))),
            Definition(id="root",node=Sequence(children=("allocate","call")))))
    return SelectedConstruction(program=program,selections=(block.record,)),(block,)


def pauli_encoding(select):
    """PREP-SELECT-unPREP around the one-qubit, three-term SELECT named "select".

    Two index qubits address the three terms plus padding.
    """
    from nwqlib.blocks import PauliEncoding, select_pauli_preparation
    prep=select_pauli_preparation("prep",select)
    blocks=(prep,select,transform_block("unprep",prep,adjoint=True))
    program=Program(root="root",registers=(Register(name="index",width=2),Register(name="system",width=1)),
        signatures=tuple(block.record.signature for block in blocks),
        definitions=(Definition(id="index",node=Allocate(wire="index")),
            Definition(id="system",node=Allocate(wire="system")),
            Definition(id="prep",node=BlockCall(signature="prep",ports=(PortMap(port="system",wire="index"),))),
            Definition(id="select",node=BlockCall(signature="select",
                ports=(PortMap(port="index",wire="index"),PortMap(port="system",wire="system")))),
            Definition(id="unprep",node=BlockCall(signature="unprep",ports=(PortMap(port="system",wire="index"),))),
            Definition(id="encoding",node=Sequence(children=("prep","select","unprep"))),
            Definition(id="root",node=Sequence(children=("index","system","encoding")))))
    return SelectedConstruction(program=program,selections=tuple(block.record for block in blocks),
        encodings=(PauliEncoding(root="encoding",select="select",prepare="prep",unprepare="unprep"),))


def scalar(result, metric, location=None):
    fact = result.quantity(metric, location=location).fact
    assert fact.availability == "concrete", fact
    return fact.value.numerator / fact.value.denominator if fact.value.denominator != 1 else fact.value.numerator


def test_quantity_error_keeps_exact_location_selection():
    construction, _ = preparation()
    result = estimate(construction)
    width = result.quantity("logical_width", location="logical_device")
    assert width is next(q for q in result.quantities if q.metric == "logical_width" and q.location == "logical_device")
    for metric, location in (("logical_width", None), ("logical_width", "foreign"), ("branches", None)):
        with pytest.raises(ValueError) as caught:
            result.quantity(metric, location=location)
        message = str(caught.value)
        assert f"metric={metric!r}" in message and f"location={location!r}" in message
        assert ("'logical_device'" in message) == (metric == "logical_width")
        assert ".quantities" in message


def test_hundred_register_lifetimes_use_aggregates_without_native_work():
    """100 simultaneous one-bit wires have peak 100, then release to zero."""
    from nwqlib.resources.fold import _Fold

    registers = tuple(Register(name=f"q{i}", width=1) for i in range(100))
    definitions = tuple(Definition(id=f"a{i}", node=Allocate(wire=r.name)) for i, r in enumerate(registers))
    definitions += tuple(Definition(id=f"r{i}", node=Release(wire=r.name)) for i, r in enumerate(registers))
    program = Program(root="root", registers=registers, definitions=definitions + (
        Definition(id="root", node=Sequence(children=tuple(d.id for d in definitions))),))
    folder = _Fold(SelectedConstruction(program=program, selections=()), ResourceContext())
    result = folder.run()
    assert scalar(result, "logical_width", "logical_device") == 100
    assert scalar(result, "operations") == 0
    assert result.limits == program.limits  # The configured cap was not increased.
    # The fold stores one role/location subtotal per state, independent of the
    # identities of all the live wires. Admission alone owns their lifetimes.
    assert sum(len(state) for state in folder.quantum_footprints) <= 100


def test_symbolic_release_preserves_formal_width_and_varying_bindings():
    expr = Expression(id="n", value=ParameterRef(parameter="n"))
    registers = (Register(name="a", width=ExprRef(expression="n")),
        Register(name="b", width=ExprRef(expression="n")),
        Register(name="c", width=7, role="dirty_ancilla"))
    nodes = (Allocate(wire="a"), Allocate(wire="b"), Release(wire="a"),
        Allocate(wire="c"), Release(wire="b"), Release(wire="c"))
    definitions = tuple(Definition(id=str(i), node=node) for i, node in enumerate(nodes))
    program = Program(root="root", registers=registers, parameters=(Parameter(name="n", domain="integer", lower=1),),
        expressions=(expr,), definitions=definitions + (
            Definition(id="root", node=Sequence(children=tuple(d.id for d in definitions))),))
    construction = SelectedConstruction(program=program, selections=())
    symbolic = estimate(construction)
    assert symbolic.quantity("logical_width", location="logical_device").fact.availability == "symbolic"
    width = symbolic.quantity("logical_width", location="logical_device")
    assert f"symbolic ({width.fact.symbol.name})" in str(symbolic)
    # Exact independently listed live totals: n, 2n, n, n+7, 7, 0.
    for n, peak in ((3, 10), (11, 22)):
        result = estimate(construction.revise(program=program.revise(bindings=(Binding(parameter="n", value=n),))))
        assert scalar(result, "logical_width", "logical_device") == peak
        assert scalar(result, "system", "logical_device") == 2 * n
        assert scalar(result, "dirty_ancilla", "logical_device") == 7


def selected(name="ft", *, width=1, laws=None, bindings=(), workspace=(), parameters=()):
    signature = BlockSignature(name=name, target=SOURCE,
        quantum=(QuantumPort(name="q", width=width),), parameters=parameters)
    if laws is None:
        laws = tuple(ResourceLaw(metric=metric, basis="clifford_t", value=value,
                    interpretation="exact", evidence=EVIDENCE)
                     for metric, value in {"t": 3, "logical_depth": 3, "t_depth": 3, "operations": 3,
                                           "toffoli": 0, "ccz": 0, "arbitrary_rotations": 0}.items())
    return SelectedDefinition(signature=signature,
        semantics=BlockSemantics(kind="unknown", input=None,
            basis=Basis(identity="computational", dimension=1 << width, ordering="q0 rightmost"),
            relation="declared unitary recipe", input_projector="identity", output_projector="identity",
            success="no postselection", workspace=0, restoration="none", epsilon=None,
            approximation_metric="operator norm", approximation_evidence="unknown",
            inverse_legal=False, control_legal=False, phase="declared"),
        implementation=SOURCE, choice="declared only", decomposition=None, cost_law=None,
        cost_parameters=bindings, cost_context="exact selected declaration, conditional on its source",
        construction_work=None, blocker="law-only declaration has no native lowering",
        resource_laws=laws, workspace=workspace)


def calls(record, *, parallel=False, same_wire=False, repeat=1):
    regs = (Register(name="a", width=record.signature.quantum[0].width), Register(name="b", width=record.signature.quantum[0].width))
    definitions = (
        Definition(id="aa", node=Allocate(wire="a")), Definition(id="ab", node=Allocate(wire="b")),
        Definition(id="a", node=BlockCall(signature=record.signature.name, ports=(PortMap(port="q", wire="a"),))),
        Definition(id="b", node=BlockCall(signature=record.signature.name, ports=(PortMap(port="q", wire="a" if same_wire else "b"),))),
        Definition(id="body", node=(Parallel if parallel else Sequence)(children=("a", "b"))),
        Definition(id="repeat", node=Repeat(body="body", count=repeat)),
        Definition(id="root", node=Sequence(children=("aa", "ab", "repeat"))),
    )
    return SelectedConstruction(program=Program(root="root", definitions=definitions,
                                registers=regs, signatures=(record.signature,)), selections=(record,))


def test_controlled_cx_primitive_counts_toffoli_without_synthesis(monkeypatch):
    from nwqlib.blocks.records import Primitive

    record = selected(width=3, laws=()).revise(
        decomposition=(Primitive(gate="cx", qubits=(0,1)),), controlled=True)
    logical = estimate(calls(record), context=ResourceContext(basis="selected_logical"))
    assert scalar(logical,"toffoli") == 2  # Two stored calls, each exactly CCX.
    assert scalar(logical,"cx") == 0
    assert logical.quantity("t").fact.availability == "unknown"
    for basis in ("cx","clifford_t"):
        unresolved = estimate(calls(record), context=ResourceContext(basis=basis))
        assert unresolved.quantity("cx").fact.availability == "unknown"
    # Display must preserve known counts beside missing basis conversion and
    # read stored facts, without folding or serializing the construction again.
    from nwqlib.resources.fold import _Fold
    def forbidden(*args, **kwargs):
        pytest.fail("resource display performed work beyond reading stored quantities")
    monkeypatch.setattr(_Fold, "run", forbidden)
    monkeypatch.setattr(WorkloadEstimate, "model_dump", forbidden)
    rendered = str(logical)
    assert "toffoli: 2 count [exact]" in rendered
    assert logical.quantity("t").fact.reason in rendered
    assert "basis=selected_logical" in rendered and "population=dynamic workload" in rendered


def test_same_choice_small_lowered_inventory_and_trillion_repeat():
    for choice, recipe in (("native", ("x",)), ("hzh", ("h", "z", "h"))):
        construction, native = preparation(choice)
        p = construction.program
        definitions = tuple(d for d in p.definitions if d.id != "root") + (
            Definition(id="repeat", node=Repeat(body="call", count=2)),
            Definition(id="root", node=Sequence(children=("allocate", "repeat"))),
        )
        construction = construction.revise(program=p.revise(definitions=definitions))
        result = estimate(construction)
        # Independent event enumeration: two invocations of the specified recipe.
        events = Counter(gate for _ in range(2) for gate in recipe)
        circuit = lower_qiskit(construction, blocks=native, **LOWERING).circuit.decompose()
        assert dict(circuit.count_ops()) == events
        assert scalar(result, "operations") == sum(events.values())
        assert scalar(result, "logical_depth") == sum(events.values())
        huge_p = construction.program.revise(definitions=tuple(
            d.revise(node=Repeat(body="call", count=10**12)) if d.id == "repeat" else d for d in definitions))
        huge = estimate(construction.revise(program=huge_p))
        assert scalar(huge, "operations") == len(recipe) * 10**12
        assert huge.evaluated_contexts == result.evaluated_contexts
        assert huge.work_units == result.work_units
        assert len(huge.model_dump_json()) < len(result.model_dump_json()) + 1000
        assert WorkloadEstimate.model_validate_json(huge.model_dump_json()) == huge
    # A complete summary must preserve the same conformance check as a partial
    # summary. The one-X inventory is derived by hand; completeness uses the
    # actual metric owner only to exercise this admission boundary.
    from nwqlib.resources.records import GATES
    construction, _ = preparation()
    record = construction.selections[0]
    inventory = dict.fromkeys(GATES, 0)
    inventory.update(operations=1, single_qubit=1, clifford=1, logical_depth=1)
    laws = tuple(ResourceLaw(metric=metric, basis="selected_logical", value=value,
        interpretation="exact", evidence=EVIDENCE, bindings=record.cost_parameters)
        for metric, value in inventory.items())
    full = construction.revise(selections=(record.revise(resource_laws=laws),))
    assert scalar(estimate(full), "operations") == 1
    wrong = tuple(law.revise(value=2) if law.metric == "operations" else law for law in laws)
    with pytest.raises(ValueError, match="conflicts with exact primitive"):
        estimate(construction.revise(selections=(record.revise(resource_laws=wrong),)))
    historical = ResourceLaw(metric="operations", basis="selected_logical", value=2,
        interpretation="upper_bound", evidence=Evidence(kind="observed", source=SOURCE), bindings=record.cost_parameters)
    literal = estimate(construction.revise(selections=(record.revise(resource_laws=(historical,)),))).quantity("operations")
    assert literal.fact.value.numerator == 1 and literal.interpretation == "exact"
    assert historical.evidence.content_id in literal.sources
    assert historical.evidence.content_id not in literal.derivation_sources
    assert literal.fact.evidence.kind == "proved_relation"


def test_ft_serial_parallel_basis_precision_and_dependency():
    import json

    record = selected()
    context = ResourceContext(basis="clifford_t")
    serial = estimate(calls(record), context=context)
    parallel = estimate(calls(record, parallel=True), context=context)
    assert scalar(serial, "t") == scalar(parallel, "t") == 6
    assert scalar(serial, "logical_depth") == scalar(serial, "t_depth") == 6
    assert scalar(parallel, "logical_depth") == scalar(parallel, "t_depth") == 3
    assert parallel.quantity("t").fact.evidence.kind == "user_assertion"
    assert all(source.status == "declared" for source in parallel.sources)
    with pytest.raises(ValueError, match="share a wire"):
        calls(record, parallel=True, same_wire=True)
    with pytest.raises(ValueError, match="no exact persisted native binding|law-only"):
        lower_qiskit(calls(record), blocks=(), **LOWERING)
    missing = estimate(calls(record), context=context.revise(precision=Float64(value=1e-3)))
    assert missing.quantity("t").fact.availability == "unknown"
    # Portable resource data preserves missing inventory versus the declared
    # exact zero Toffoli count without resource replay.
    available = {item["metric"]: item for item in json.loads(serial.model_dump_json())["quantities"]}
    unavailable = {item["metric"]: item for item in json.loads(missing.model_dump_json())["quantities"]}
    assert available["toffoli"]["fact"]["value"]["numerator"] == 0
    assert available["toffoli"]["interpretation"] == "exact"
    assert unavailable["t"]["fact"]["availability"] == "unknown"
    assert unavailable["t"]["fact"]["value"] is None
    assert unavailable["t"]["interpretation"] == "unavailable"
    # Alternative bases for one Toffoli declaration: seven T or one Toffoli.
    laws = tuple(ResourceLaw(metric=m, basis=b, value=v, interpretation="exact", evidence=EVIDENCE)
                 for b, m, v in (("toffoli", "toffoli", 1), ("toffoli", "t", 0),
                                 ("clifford_t", "toffoli", 0), ("clifford_t", "t", 7)))
    construction = calls(selected(width=3, laws=laws))
    primitive = estimate(construction, context=ResourceContext(basis="toffoli"))
    decomposed = estimate(construction, context=context)
    assert (scalar(primitive, "toffoli"), scalar(primitive, "t")) == (2, 0)
    assert (scalar(decomposed, "toffoli"), scalar(decomposed, "t")) == (0, 14)


def test_symbolic_shared_dag_is_bounded_and_bindings_do_not_reuse_laws():
    c, _ = preparation("hzh")
    definitions = list(c.program.definitions[:-1])
    child = "call"
    # A diamond with 2**18 logical leaves; only 18 kept compositions.
    for index in range(18):
        name = f"d{index}"
        definitions.append(Definition(id=name, node=Sequence(children=(child, child))))
        child = name
    definitions += [Definition(id="repeat", node=Repeat(body=child, count=ExprRef(expression="n"))),
                    Definition(id="root", node=Sequence(children=("allocate", "repeat")))]
    p = c.program.revise(definitions=tuple(definitions), parameters=(Parameter(name="n", domain="integer", lower=0),),
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),))
    symbolic = estimate(c.revise(program=p))
    assert symbolic.quantity("operations").fact.availability == "symbolic"
    assert len(symbolic.expressions) < 10
    assert symbolic.evaluated_contexts < 30
    assert scalar(estimate(c.revise(program=p.bind(n=2))), "operations") == 3 * 2**19
    # The same selected signature at different actual arguments must use different laws.
    param = Parameter(name="k", domain="integer", lower=0, upper=5)
    laws = tuple(ResourceLaw(metric="t", basis="clifford_t", value=n, interpretation="exact", evidence=EVIDENCE,
                            bindings=(Binding(parameter="k", value=n),)) for n in (2, 5))
    record = selected(laws=laws, parameters=(param,))
    program = calls(selected()).program
    from nwqlib.ir import Constant
    expressions = tuple(Expression(id=f"n{n}", value=Constant(value=n)) for n in (2, 5))
    definitions = tuple(d.revise(node=d.node.revise(arguments=(Argument(parameter="k", value=ExprRef(expression="n2" if d.id == "a" else "n5")),)))
                        if d.id in {"a", "b"} else d for d in program.definitions)
    program = program.revise(signatures=(record.signature,), expressions=expressions, definitions=definitions)
    result = estimate(SelectedConstruction(program=program, selections=(record,)), context=ResourceContext(basis="clifford_t"))
    assert scalar(result, "t") == 7
    # A fixed declaration missing k must stay inapplicable. Only explicit
    # whole-domain coverage can price both values with the conservative cap5.
    fixed = ResourceLaw(metric="t", basis="clifford_t", value=5, interpretation="upper_bound", evidence=EVIDENCE)
    unmatched = SelectedConstruction(program=program, selections=(record.revise(resource_laws=(fixed,)),))
    assert estimate(unmatched, context=ResourceContext(basis="clifford_t")).quantity("t").fact.availability == "unknown"
    covered = fixed.revise(unbound_parameters=("k",))
    bounded = unmatched.revise(selections=(record.revise(resource_laws=(covered,)),))
    upper = estimate(bounded, context=ResourceContext(basis="clifford_t"))
    assert scalar(upper, "t") == 10  # Two calls, each bounded by5 over0<=k<=5.
    assert upper.quantity("t").interpretation == "upper_bound"
    with pytest.raises(ValueError, match="ambiguous"):
        estimate(bounded.revise(selections=(record.revise(resource_laws=(*laws, covered)),)),
                 context=ResourceContext(basis="clifford_t"))
    with pytest.raises(ValueError, match="formal parameters"):
        record.revise(resource_laws=(covered.revise(unbound_parameters=("absent",)),))
    with pytest.raises(ValueError, match="distinct"):
        laws[0].revise(unbound_parameters=("k",))


def test_metadata_only_law_and_current_mask_storage():
    source = SOURCE.revise(name="60-qubit million-mask declaration")
    law = ResourceLaw(metric="operations", basis="selected_logical", value=10**6,
        interpretation="upper_bound", evidence=Evidence(kind="external_specification", source=source),
        bindings=(Binding(parameter="q", value=60), Binding(parameter="M", value=10**6)),
        assumptions=("one declared operation slot per term; metadata assertion only",))
    record = selected(width=60, laws=(law,), bindings=law.bindings)
    record = record.revise(semantics=record.semantics.revise(input=InputRef(
        identity="external:million-pauli-masks", representation="uint64[x,z]+complex128", source=source)))
    # q=60 fits one word: two 8-byte masks plus a 16-byte coefficient.
    raw = 10**6 * (2 * 8 + 16)
    context = ResourceContext(resident=(Workspace(location="host", purpose="input", bytes=raw, source=source),))
    result = estimate(calls(record, repeat=10**12), context=context)
    assert scalar(result, "operations") == 2 * 10**18
    assert scalar(result, "input_bytes", "host") == 32_000_000
    assert scalar(result, "known_memory", "host") == 32_000_000
    assert result.quantity("t").fact.availability == "unknown"
    assert len(result.model_dump_json()) < 100_000


def test_per_location_reuse_and_simultaneous_workspace():
    # Both declared devices have 8 GiB; one concurrent peak is 10 GiB on gpu0.
    gib = 2**30
    workspace = (Workspace(location="gpu0", purpose="workspace", bytes=5*gib, source=SOURCE),)
    record = selected(workspace=workspace)
    context = ResourceContext(basis="clifford_t", capacities=tuple(
        Limit(stage="execution", metric="memory", unit=Unit(symbol="byte", dimension="bytes"),
              kind="capacity_stock", value=8*gib, scope=device) for device in ("gpu0", "gpu1")))
    serial = estimate(calls(record), context=context)
    concurrent = estimate(calls(record, parallel=True), context=context)
    assert scalar(serial, "memory", "gpu0") == 5*gib
    assert scalar(concurrent, "memory", "gpu0") == 10*gib
    assert 8*gib < scalar(concurrent, "memory", "gpu0") < sum(cap.value for cap in context.capacities)
    c, _ = preparation()
    # Independently listed live widths: gpu0 2 -> 0 -> 3 -> 0 -> 2; gpu1 remains 7.
    regs = (Register(name="a", width=2, location="gpu0", role="clean_ancilla"),
            Register(name="b", width=3, location="gpu0", role="dirty_ancilla"),
            Register(name="c", width=7, location="gpu1"))
    nodes = [Allocate(wire="c"), Allocate(wire="a"), Release(wire="a"), Allocate(wire="b"),
             Release(wire="b"), Allocate(wire="a")]
    definitions = tuple(Definition(id=str(i), node=node) for i, node in enumerate(nodes)) + (
        Definition(id="root", node=Sequence(children=tuple(str(i) for i in range(len(nodes))))),)
    p = Program(root="root", definitions=definitions, registers=regs)
    peak = estimate(SelectedConstruction(program=p, selections=()))
    assert scalar(peak, "logical_width", "gpu0") == 3
    assert scalar(peak, "logical_width", "gpu1") == 7
    assert scalar(peak, "clean_ancilla", "gpu0") == 2
    assert scalar(peak, "dirty_ancilla", "gpu0") == 3
    missing = estimate(calls(selected()))
    assert missing.quantity("memory", location="logical_device").fact.availability == "unknown"
    # Exclusive 3/7-byte paths with two always-resident bytes: the variable peak
    # is upper 9, while the input-only resident subtotal remains exact 2.
    small = selected("small", workspace=(Workspace(location="gpu0", purpose="workspace", bytes=3, source=SOURCE),))
    large = selected("large", workspace=(Workspace(location="gpu0", purpose="workspace", bytes=7, source=SOURCE),))
    p = calls(large).program
    definitions = tuple(d.revise(node=d.node.revise(signature="small")) if d.id == "a" else d
                        for d in p.definitions if d.id not in {"root", "repeat"}) + (
        Definition(id="coin", node=ClassicalStage(implementation=SOURCE, outputs=("coin",))),
        Definition(id="choose", node=Branch(condition="coin", when_true="a", when_false="b")),
        Definition(id="root", node=Sequence(children=("aa", "ab", "coin", "choose"))),
    )
    p = p.revise(definitions=definitions, signatures=(small.signature, large.signature),
                 classical=(ClassicalValue(name="coin", dtype="bool"),))
    context = ResourceContext(resident=(Workspace(location="gpu0", purpose="input", bytes=2, source=SOURCE),))
    branch = estimate(SelectedConstruction(program=p, selections=(small, large)), context=context)
    assert scalar(branch, "memory", "gpu0") == 9
    assert branch.quantity("memory", location="gpu0").interpretation == "upper_bound"
    assert any("exclusive paths" in item for item in branch.quantity("memory", location="gpu0").fact.assumptions)
    assert scalar(branch, "input_bytes", "gpu0") == 2
    assert branch.quantity("input_bytes", location="gpu0").interpretation == "exact"


def test_branch_adaptive_and_shot_populations_remain_conditional():
    record = selected()
    c = calls(record)
    p = c.program
    definitions = tuple(d for d in p.definitions if d.id not in {"root", "repeat"}) + (
        Definition(id="coin", node=ClassicalStage(implementation=SOURCE, outputs=("coin",))),
        Definition(id="branch", node=Branch(condition="coin", when_true="a", when_false="body")),
        Definition(id="root", node=Sequence(children=("aa", "ab", "coin", "branch"))),
    )
    p = p.revise(definitions=definitions, classical=(ClassicalValue(name="coin", dtype="bool"),))
    context = ResourceContext(basis="clifford_t")
    branch = estimate(c.revise(program=p), context=context)
    assert scalar(branch, "t") == 6
    assert branch.quantity("t").interpretation == "upper_bound"
    assert branch.quantity("expected_operations").fact.availability == "unknown"
    loop = AdaptiveLoop(body="a", max_rounds=5, policy=ClassicalStage(implementation=SOURCE), termination="external history")
    definitions = tuple(d.revise(node=loop) if d.id == "branch" else d for d in definitions)
    adaptive = c.revise(program=p.revise(definitions=definitions))
    assert estimate(adaptive, context=context).quantity("t").fact.availability == "unknown"
    bounded = tuple(d.revise(node=loop.revise(resource_envelope="each round uses this fixed selected body, independent of history"))
                    if d.id == "branch" else d for d in definitions)
    result = estimate(c.revise(program=p.revise(definitions=bounded)), context=context)
    assert scalar(result, "t") == 15
    assert result.quantity("t").interpretation == "upper_bound"
    # Two independent settings, three shots each, two three-T calls per shot.
    metadata = MetadataRef(format=SOURCE, data=InputRef(identity="test:setting", representation="test", source=SOURCE))
    original = c.program
    definitions = tuple(d for d in original.definitions if d.id != "root") + (
        Definition(id="ra", node=Release(wire="a")), Definition(id="rb", node=Release(wire="b")),
        Definition(id="experiment", node=Sequence(children=("aa", "ab", "body", "ra", "rb"))),
        Definition(id="batch", node=MeasurementBatch(body="experiment", repetitions=3, observation_kind="counts",
            settings=(Setting(label="a", metadata=metadata), Setting(label="b", metadata=metadata)))),
    )
    batch = c.revise(program=original.revise(root="batch", definitions=definitions))
    result = estimate(batch, context=context)
    assert (scalar(result, "settings"), scalar(result, "shots"), scalar(result, "t")) == (2, 6, 36)
    definitions = tuple(d.revise(node=d.node.revise(repetitions=None)) if d.id == "batch" else d for d in definitions)
    unknown = estimate(batch.revise(program=batch.program.revise(definitions=definitions)), context=context)
    assert scalar(unknown, "settings") == 2
    assert unknown.quantity("shots").fact.availability == unknown.quantity("t").fact.availability == "unknown"
    assert result.quantity("logical_depth").interpretation == "upper_bound"
    assert result.quantity("logical_depth").population == "workload schedule envelope"
    workspace = record.revise(workspace=(Workspace(location="gpu0", purpose="workspace", bytes=5, source=SOURCE),))
    memory_batch = batch.revise(selections=(workspace,))
    memory_context = context.revise(resident=(Workspace(location="host", purpose="input", bytes=2, source=SOURCE),))
    memory = estimate(memory_batch, context=memory_context)
    assert scalar(memory, "memory", "gpu0") == 5
    peak = memory.quantity("memory", location="gpu0")
    assert peak.interpretation == "conditional" and peak.required_schedule == "serial_acquisitions"
    assert "required schedule: serial_acquisitions" in str(peak)
    assert all(condition in str(peak) for condition in peak.fact.assumptions)
    assert memory.quantity("input_bytes", location="host").interpretation == "exact"
    assert memory.quantity("input_bytes", location="host").required_schedule is None
    serial = estimate(memory_batch, context=memory_context.revise(batch_schedule="serial"))
    assert scalar(serial, "memory", "gpu0") == 5
    assert serial.quantity("memory", location="gpu0").interpretation == "exact"
    assert WorkloadEstimate.model_validate_json(memory.model_dump_json()) == memory

    # Two terminal settings, two shots each; the outer batch repeats that whole
    # workload three times. No terminal populations may be replaced by 1/3.
    inner = batch.program.definitions[-1].node.revise(repetitions=2)
    nested_defs = tuple(d.revise(node=inner) if d.id == "batch" else d for d in batch.program.definitions) + (
        Definition(id="outer", node=MeasurementBatch(body="batch", settings=(inner.settings[0],), repetitions=3)),)
    nested = batch.revise(program=batch.program.revise(root="outer", definitions=nested_defs))
    acquisition = estimate(nested, context=context)
    assert (scalar(acquisition, "shots"), scalar(acquisition, "settings"), scalar(acquisition, "unique_settings")) == (12, 6, 2)
    assert (scalar(acquisition, "root_setting_declarations"), scalar(acquisition, "root_repetitions")) == (1, 3)
    assert scalar(acquisition, "t") == 12 * 6  # Each terminal acquisition has two three-T calls.
    assert acquisition.quantity("shots").population == "terminal MeasurementBatch sampled shots"
    repeated = nested.revise(program=nested.program.revise(root="twice", definitions=nested_defs + (
        Definition(id="twice", node=Repeat(body="outer", count=2)),)))
    repeated_result = estimate(repeated, context=context)
    assert (scalar(repeated_result, "shots"), scalar(repeated_result, "settings"), scalar(repeated_result, "unique_settings")) == (24, 12, 2)
    mixed_defs = tuple(d.revise(node=d.node.revise(body="serial_batches")) if d.id == "outer" else d for d in nested_defs) + (
        Definition(id="serial_batches", node=Sequence(children=("batch", "batch"))),)
    mixed = estimate(nested.revise(program=nested.program.revise(definitions=mixed_defs)), context=context)
    assert (scalar(mixed, "shots"), scalar(mixed, "settings"), scalar(mixed, "unique_settings")) == (24, 12, 2)
    unknown_inner = tuple(d.revise(node=d.node.revise(repetitions=None)) if d.id == "batch" else d for d in nested_defs)
    inner_unknown = estimate(nested.revise(program=nested.program.revise(definitions=unknown_inner)), context=context)
    assert inner_unknown.quantity("shots").fact.availability == "unknown"
    assert scalar(inner_unknown, "exact_evaluations") == 0
    assert scalar(inner_unknown, "settings") == 6 and scalar(inner_unknown, "unique_settings") == 2
    # A grouped exact statistic is one evaluation per terminal invocation,
    # independent of how many labels/bins its future native consumer returns.
    for kind in ("pauli_expectation", "probabilities"):
        exact_defs = tuple(d.revise(node=d.node.revise(repetitions=1, observation_kind=kind))
                           if d.id == "batch" else d for d in nested_defs)
        exact = estimate(nested.revise(program=nested.program.revise(definitions=exact_defs)), context=context)
        assert (scalar(exact, "shots"), scalar(exact, "exact_evaluations"), scalar(exact, "settings")) == (0, 6, 6)
        assert scalar(exact, "t") == 36
    # Known kind excludes the opposite population even when both the terminal
    # and outer repetition counts are unknown. No default-to-one is legal.
    for kind, requested, excluded in (("counts", "shots", "exact_evaluations"),
                                      ("probabilities", "exact_evaluations", "shots")):
        unresolved_defs = tuple(d.revise(node=d.node.revise(repetitions=None,
            **({"observation_kind": kind} if d.id == "batch" else {})))
            if d.id in {"batch", "outer"} else d for d in nested_defs)
        unresolved = estimate(nested.revise(program=nested.program.revise(definitions=unresolved_defs)), context=context)
        assert unresolved.quantity(requested).fact.availability == "unknown"
        assert scalar(unresolved, excluded) == 0
        assert unresolved.quantity(excluded).interpretation == "exact"
    unknown_kind_defs = tuple(d.revise(node=d.node.revise(observation_kind=None))
                              if d.id == "batch" else d for d in nested_defs)
    unknown_kind = estimate(nested.revise(program=nested.program.revise(definitions=unknown_kind_defs)), context=context)
    assert scalar(unknown_kind, "t") == 72
    assert all(unknown_kind.quantity(metric).fact.availability == "unknown" for metric in ("shots", "exact_evaluations"))
    zero_terminal_defs = tuple(d.revise(node=d.node.revise(repetitions=0))
                              if d.id == "batch" else d for d in unknown_kind_defs)
    zero_terminal = estimate(nested.revise(program=nested.program.revise(definitions=zero_terminal_defs)), context=context)
    assert (scalar(zero_terminal, "shots"), scalar(zero_terminal, "exact_evaluations"),
            scalar(zero_terminal, "t"), scalar(zero_terminal, "settings"), scalar(zero_terminal, "unique_settings")) == (0, 0, 0, 6, 2)
    zero_outer = tuple(d.revise(node=d.node.revise(repetitions=0)) if d.id == "outer" else d for d in nested_defs)
    none = estimate(nested.revise(program=nested.program.revise(definitions=zero_outer)), context=context)
    assert (scalar(none, "shots"), scalar(none, "settings"), scalar(none, "unique_settings")) == (0, 0, 0)
    from nwqlib.ir import RangeAxis
    axis_defs = tuple(d.revise(node=d.node.revise(axes=(RangeAxis(parameter="n", start=0, stop=3),)))
                      if d.id == "outer" else d for d in nested_defs)
    axis = estimate(nested.revise(program=nested.program.revise(definitions=axis_defs,
        parameters=(Parameter(name="n", domain="integer", lower=0),))), context=context)
    assert axis.quantity("unique_settings").fact.availability == "unknown"
    assert scalar(axis, "shots") == 36 and scalar(axis, "settings") == 18
    # A terminal n-dependent range needs a sum law; selecting n=2 instead
    # gives a concrete two-acquisition request without enumerating the range.
    dependent_defs = tuple(d.revise(node=d.node.revise(repetitions=ExprRef(expression="n"),
        axes=(RangeAxis(parameter="n", start=0, stop=3),))) if d.id == "batch" else d
        for d in batch.program.definitions)
    dependent = batch.program.revise(definitions=dependent_defs,
        parameters=(Parameter(name="n", domain="integer", lower=0),),
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),))
    aggregate = estimate(batch.revise(program=dependent), context=context)
    assert aggregate.quantity("shots").fact.availability == "unknown"
    assert scalar(aggregate, "exact_evaluations") == 0
    assert scalar(aggregate, "settings") == 6
    point = estimate(batch.revise(program=dependent.select_experiment("batch", 0, n=2)), context=context)
    assert (scalar(point, "shots"), scalar(point, "t")) == (2, 12)
    # Repetition legality belongs to each setting's binding context, not only globals.
    bad_batch = batch.program.definitions[-1].node.revise(repetitions=ExprRef(expression="n"),
        settings=(Setting(label="negative", metadata=metadata, bindings=(Binding(parameter="n", value=-1),)),))
    definitions = tuple(d.revise(node=bad_batch) if d.id == "batch" else d for d in definitions)
    with pytest.raises(ValueError, match="repetitions must be nonnegative"):
        batch.program.revise(definitions=definitions,
            parameters=(Parameter(name="n", domain="integer"),),
            expressions=(Expression(id="n", value=ParameterRef(parameter="n")),))


def test_batch_lowering_materializes_one_body_and_ir_repeat_still_repeats(monkeypatch):
    """Two resets per body and four shots imply eight dynamic resets, but native lowering
    materializes only one body.
    """
    import sys
    from types import SimpleNamespace
    from nwqlib.ir import Reset

    class Circuit:
        def __init__(self, *registers):
            self.resets = 0

        def reset(self, wires):
            self.resets += len(wires)

        def find_bit(self, bit):
            return SimpleNamespace(index=bit)

    def forbidden(*args):
        pytest.fail("native gate preparation is forbidden in this metadata/stub witness")

    monkeypatch.setitem(sys.modules, "qiskit", SimpleNamespace(
        QuantumCircuit=Circuit, QuantumRegister=lambda width, name: tuple(range(width)),
        ClassicalRegister=forbidden))
    monkeypatch.setitem(sys.modules, "nwqlib.subroutines.qiskit_compat", SimpleNamespace(
        controlled=forbidden, inverse_realized_gate=forbidden))
    metadata = MetadataRef(format=SOURCE, data=InputRef(identity="test:reset", representation="test", source=SOURCE))
    program = Program(root="batch", registers=(Register(name="q", width=1),), definitions=(
        Definition(id="allocate", node=Allocate(wire="q")),
        Definition(id="reset", node=Reset(wire="q")),
        Definition(id="repeat", node=Repeat(body="reset", count=2)),
        Definition(id="release", node=Release(wire="q")),
        Definition(id="body", node=Sequence(children=("allocate", "repeat", "release"))),
        Definition(id="batch", node=MeasurementBatch(body="body", repetitions=4,
            observation_kind="counts", settings=(Setting(label="reset", metadata=metadata),))),
    ))
    construction = SelectedConstruction(program=program, selections=())
    workload = estimate(construction)
    assert (scalar(workload, "shots"), scalar(workload, "resets")) == (4, 8)
    logical = lower_qiskit(construction, blocks=(), **LOWERING)
    assert logical.circuit.resets == 2  # One body containing coherent IR Repeat(2).
    assert logical.construction_id == construction.content_id
    unknown_defs = program.definitions[:-1] + (program.definitions[-1].revise(
        node=program.definitions[-1].node.revise(repetitions=None, observation_kind=None)),)
    unknown = construction.revise(program=program.revise(definitions=unknown_defs))
    assert lower_qiskit(unknown, blocks=(), **LOWERING).circuit.resets == 2


def test_native_bounds_product_and_control_phase_are_not_zero_or_exact(monkeypatch):
    import numpy as np
    from nwqlib.blocks.selection import _source
    from nwqlib.problems import ingest_product
    product = select_preparation("prep", ingest_product(np.array([[1, 1j], [2, 1]], dtype=complex), max_bytes=1024))
    # Build the same actual selected call, with whole-register port named system.
    def one(block):
        ports = block.record.signature.quantum
        definitions = tuple(Definition(id=f"allocate_{port.name}", node=Allocate(wire=port.name)) for port in ports) + (
            Definition(id="call", node=BlockCall(signature=block.record.signature.name,
                ports=tuple(PortMap(port=p.name, wire=p.name) for p in ports))),
            Definition(id="root", node=Sequence(children=tuple(f"allocate_{p.name}" for p in ports) + ("call",))),
        )
        return SelectedConstruction(program=Program(root="root", definitions=definitions,
            registers=tuple(Register(name=p.name, width=p.width) for p in ports), signatures=(block.record.signature,)),
            selections=(block.record,))
    base = estimate(one(product), context=ResourceContext(basis="cx"))
    controlled = transform_block("controlled", product, control=True)
    ctrl = estimate(one(controlled), context=ResourceContext(basis="cx"))
    assert scalar(base, "cx") == 0  # Two independent one-qubit preparations have no CX.
    assert scalar(base, "preparation_components") == scalar(ctrl, "preparation_components") == 2
    assert scalar(ctrl, "cx") > 0
    assert ctrl.quantity("cx").interpretation == "upper_bound"
    assert ctrl.quantity("operations").fact.availability == "unknown"
    # The tiny selected native constructor supplies an independent CX inventory;
    # an envelope is compared by inequality, never forced to equal optimization.
    native = lower_qiskit(one(controlled), blocks=(controlled,), **LOWERING).circuit.decompose(reps=8)
    assert set(native.count_ops()) <= {"u", "cx"}
    assert native.count_ops().get("cx", 0) <= scalar(ctrl, "cx")
    reflection = select_zero_reflection("reflection", 1)
    ordinary = estimate(one(reflection))
    coherent = estimate(one(transform_block("controlled", reflection, control=True)))
    assert (scalar(ordinary, "operations"), scalar(ordinary, "global_phases")) == (3, 1)
    assert (scalar(coherent, "operations"), scalar(coherent, "global_phases"), scalar(coherent, "controlled")) == (4, 0, 4)
    assert scalar(estimate(one(select_zero_reflection("empty", 0))), "operations") == 0
    # Actual J05 PREP[i,0] has global pi/2; coherent control is S, which maps
    # X to Y and fixes Z. PREP[1+i,0] analogously gives the named T phase.
    from nwqlib.problems import ingest_vector
    phase = transform_block("controlled_s", select_preparation("s", ingest_vector(
        np.array([1j, 0]), max_bytes=1024)), control=True)
    for basis in ("selected_logical", "clifford_t", "cx"):
        named = estimate(one(phase), context=ResourceContext(basis=basis))
        assert (scalar(named, "clifford"), scalar(named, "non_clifford_depth"), scalar(named, "arbitrary_rotations")) == (1, 0, 0)
    t_phase = transform_block("controlled_t", select_preparation("t", ingest_vector(
        np.array([1 + 1j, 0]), max_bytes=1024)), control=True)
    named_t = estimate(one(t_phase), context=ResourceContext(basis="clifford_t"))
    assert (scalar(named_t, "t"), scalar(named_t, "t_depth"), scalar(named_t, "arbitrary_rotations")) == (1, 1, 0)
    near = transform_block("near_s", select_preparation("near", ingest_vector(
        np.array([complex(-2**-52, 1), 0]), max_bytes=1024)), control=True)
    arbitrary = estimate(one(near), context=ResourceContext(basis="clifford_t"))
    assert scalar(arbitrary, "arbitrary_rotations") == 1
    assert arbitrary.quantity("t").fact.availability == "unknown"
    correct_s = ResourceLaw(metric="clifford", basis="clifford_t", value=1, interpretation="exact",
        evidence=EVIDENCE, bindings=phase.record.cost_parameters, controlled=True)
    phase_construction = one(phase)
    conformance = phase_construction.revise(selections=(phase.record.revise(resource_laws=(correct_s,)),))
    assert scalar(estimate(conformance, context=ResourceContext(basis="clifford_t")), "clifford") == 1
    # Huge detached exponent must be rejected before entering the shift law owner.
    import nwqlib._preparation_laws as owner
    monkeypatch.setattr(owner, "direct_preparation_cx_bound", lambda *args: pytest.fail("unbounded law evaluated"))
    bad = selected(laws=(), bindings=(Binding(parameter="num_qubits", value=10**12), Binding(parameter="multiplicity", value=1)))
    bad = bad.revise(cost_law=_source("direct_preparation_cx_bound"), implementation=_source("preparation.native"))
    with pytest.raises(ValueError, match="integer growth"):
        estimate(calls(bad), context=ResourceContext(basis="cx"))


def test_positive_symbolic_repeat_preserves_exported_classical_lifetime():
    # x is defined inside the guaranteed-positive loop; y is defined after it.
    def program(lower):
        return Program(root="root", parameters=(Parameter(name="n", domain="integer", lower=lower),),
            expressions=(Expression(id="n", value=ParameterRef(parameter="n")),),
            classical=(ClassicalValue(name="x", dtype="integer"), ClassicalValue(name="y", dtype="integer")),
            definitions=(
                Definition(id="x", node=ClassicalStage(implementation=SOURCE, outputs=("x",))),
                Definition(id="loop", node=Repeat(body="x", count=ExprRef(expression="n"))),
                Definition(id="y", node=ClassicalStage(implementation=SOURCE, outputs=("y",))),
                Definition(id="root", node=Sequence(children=("loop", "y"))),
            ))
    positive = estimate(SelectedConstruction(program=program(1), selections=()))
    assert scalar(positive, "classical_registers", "host") == 2
    may_skip = estimate(SelectedConstruction(program=program(0), selections=()))
    # The peak over possible positive executions can also be two; without
    # positivity it cannot be called an exact guaranteed live inventory.
    assert may_skip.quantity("classical_registers", location="host").interpretation == "upper_bound"
    assert scalar(may_skip, "classical_registers", "host") == 2
    # Reset layer count is min(1,width); the admitted zero-width target is empty.
    from nwqlib.ir import Reset
    reset = Program(root="root", parameters=(Parameter(name="n", domain="integer", lower=0),),
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),),
        registers=(Register(name="q", width=ExprRef(expression="n")),), definitions=(
            Definition(id="allocate", node=Allocate(wire="q")),
            Definition(id="reset", node=Reset(wire="q")),
            Definition(id="root", node=Sequence(children=("allocate", "reset"))),))
    for width, depth in ((0, 0), (2, 1)):
        result = estimate(SelectedConstruction(program=reset.bind(n=width), selections=()))
        assert scalar(result, "resets") == width and scalar(result, "logical_depth") == depth
    symbolic = estimate(SelectedConstruction(program=reset, selections=()))
    assert symbolic.quantity("logical_depth").fact.availability == "symbolic"
    positive_reset = reset.revise(parameters=(Parameter(name="n", domain="integer", lower=1),))
    assert scalar(estimate(SelectedConstruction(program=positive_reset, selections=())), "logical_depth") == 1


def test_rotation_precision_is_required_and_observation_is_not_projected_as_execution():
    laws = tuple(ResourceLaw(metric=metric, basis="clifford_t", value=value,
        interpretation="exact", evidence=EVIDENCE) for metric, value in (("arbitrary_rotations", 1), ("t", 3)))
    unresolved = estimate(calls(selected(laws=laws)), context=ResourceContext(basis="clifford_t"))
    assert scalar(unresolved, "arbitrary_rotations") == 2
    assert unresolved.quantity("t").fact.availability == "unknown"
    precision = Float64(value=1e-4)
    resolved_laws = tuple(law.revise(precision=precision, synthesis=SOURCE) for law in laws)
    resolved = estimate(calls(selected(laws=resolved_laws)),
                        context=ResourceContext(basis="clifford_t", precision=precision, synthesis=SOURCE))
    assert scalar(resolved, "t") == 6
    # Historical observed source data can inform a planned projection, never
    # create a current executed population or witnessed derived receipt.
    observed = ResourceLaw(metric="operations", basis="selected_logical", value=3,
                          interpretation="exact", evidence=Evidence(kind="observed", source=SOURCE))
    prediction = estimate(calls(selected(laws=(observed,)))).quantity("operations")
    assert prediction.lifecycle == "planned"
    assert prediction.interpretation == "estimate"
    assert prediction.fact.evidence.kind == "empirical_prediction"
    assert prediction.fact.evidence.status == "declared"
    asserted = ResourceLaw(metric="operations", basis="selected_logical", value=1,
                           interpretation="exact", evidence=EVIDENCE)
    obs, assertion = selected("observed", laws=(observed.revise(value=1),)), selected("asserted", laws=(asserted,))
    p = calls(obs).program
    definitions = tuple(d.revise(node=d.node.revise(signature="asserted")) if d.id == "b" else d for d in p.definitions)
    mixed = estimate(SelectedConstruction(program=p.revise(definitions=definitions,
        signatures=(obs.signature, assertion.signature)), selections=(obs, assertion)))
    assert scalar(mixed, "operations") == 2
    assert mixed.quantity("operations").interpretation == "estimate"
    assert mixed.quantity("operations").fact.evidence.kind == "empirical_prediction"
    assert {source.kind for source in mixed.sources} >= {"observed", "user_assertion"}
    zero = selected(laws=(ResourceLaw(metric="operations", basis="selected_logical", value=0,
        interpretation="estimate", evidence=Evidence(kind="numerical_estimate", source=SOURCE)),))
    estimated_zero = estimate(calls(zero, repeat=2)).quantity("operations")
    assert estimated_zero.fact.value.numerator == 0 and estimated_zero.interpretation == "estimate"
    unreachable = estimate(calls(selected(laws=()), repeat=0)).quantity("operations")
    assert unreachable.fact.value.numerator == 0 and unreachable.interpretation == "exact"


def test_resource_quantity_domain_preserves_expectations_and_generic_facts():
    from nwqlib.core import Rational, Scope, Complex128
    from nwqlib.evidence import Fact
    from nwqlib.resources import ResourceQuantity

    def fact(value, metric="operations"):
        return Fact(quantity=metric, unit=Unit(symbol="count", dimension="count"),
                    scope=Scope(domain="supplied inventory"), availability="concrete",
                    value=value, evidence=EVIDENCE)
    def quantity(value, metric="operations", interpretation="exact"):
        return ResourceQuantity(metric=metric, fact=fact(value, metric), population="one call",
                                basis="supplied", interpretation=interpretation)
    for invalid in (Rational(numerator=-1, denominator=1),
                    Rational(numerator=3, denominator=2), Float64(value=-.5)):
        with pytest.raises(ValueError):
            quantity(invalid)
    half = Rational(numerator=1, denominator=2)
    for metric, interpretation in (("expected_operations", "exact"), ("operations", "estimate"),
                                   ("operations", "upper_bound")):
        result = quantity(half, metric, interpretation)
        assert ResourceQuantity.model_validate_json(result.model_dump_json()) == result
    zero = quantity(Rational(numerator=0, denominator=1))
    with pytest.raises(ValueError):
        zero.revise(fact=fact(half))
    # Signed/complex Facts are useful outside resource cardinalities.
    signed = fact(Rational(numerator=-1, denominator=1))
    assert signed.value.numerator == -1
    complex_fact = fact(Complex128(real=1., imag=1.))
    with pytest.raises(ValueError, match="real scalars"):
        zero.revise(fact=complex_fact)
    unknown = zero.revise(interpretation="unavailable", fact=zero.fact.revise(
        availability="unknown", value=None, evidence=None, reason="not inventoried"))
    assert unknown.fact.value is None
    # Portable workload interchange reaches the same admission owner.
    workload = estimate(calls(selected()))
    payload = workload.model_dump(mode="json")
    operations = next(q for q in payload["quantities"] if q["metric"] == "operations")
    operations["fact"]["value"] = Rational(numerator=-3, denominator=1).model_dump(mode="json")
    with pytest.raises(ValueError):
        WorkloadEstimate.model_validate(payload)


def test_signed_pauli_native_cx_law_is_consumed_at_actual_fold_binding(monkeypatch):
    from test_semantic_blocks import one
    from nwqlib.blocks import select_pauli_readout, select_signed_pauli
    from nwqlib.blocks import selection
    from nwqlib.operators import ingest_pauli

    block = select_signed_pauli("select", ingest_pauli((("X", -2.), ("Y", 1.), ("I", 3.)), num_qubits=1))
    context = ResourceContext(basis="cx")
    # Four addresses: three UCG transitions per system bit plus two sign CX
    # give 11 slots. The extra control: six per CX plus two per 14 rotations.
    assert scalar(estimate(one(block), context=context), "cx") == 11
    # Names alone must not authorize a built-in law for a different Source.
    declared = one(block)
    for field in ("cost_law", "implementation"):
        original = getattr(block.record, field)
        foreign = original.revise(version="unrecognized-revision")
        mismatched = declared.revise(selections=(block.record.revise(**{field: foreign}),))
        assert estimate(mismatched, context=context).quantity("cx").fact.availability == "unknown"
    controlled = transform_block("controlled", block, control=True)
    assert scalar(estimate(one(controlled), context=context), "cx") == 94
    assert scalar(estimate(one(transform_block("adjoint", block, control=True, adjoint=True)), context=context), "cx") == 94
    readout = select_pauli_readout("readout", select_signed_pauli(
        "select_3", ingest_pauli((("XZI", 1.), ("YIZ", -2.), ("ZZZ", .5)), num_qubits=3)))
    # Three labels pad to L=4 addresses. Each of the three system UCGs has at
    # most 3(L-1)=9 CX and readout has no sign diagonal: 27. One control costs
    # six per CX plus two per one-qubit slot; each UCG has L core and 2L-1
    # completion slots, so 6*27 + 2*3*11 = 228.
    assert scalar(estimate(one(readout), context=context), "cx") == 27
    controlled_readout = transform_block("controlled_readout", readout, control=True)
    assert scalar(estimate(one(controlled_readout), context=context), "cx") == 228
    construction = pauli_encoding(block)
    assert scalar(estimate(construction, context=context), "cx") == 19  # 4 PREP + 11 SELECT + 4 PREP inverse.
    law = selection.signed_pauli_cx_bound
    monkeypatch.setattr(selection, "signed_pauli_cx_bound", lambda *a, **k: law(*a, **k) + 1)
    assert scalar(estimate(construction, context=context), "cx") == 20


def test_fold_work_limit_is_proportional_to_program_admission_work(monkeypatch):
    """A degree-103 QLS Program admits within the shared default max_steps, but its fold carries every
    count metric through the traversal and needs more steps than that Program limit. Estimation stays within a
    limit proportional to the Program's own admission work, and faster growth is still rejected.
    """
    import numpy as np
    from nwqlib import LinearSystem, plan
    from nwqlib.algorithms.qls import QLS
    from nwqlib.ir.validation import _Admission
    from nwqlib.ir import AdmissionLimits
    from nwqlib.resources.fold import _Fold
    from nwqlib.resources.records import FOLD_WORK_PER_ADMISSION_STEP

    def chain(n):
        return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

    # The Laplacian of an 8 x 4 interior grid selects a degree-103 inverse polynomial.
    A = np.kron(chain(8), np.eye(4)) + np.kron(np.eye(8), chain(4))
    b = np.zeros(32)
    b[1] = 1.0
    selected = plan(LinearSystem(A=A, b=b), method=QLS(max_admission_steps=AdmissionLimits().max_steps), seed=7)
    admission = _Admission(selected.construction.program)
    admission.check()
    limit = selected.construction.program.limits.max_steps
    assert admission.steps <= limit
    workload = estimate(selected.construction, context=ResourceContext(basis="cx"))
    assert limit < workload.work_units <= FOLD_WORK_PER_ADMISSION_STEP * admission.steps
    assert workload.quantity("cx").fact.value.value > 0
    with pytest.raises(ValueError, match="admission envelope"):
        workload.revise(work_units=FOLD_WORK_PER_ADMISSION_STEP * limit + 1)

    # Work that grows with the square of the traversal exceeds the proportional limit.
    merge = _Fold.merge_cost

    def quadratic_merge(self, left, right, **options):
        self.admission.tick(self.admission.steps // 100)
        return merge(self, left, right, **options)

    monkeypatch.setattr(_Fold, "merge_cost", quadratic_merge)
    with pytest.raises(ValueError, match="proportional to Program admission work"):
        estimate(selected.construction, context=ResourceContext(basis="cx"))
