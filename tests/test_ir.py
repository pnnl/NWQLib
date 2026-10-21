"""Independent finite graph, exact arithmetic and linear lifecycle witnesses."""


import pytest

from nwqlib.core import Float64, InputRef, Source
from nwqlib.ir import (
    AdmissionLimits, AdaptiveLoop, Allocate, Argument, Binary, BlockCall,
    Binding, BlockSignature, Branch, ClassicalStage, ClassicalValue, CoherentRegion, Constant,
    Definition, ExprRef, Expression, Measure, MeasurementBatch, MetadataRef, Parameter,
    ParameterRef, PortMap, Program, QuantumPort, RangeAxis, Register, Release, Repeat,
    Reset, Sequence, Setting, StateClaim,
)

SOURCE = Source(name="test contract", version="1", domain="declarations", reference="test:contract")


def make(nodes, *, classical=(), signatures=(), registers=None, **kwargs):
    """Hand-listed instruction order is the independent lifecycle witness."""
    definitions = tuple(Definition(id=key, node=value) for key, value in nodes.items())
    return Program(root="root", definitions=definitions,
                   registers=(Register(name="q", width=1), Register(name="other", width=1))
                   if registers is None else registers,
                   classical=classical, signatures=signatures, **kwargs)


def metadata(name):
    return MetadataRef(format=Source(name=name, version="1", domain="algorithm metadata",
                                     reference=f"test:{name}/schema"),
                       data=InputRef(identity=f"test:{name}/entry", representation=name, source=SOURCE))


def make_program():
    """Two theta settings share one kept rotation body, repeated a symbolic n >= 0 times."""
    rotation = BlockSignature(
        name="rotation", target=SOURCE, quantum=(QuantumPort(name="target", width=1, requires="coherent"),),
        parameters=(Parameter(name="theta", domain="real"),),
        obligations=("The selected implementation must implement its declared rotation.",))
    call = BlockCall(signature="rotation", ports=(PortMap(port="target", wire="q"),),
                     arguments=(Argument(parameter="theta", value=ExprRef(expression="theta")),))
    settings = tuple(Setting(label="Z", bindings=(Binding(parameter="theta", value=Float64(value=x)),),
                             metadata=metadata("readout")) for x in (0., .5))
    nodes = {"allocate": Allocate(wire="q"), "call": call,
             "repeat": Repeat(body="call", count=ExprRef(expression="n")),
             "measure": Measure(wire="q", result="bits"), "release": Release(wire="q"),
             "experiment": Sequence(children=("allocate", "repeat", "call", "measure", "release")),
             "batch": MeasurementBatch(body="experiment", settings=settings)}
    return Program(
        root="batch",
        parameters=(Parameter(name="n", domain="integer", lower=0), Parameter(name="theta", domain="real")),
        # 0 <= n rejects negative counts without expanding the repeated body.
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),
                     Expression(id="theta", value=ParameterRef(parameter="theta")),
                     Expression(id="zero", value=Constant(value=0)),
                     Expression(id="valid", value=Binary(op="le", left="zero", right="n"))),
        constraints=(ExprRef(expression="valid"),),
        registers=(Register(name="q", width=1),),
        classical=(ClassicalValue(name="bits", dtype="bits", width=1),),
        signatures=(rotation,),
        definitions=tuple(Definition(id=key, node=value) for key, value in nodes.items()),
    )


def test_shared_repeat_roundtrip_binding_and_detached_identity():
    program = make_program()
    assert not program.check_readiness().ready
    with pytest.raises(ValueError, match="not ready"):
        program.check_readiness().require_ready()
    bound = program.bind(n=10**12)
    readiness = bound.check_readiness()
    assert readiness.ready
    # Hand inventory: allocate, call, repeat, measure, release, experiment, batch.
    assert len(tuple(bound.iter_definitions())) == 7
    # Four expressions evaluated once in each of the unbound-theta root and two point contexts.
    assert readiness.expression_evaluations == 12
    assert readiness.lifecycle_steps == program.bind(n=2).check_readiness().lifecycle_steps
    data = bound.model_dump(mode="json")
    assert len(data["definitions"]) == 7
    assert [d["id"] for d in data["definitions"]].count("call") == 1
    assert Program.model_validate_json(bound.model_dump_json()) == bound
    data["definitions"][1]["node"]["signature"] = "corrupted"
    assert bound.definitions[1].node.signature == "rotation"
    assert bound.parent_id == program.content_id
    assert program.bind(n=1).content_id != bound.content_id
    points = bound.definitions[-1].node.settings
    assert points[0].label == points[1].label and points[0].content_id != points[1].content_id


def test_expression_exactness_sharing_types_cycles_and_limits():
    # ceil((7*7 + 7)/3) = 19, independently by 56=18*3+2.
    expressions = (
        Expression(id="seven", value=Constant(value=7)),
        Expression(id="three", value=Constant(value=3)),
        Expression(id="square", value=Binary(op="multiply", left="seven", right="seven")),
        Expression(id="sum", value=Binary(op="add", left="square", right="seven")),
        Expression(id="ceil", value=Binary(op="ceildiv", left="sum", right="three")),
        Expression(id="expected", value=Constant(value=19)),
        Expression(id="equal", value=Binary(op="eq", left="ceil", right="expected")),
    )
    p = make({"root": Sequence()}, expressions=expressions,
             constraints=(ExprRef(expression="equal"),))
    assert p.check_readiness().expression_evaluations == 7
    wrong = list(expressions)
    wrong[-2] = Expression(id="expected", value=Constant(value=18))
    with pytest.raises(ValueError, match="constraint is false"):
        make({"root": Sequence()}, expressions=tuple(wrong), constraints=p.constraints)
    for expressions in (
        (Expression(id="a", value=Binary(op="add", left="a", right="a")),),
        (Expression(id="a", value=Binary(op="add", left="missing", right="missing")),),
        (Expression(id="a", value=Constant(value=1)), Expression(id="a", value=Constant(value=1))),
    ):
        with pytest.raises(ValueError):
            make({"root": Sequence()}, expressions=expressions)
    with pytest.raises(ValueError, match="integer growth"):
        make({"root": Sequence()}, expressions=(Expression(id="a", value=Constant(value=256)),),
             limits=AdmissionLimits(max_integer_bits=8))
    with pytest.raises(ValueError, match="exceeds max_steps"):
        p.revise(limits=AdmissionLimits(max_steps=1))


def feedforward_nodes():
    return {
        "root": Sequence(children=("aq", "ao", "u", "m", "process", "branch", "othercall", "reset", "u")),
        "aq": Allocate(wire="q"), "ao": Allocate(wire="other"),
        "u": BlockCall(signature="u", ports=(PortMap(port="p", wire="q"),)),
        "othercall": BlockCall(signature="u", ports=(PortMap(port="p", wire="other"),)),
        "m": Measure(wire="q", result="bits"), "reset": Reset(wire="q"),
        "process": ClassicalStage(implementation=SOURCE, inputs=("bits",), outputs=("condition",)),
        "branch": Branch(condition="condition", when_true="noop", when_false="noop"),
        "noop": Sequence(),
    }


def feedforward(nodes=None):
    return make(nodes or feedforward_nodes(),
                classical=(ClassicalValue(name="bits", dtype="bits", width=1),
                           ClassicalValue(name="condition", dtype="bool"),
                           ClassicalValue(name="arm_only", dtype="integer")),
                signatures=(BlockSignature(name="u", target=SOURCE,
                                           quantum=(QuantumPort(name="p", width=1, requires="coherent"),)),))


def test_coherence_epoch_broken_and_stale_after_reset_or_reallocate():
    nodes = feedforward_nodes()
    nodes["region"] = CoherentRegion(body="u", claims=(StateClaim(wire="q", epoch=0),))
    nodes["root"] = Sequence(children=("aq", "region", "m", "region"))
    with pytest.raises(ValueError, match="premeasurement"):
        feedforward(nodes)
    nodes["root"] = Sequence(children=("aq", "m", "reset", "region"))
    with pytest.raises(ValueError, match="stale coherent epoch"):
        feedforward(nodes)
    nodes["release"] = Release(wire="q")
    nodes["root"] = Sequence(children=("aq", "release", "aq", "region"))
    with pytest.raises(ValueError, match="stale coherent epoch"):
        feedforward(nodes)
    nodes["region"] = CoherentRegion(body="m", claims=(StateClaim(wire="q", epoch=0),))
    nodes["root"] = Sequence(children=("aq", "region"))
    with pytest.raises(ValueError, match="epoch is broken"):
        feedforward(nodes)


def test_joint_call_invalidates_partner_but_declared_independent_call_preserves_it():
    nodes = feedforward_nodes()
    nodes["joint"] = BlockCall(signature="joint", ports=(PortMap(port="a", wire="q"),
                                                         PortMap(port="b", wire="other")))
    nodes["claim"] = CoherentRegion(body="noop", claims=(StateClaim(wire="other", epoch=0),))
    nodes["root"] = Sequence(children=("aq", "ao", "joint", "m", "claim"))
    baseline = feedforward()
    signature = BlockSignature(name="joint", target=SOURCE,
                               quantum=(QuantumPort(name="a", width=1), QuantumPort(name="b", width=1)))
    args = dict(classical=baseline.classical, signatures=baseline.signatures + (signature,))
    with pytest.raises(ValueError, match="premeasurement"):
        make(nodes, **args)
    args["signatures"] = baseline.signatures + (signature.revise(coupling="independent"),)
    assert make(nodes, **args).check_readiness().ready
    # Protecting the partner also rejects measurement inside the coherent region.
    nodes["claim"] = CoherentRegion(body="m", claims=(StateClaim(wire="other", epoch=0),))
    nodes["root"] = Sequence(children=("aq", "ao", "joint", "claim"))
    args["signatures"] = baseline.signatures + (signature,)
    with pytest.raises(ValueError, match="epoch is broken"):
        make(nodes, **args)


def test_unitary_does_not_preserve_zero_and_adaptive_policy_can_read_body_measurement():
    nodes = feedforward_nodes()
    nodes["needszero"] = BlockCall(signature="zero", ports=(PortMap(port="p", wire="q"),))
    nodes["root"] = Sequence(children=("aq", "u", "needszero"))
    baseline = feedforward()
    with pytest.raises(ValueError, match="state requirement"):
        make(nodes, classical=baseline.classical,
             signatures=baseline.signatures + (BlockSignature(name="zero", target=SOURCE,
                  quantum=(QuantumPort(name="p", width=1, requires="zero"),)),))
    nodes["body"] = Sequence(children=("reset", "u", "m"))
    nodes["loop"] = AdaptiveLoop(body="body", max_rounds=100,
                                 policy=nodes["process"], termination="stop when condition is true")
    nodes["root"] = Sequence(children=("aq", "loop"))
    del nodes["needszero"]
    assert feedforward(nodes).check_readiness().ready


def test_compact_axis_selection_resolves_count_without_expansion():
    nodes = {"root": MeasurementBatch(body="repeat", settings=(Setting(label="Z", metadata=metadata("readout")),),
                                      axes=(RangeAxis(parameter="k", start=1, stop=10**12, step=2),)),
             "repeat": Repeat(body="empty", count=ExprRef(expression="k")), "empty": Sequence()}
    program = make(nodes, parameters=(Parameter(name="k", domain="integer", lower=0),),
                   expressions=(Expression(id="k", value=ParameterRef(parameter="k")),))
    assert not program.check_readiness().ready
    point = program.select_experiment("root", 0, k=3)
    assert point.check_readiness().ready
    assert len(point.definitions) == 3 and len(point.expressions) == 1
    assert Program.model_validate_json(point.model_dump_json()) == point
    for bad in (0, 2, 10**12):
        with pytest.raises(ValueError, match="range axis"):
            program.select_experiment("root", 0, k=bad)


def test_stored_collections_reserve_before_copying_their_frontier():
    from nwqlib.ir import expressions
    from nwqlib.blocks import SelectedConstruction
    from nwqlib.resources import ResourceContext, Workspace, estimate
    events = []

    class WatchedTuple(tuple):
        def __iter__(self):
            events.append("children copied")
            return super().__iter__()

    used = 0
    def reserve(amount):
        nonlocal used
        used += amount
        if used > 128:
            raise ValueError("test stored envelope exceeded")

    oversized = WatchedTuple((1,) * 129)
    with pytest.raises(ValueError, match="stored envelope"):
        tuple(expressions.walk_kept(oversized, reserve))
    assert events == []  # The rejected tuple is never iterated into a new stack.
    used = 0
    visited = tuple(expressions.walk_kept(WatchedTuple((1, 2)), reserve))
    assert len(visited) == 3 and sorted(item for item in visited if type(item) is int) == [1, 2]
    assert events == ["children copied"]

    # The two actual public entry paths reject oversized stored collections.
    with pytest.raises(ValueError, match="stored field inventory"):
        Program(root="root", definitions=(Definition(id="root", node=Sequence()),),
                premises=("small scalar",) * 129, limits=AdmissionLimits(max_steps=128))
    empty = SelectedConstruction(program=Program(root="root", definitions=(Definition(id="root", node=Sequence()),),
        limits=AdmissionLimits(max_steps=128)), selections=())
    resident = Workspace(location="host", purpose="stored", bytes=1, source=SOURCE)
    context = ResourceContext(resident=(resident,) * 129)
    with pytest.raises(ValueError, match="admission work"):
        estimate(empty, context=context)
