"""Independent finite graph, exact arithmetic and linear lifecycle witnesses."""

import json

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


def test_only_terminal_batches_declare_observation_kind():
    setting = Setting(label="s", metadata=metadata("readout"))
    inner = MeasurementBatch(body="empty", settings=(setting,), repetitions=4, observation_kind="counts")
    nodes = {"root": MeasurementBatch(body="repeat", settings=(setting,), repetitions=3),
             "repeat": Repeat(body="shared", count=2),
             "shared": Sequence(children=("inner", "inner")), "inner": inner, "empty": Sequence()}
    legal = make(nodes)
    assert legal.check_readiness().ready
    selected = legal.select_experiment("inner", 0).definitions[-2].node
    assert (selected.observation_kind, selected.repetitions) == ("counts", 4)
    for kind in ("counts", "pauli_expectation", "probabilities"):
        with pytest.raises(ValueError, match="nonterminal MeasurementBatch.*observation_kind"):
            make(nodes | {"root": nodes["root"].revise(observation_kind=kind)})
    for value in (-1, 1.5, True):
        with pytest.raises(ValueError, match="repetitions"):
            inner.revise(repetitions=value)


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
    with pytest.raises(ValueError, match="content_id"):
        Program.model_validate(data)
    assert bound.parent_id == program.content_id
    assert program.bind(n=1).content_id != bound.content_id
    points = bound.definitions[-1].node.settings
    assert points[0].label == points[1].label and points[0].content_id != points[1].content_id


@pytest.mark.parametrize("bad", [-1, 0.5, True])
def test_exact_parameter_domain_rejects_negative_fraction_bool(bad):
    with pytest.raises(ValueError):
        make_program().bind(n=bad)


def test_zero_and_one_counts_and_unknown_width_constraints():
    assert make_program().bind(n=0).check_readiness().ready
    assert make_program().bind(n=1).check_readiness().ready
    nodes = {"root": Sequence(children=("a", "call")), "a": Allocate(wire="q"),
             "call": BlockCall(signature="u", ports=(PortMap(port="p", wire="q"),))}
    args = dict(parameters=(Parameter(name="w", domain="integer", lower=0),),
                expressions=(Expression(id="w", value=ParameterRef(parameter="w")),),
                registers=(Register(name="q", width=ExprRef(expression="w")),),
                signatures=(BlockSignature(name="u", target=SOURCE,
                                           quantum=(QuantumPort(name="p", width=1),)),))
    program = make(nodes, **args)
    assert any("width" in b for b in program.check_readiness().blockers)
    assert program.bind(w=1).check_readiness().ready
    with pytest.raises(ValueError, match="width"):
        program.bind(w=0)


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


def test_node_cycles_missing_duplicate_kind_and_depth():
    for nodes in ({"root": Repeat(body="root", count=1)}, {"root": Sequence(children=("missing",))}):
        with pytest.raises(ValueError):
            make(nodes)
    with pytest.raises(ValueError, match="duplicate definition"):
        Program(root="root", definitions=(Definition(id="root", node=Sequence()),) * 2)
    with pytest.raises(ValueError):
        Definition.model_validate({"id": "root", "node": {"kind": "arbitrary_callable"}})
    with pytest.raises(ValueError, match="depth"):
        make({"root": Repeat(body="a", count=1), "a": Sequence()},
             limits=AdmissionLimits(max_depth=1))


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


def test_measure_feedforward_unrelated_wire_reset_reuse_and_host_boundary():
    assert feedforward().check_readiness().ready
    nodes = feedforward_nodes()
    nodes["process"] = nodes["process"].revise(boundary="host")
    with pytest.raises(ValueError, match="host boundary"):
        feedforward(nodes)
    nodes["releaseq"] = Release(wire="q")
    nodes["releaseo"] = Release(wire="other")
    nodes["root"] = Sequence(children=("aq", "ao", "m", "releaseq", "releaseo", "process", "aq", "u"))
    assert feedforward(nodes).check_readiness().ready


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


def test_read_before_measure_branch_only_output_release_and_aliasing():
    nodes = feedforward_nodes()
    nodes["root"] = Sequence(children=("aq", "process"))
    with pytest.raises(ValueError, match="read before"):
        feedforward(nodes)
    nodes["arm"] = ClassicalStage(implementation=SOURCE, outputs=("arm_only",))
    nodes["consume"] = ClassicalStage(implementation=SOURCE, inputs=("arm_only",))
    nodes["branch"] = nodes["branch"].revise(when_true="arm")
    nodes["root"] = Sequence(children=("aq", "m", "process", "branch", "consume"))
    with pytest.raises(ValueError, match="read before"):
        feedforward(nodes)
    nodes["branch"] = nodes["branch"].revise(when_false="arm")
    assert feedforward(nodes).check_readiness().ready
    nodes["release"] = Release(wire="q")
    nodes["root"] = Sequence(children=("aq", "release", "u"))
    with pytest.raises(ValueError, match="not live"):
        feedforward(nodes)
    with pytest.raises(ValueError, match="aliased"):
        make({"root": BlockCall(signature="two", ports=(PortMap(port="a", wire="q"),
                                                               PortMap(port="b", wire="q")))},
             signatures=(BlockSignature(name="two", target=SOURCE,
                                        quantum=(QuantumPort(name="a", width=1), QuantumPort(name="b", width=1))),))


def test_zero_symbolic_and_adaptive_loop_output_intersection_and_preservation():
    nodes = feedforward_nodes()
    nodes["produce"] = ClassicalStage(implementation=SOURCE, outputs=("arm_only",))
    nodes["consume"] = ClassicalStage(implementation=SOURCE, inputs=("arm_only",))
    nodes["loop"] = Repeat(body="produce", count=0)
    nodes["root"] = Sequence(children=("loop", "consume"))
    with pytest.raises(ValueError, match="read before"):
        feedforward(nodes)
    nodes["loop"] = nodes["loop"].revise(count=1)
    assert feedforward(nodes).check_readiness().ready
    nodes["loop"] = AdaptiveLoop(body="produce", max_rounds=10**12,
                                 policy=ClassicalStage(implementation=SOURCE), termination="declared predicate")
    with pytest.raises(ValueError, match="read before"):
        feedforward(nodes)
    nodes["root"] = Sequence(children=("produce", "loop", "consume"))
    assert feedforward(nodes).check_readiness().ready
    nodes["loop"] = Repeat(body="produce", count=ExprRef(expression="n"))
    baseline = feedforward()
    args = dict(classical=baseline.classical, signatures=baseline.signatures,
                parameters=(Parameter(name="n", domain="integer", lower=0),),
                expressions=(Expression(id="n", value=ParameterRef(parameter="n")),))
    preserved = make(nodes, **args)
    assert not preserved.check_readiness().ready
    assert preserved.bind(n=0).check_readiness().ready
    nodes["root"] = Sequence(children=("loop", "consume"))
    with pytest.raises(ValueError, match="read before"):
        make(nodes, **args)
    nodes["root"] = Sequence(children=("aq", "loop", "u"))
    nodes["loop"] = Repeat(body="u", count=10**12)
    assert feedforward(nodes).check_readiness().ready
    nodes["release"] = Release(wire="q")
    nodes["loop"] = Repeat(body="release", count=2)
    with pytest.raises(ValueError, match="not live"):
        feedforward(nodes)


def test_generic_metadata_compact_axis_and_independent_scopes():
    metadata_refs = (metadata("lanczos-degree-parity-readout"), metadata("gcim-pair-pauli-quadrature"))
    nodes = {"root": MeasurementBatch(body="experiment", settings=tuple(
        Setting(label="same", metadata=ref) for ref in metadata_refs),
        axes=(RangeAxis(parameter="k", start=0, stop=10**12),)),
        "experiment": Sequence(children=("a", "m", "r")), "a": Allocate(wire="q"),
        "m": Measure(wire="q", result="bits"), "r": Release(wire="q")}
    p = make(nodes, classical=(ClassicalValue(name="bits", dtype="bits", width=1),),
             parameters=(Parameter(name="k", domain="integer", lower=0),))
    assert p.check_readiness().ready
    # A trillion axis values do not create more admission visits than one axis value.
    one = p.definitions[0].node.revise(axes=(RangeAxis(parameter="k", start=0, stop=1),))
    small = p.revise(definitions=(p.definitions[0].revise(node=one),) + p.definitions[1:])
    assert p.check_readiness().lifecycle_steps == small.check_readiness().lifecycle_steps
    restored = Program.model_validate_json(p.model_dump_json())
    assert restored.definitions[0].node.settings[1].metadata == metadata_refs[1]
    assert len(json.loads(p.model_dump_json())["definitions"]) == 5
    nodes["experiment"] = Sequence(children=("a", "m"))
    with pytest.raises(ValueError, match="release all wires"):
        make(nodes, classical=p.classical, parameters=p.parameters)


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
    for bad in (0, 2, True, 10**12):
        with pytest.raises(ValueError, match="range axis"):
            program.select_experiment("root", 0, k=bad)


def test_selected_closure_keeps_global_constraints_and_entire_readout_layout():
    setting = Setting(label="Z", metadata=metadata("readout"))
    width = ExprRef(expression="width")
    nodes = {
        "root": Sequence(children=("chosen", "other")),
        "chosen": MeasurementBatch(body="body", settings=(setting,),
            axes=(RangeAxis(parameter="k", start=0, stop=4),), repetitions=1, observation_kind="counts"),
        "other": MeasurementBatch(body="unused", settings=(setting,)),
        "body": Sequence(children=("allocate", "call", "measure", "release")),
        "allocate": Allocate(wire="q"), "call": BlockCall(signature="u", ports=(PortMap(port="p", wire="q"),)),
        "measure": Measure(wire="q", result="bits"), "release": Release(wire="q"), "unused": Sequence(),
    }
    program = make(nodes, registers=(Register(name="q", width=width), Register(name="idle", width=2)),
        classical=(ClassicalValue(name="bits", dtype="bits", width=width),
                   ClassicalValue(name="unused_bits", dtype="bits", width=3)),
        signatures=(BlockSignature(name="u", target=SOURCE, quantum=(QuantumPort(name="p", width=width),)),
                    BlockSignature(name="unused_signature", target=SOURCE)),
        parameters=(Parameter(name="g", domain="integer", lower=0), Parameter(name="k", domain="integer", lower=0),
                    Parameter(name="unused_parameter", domain="integer", lower=0)),
        bindings=(Binding(parameter="g", value=2),),
        expressions=(Expression(id="g", value=ParameterRef(parameter="g")),
                     Expression(id="k", value=ParameterRef(parameter="k")),
                     Expression(id="global", value=Binary(op="le", left="g", right="k")),
                     Expression(id="width", value=Constant(value=1)),
                     Expression(id="unused_expression", value=ParameterRef(parameter="unused_parameter"))),
        constraints=(ExprRef(expression="global"),), premises=("explicit whole-program premise",))
    selected = program.select_experiment("chosen", 0, k=3)
    assert selected.check_readiness().ready
    assert {d.id for d in selected.definitions} == {"chosen", "body", "allocate", "call", "measure", "release"}
    assert {e.id for e in selected.expressions} == {"g", "k", "global", "width"}
    assert {p.name for p in selected.parameters} == {"g", "k"}
    assert tuple(s.name for s in selected.signatures) == ("u",)
    assert selected.registers == program.registers and selected.classical == program.classical
    assert selected.constraints == program.constraints and selected.premises == program.premises
    assert selected.parent_id == program.content_id
    assert Program.model_validate_json(selected.model_dump_json()) == selected
    # The selected axis cannot erase a constraint outside the body: 2 <= 1 is
    # false, although k=1 is independently a legal range/domain value.
    with pytest.raises(ValueError, match="constraint is false"):
        program.select_experiment("chosen", 0, k=1)


def mixed_transfer_nodes():
    """Bell partner discard leaves I/2; an allowed SWAP transfers it to fresh."""
    return {
        "a": Allocate(wire="q"), "b": Allocate(wire="r"), "fresh": Allocate(wire="fresh"),
        "entangle": BlockCall(signature="joint", ports=(PortMap(port="a", wire="q"), PortMap(port="b", wire="r"))),
        "discard": Release(wire="r"),
        "transfer": BlockCall(signature="joint", ports=(PortMap(port="a", wire="q"), PortMap(port="b", wire="fresh"))),
        "claim": CoherentRegion(body="empty", claims=(StateClaim(wire="fresh", epoch=0),)),
        "empty": Sequence(),
        "root": Sequence(children=("a", "b", "entangle", "discard", "fresh", "transfer", "claim")),
    }


def test_joint_unitary_mixed_input_cannot_preserve_fresh_coherence():
    nodes = mixed_transfer_nodes()
    signature = BlockSignature(name="joint", target=SOURCE,
                               quantum=(QuantumPort(name="a", width=1), QuantumPort(name="b", width=1)))
    args = dict(registers=tuple(Register(name=w, width=1) for w in ("q", "r", "fresh")), signatures=(signature,))
    with pytest.raises(ValueError, match="coherent"):
        make(nodes, **args)
    # The actual mixed-input call is legal when the consumer only requires a live wire.
    nodes["root"] = Sequence(children=nodes["root"].children[:-1])
    assert make(nodes, **args).check_readiness().ready
    nodes["protected_transfer"] = CoherentRegion(body="transfer", claims=(StateClaim(wire="fresh", epoch=0),))
    nodes["root"] = Sequence(children=("a", "b", "entangle", "discard", "fresh", "protected_transfer"))
    with pytest.raises(ValueError, match="coherent"):
        make(nodes, **args)
    # Joint action with entirely coherent inputs and an independent action with mixed input remain legal.
    nodes["root"] = Sequence(children=("a", "fresh", "protected_transfer", "claim"))
    assert make(nodes, **args).check_readiness().ready
    nodes["independent"] = nodes["transfer"].revise(signature="independent")
    nodes["root"] = Sequence(children=("a", "b", "entangle", "discard", "fresh", "independent", "claim"))
    args["signatures"] += (signature.revise(name="independent", coupling="independent"),)
    assert make(nodes, **args).check_readiness().ready


def test_nested_independent_batch_cannot_hide_outer_host_obligations():
    nodes = {"a": Allocate(wire="q"), "r": Release(wire="q"),
             "host": ClassicalStage(implementation=SOURCE, boundary="host"),
             "batch": MeasurementBatch(body="host", settings=(Setting(label="s", metadata=metadata("host")),)),
             "protected": CoherentRegion(body="batch", claims=(StateClaim(wire="q", epoch=0),)),
             "root": Sequence(children=("a", "protected"))}
    with pytest.raises(ValueError, match="outer live"):
        make(nodes)
    nodes["root"] = Sequence(children=("batch",))
    assert make(nodes).check_readiness().ready
    nodes["root"] = Sequence(children=("a", "r", "batch", "a"))
    assert make(nodes).check_readiness().ready


def test_unknown_protected_action_stays_symbolic_until_resolved():
    nodes = {"root": Sequence(children=("a", "region", "claim")), "a": Allocate(wire="q"),
             "region": CoherentRegion(body="call", claims=(StateClaim(wire="q", epoch=0),)),
             "claim": CoherentRegion(body="empty", claims=(StateClaim(wire="q", epoch=0),)),
             "call": BlockCall(signature="u", ports=(PortMap(port="p", wire="q"),)), "empty": Sequence()}
    signature = BlockSignature(name="u", target=SOURCE, quantum=(QuantumPort(name="p", width=1),))
    for unresolved in (signature.revise(interface="unknown"),
                       signature.revise(quantum=(signature.quantum[0].revise(ensures="unknown"),))):
        symbolic = make(nodes, signatures=(unresolved,))
        assert not symbolic.check_readiness().ready
        assert any("unknown block" in blocker for blocker in symbolic.check_readiness().blockers)
        resolved = symbolic.revise(signatures=(signature,))
        assert resolved.check_readiness().ready
        assert Program.model_validate_json(symbolic.model_dump_json()) == symbolic
    for known_effect in ("zero", "coherent"):
        with pytest.raises(ValueError, match="epoch is broken"):
            make(nodes, signatures=(signature.revise(quantum=(signature.quantum[0].revise(ensures=known_effect),)),))


def test_positive_direct_symbolic_repeat_guarantees_classical_output():
    nodes = {"root": Sequence(children=("loop", "consume")),
             "loop": Repeat(body="produce", count=ExprRef(expression="n")),
             "produce": ClassicalStage(implementation=SOURCE, outputs=("x",)),
             "consume": ClassicalStage(implementation=SOURCE, inputs=("x",))}
    args = dict(classical=(ClassicalValue(name="x", dtype="integer"),),
                expressions=(Expression(id="n", value=ParameterRef(parameter="n")),))
    symbolic = make(nodes, parameters=(Parameter(name="n", domain="integer", lower=1),), **args)
    assert symbolic.check_readiness().blockers == ("unresolved repeat count: n",)
    assert symbolic.bind(n=1).check_readiness().ready
    assert symbolic.bind(n=10**12).check_readiness().ready
    with pytest.raises(ValueError, match="read before"):
        make(nodes, parameters=(Parameter(name="n", domain="integer", lower=0),), **args)


def test_width_admission_once_per_binding_context(monkeypatch):
    import nwqlib.ir.validation as validation
    original = validation._Admission.integer
    checked = []

    def counted(self, value, context, role, **kwargs):
        if role == "register width":
            checked.append((value, context))
        return original(self, value, context, role, **kwargs)

    monkeypatch.setattr(validation._Admission, "integer", counted)
    for size in (4, 8, 16):
        nodes = {"root": MeasurementBatch(body="empty", settings=tuple(
            Setting(label=str(i), metadata=metadata("width")) for i in range(size))), "empty": Sequence()}
        program = make(nodes, registers=tuple(Register(name=f"q{i}", width=1) for i in range(size)))
        checked.clear()
        validation.check_program(program).require_ready()
        assert len(checked) == size  # One context, each declared width once; labels do not change widths.
    from nwqlib.ir import Binding
    nodes["root"] = nodes["root"].revise(settings=tuple(
        Setting(label="same", metadata=metadata("width"), bindings=(Binding(parameter="w", value=i),))
        for i in (1, 2)))
    program = make(nodes, registers=(Register(name="q", width=ExprRef(expression="w")),),
                   parameters=(Parameter(name="w", domain="integer"),),
                   expressions=(Expression(id="w", value=ParameterRef(parameter="w")),))
    checked.clear()
    validation.check_program(program)
    assert len(checked) == 3  # Unresolved root and two genuinely distinct point contexts.
    nodes["root"] = nodes["root"].revise(settings=(nodes["root"].settings[0].revise(
        bindings=(Binding(parameter="w", value=-1),)),))
    with pytest.raises(ValueError, match="nonnegative"):
        make(nodes, registers=program.registers, parameters=program.parameters, expressions=program.expressions)


def test_identical_branch_partition_does_not_reconnect_each_wire(monkeypatch):
    import nwqlib.ir.validation as validation
    original = validation._Admission.connect
    inputs = []

    def counted(self, q, wires):
        wires = tuple(wires)
        inputs.append(sum(len(q[wire][2]) for wire in wires))
        return original(self, q, wires)

    monkeypatch.setattr(validation._Admission, "connect", counted)
    for size in (4, 8, 16):
        names = tuple(f"q{i}" for i in range(size))
        nodes = {f"a{i}": Allocate(wire=name) for i, name in enumerate(names)}
        nodes.update({"root": Sequence(children=tuple(nodes) + ("joint", "condition", "branch")),
                      "joint": BlockCall(signature="joint", ports=tuple(PortMap(port=w, wire=w) for w in names)),
                      "condition": ClassicalStage(implementation=SOURCE, outputs=("condition",)),
                      "branch": Branch(condition="condition", when_true="empty", when_false="empty"),
                      "empty": Sequence()})
        program = make(nodes, registers=tuple(Register(name=w, width=1) for w in names),
                       classical=(ClassicalValue(name="condition", dtype="bool"),),
                       signatures=(BlockSignature(name="joint", target=SOURCE,
                                                  quantum=tuple(QuantumPort(name=w, width=1) for w in names)),))
        inputs.clear()
        validation.check_program(program).require_ready()
        assert sum(inputs) == size  # Only the initial joint call combines singleton groups.


def test_different_branch_partitions_join_transitively_and_preserve_unrelated_wire():
    names = ("a", "b", "c", "d")
    nodes = {f"alloc_{w}": Allocate(wire=w) for w in names}
    nodes.update({
        "condition": ClassicalStage(implementation=SOURCE, outputs=("condition",)),
        "ab": BlockCall(signature="joint", ports=(PortMap(port="x", wire="a"), PortMap(port="y", wire="b"))),
        "bc": BlockCall(signature="joint", ports=(PortMap(port="x", wire="b"), PortMap(port="y", wire="c"))),
        "branch": Branch(condition="condition", when_true="ab", when_false="bc"),
        "measure_a": Measure(wire="a", result="bits"),
        "claim": CoherentRegion(body="empty", claims=(StateClaim(wire="c", epoch=0),)),
        "empty": Sequence(),
        "root": Sequence(children=tuple(nodes) + ("condition", "branch", "measure_a", "claim")),
    })
    args = dict(registers=tuple(Register(name=w, width=1) for w in names),
                classical=(ClassicalValue(name="condition", dtype="bool"), ClassicalValue(name="bits", dtype="bits", width=1)),
                signatures=(BlockSignature(name="joint", target=SOURCE,
                                           quantum=(QuantumPort(name="x", width=1), QuantumPort(name="y", width=1))),))
    # The declared conservative partition join closes {a,b} and {b,c} to {a,b,c}.
    with pytest.raises(ValueError, match="premeasurement"):
        make(nodes, **args)
    nodes["claim"] = nodes["claim"].revise(claims=(StateClaim(wire="d", epoch=0),))
    assert make(nodes, **args).check_readiness().ready


def test_correlation_merge_charges_distinct_members_before_iteration():
    import nwqlib.ir.validation as validation
    program = make({"root": Sequence()})
    events = []

    class ObservedGroup:
        def __init__(self, members):
            self.members = tuple(members)

        def __len__(self):
            return len(self.members)

        def __iter__(self):
            events.append("members read")
            return iter(self.members)

    def components():
        return {"a": ("coherent", 0, ObservedGroup({"a"})), "b": ("zero", 1, ObservedGroup({"b"}))}

    # The one-step-per-wire scan fits the remaining work exactly; the member
    # charge does not, so it must reject before reading either component.
    admission = validation._Admission(program)
    q = components()
    admission.steps = admission.limits.max_steps - len(q)
    with pytest.raises(ValueError, match="work exceeds"):
        admission.connect(q, ("a", "b"))
    assert events == []
    # A legal merge reads each component once and shares one member set.
    admission = validation._Admission(program)
    q = components()
    merged = admission.connect(q, ("a", "b"))
    assert merged == {"a", "b"} and q["a"][2] is q["b"][2] is merged
    assert (q["a"][:2], q["b"][:2]) == (("coherent", 0), ("zero", 1))
    assert events == ["members read", "members read"]


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
