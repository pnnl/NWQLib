"""Independent small action, phase, binding, and materialization witnesses."""

import builtins
from collections import Counter

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from nwqlib.blocks import (
    PauliEncoding, SelectedConstruction, lower_qiskit, select_pauli_preparation, select_pauli_readout,
    select_preparation, select_signed_pauli, select_zero_reflection, signed_pauli_cx_bound, transform_block,
)
from nwqlib.ir import (
    Allocate, Binding, BlockCall, ClassicalValue, Definition, ExprRef, Expression, Measure,
    MeasurementBatch, MetadataRef, Parameter, ParameterRef, PortMap, Program, Register,
    Release, Repeat, Sequence, Setting,
)
from nwqlib.operators import ingest_pauli
from nwqlib.problems import ingest_occupation, ingest_product, ingest_vector

LOWER = dict(max_operations=100000, max_qubits=3, max_clbits=3)


def construction(blocks, calls, registers, *, repeat=1):
    signatures = tuple(block.record.signature for block in blocks)
    definitions = [Definition(id=f"allocate_{name}", node=Allocate(wire=name)) for name, _ in registers]
    definitions += [Definition(id=f"call_{i}", node=BlockCall(
        signature=name, ports=tuple(PortMap(port=port, wire=wire) for port, wire in ports),
    )) for i, (name, ports) in enumerate(calls)]
    definitions += [Definition(id="body", node=Sequence(children=tuple(f"call_{i}" for i in range(len(calls))))),
                    Definition(id="repeat", node=Repeat(body="body", count=repeat)),
                    Definition(id="root", node=Sequence(children=tuple(f"allocate_{name}" for name, _ in registers) + ("repeat",)))]
    return SelectedConstruction(program=Program(
        root="root", definitions=tuple(definitions), registers=tuple(Register(name=name, width=width) for name, width in registers),
        signatures=signatures,
    ), selections=tuple(block.record for block in blocks))


def one(block, *, repeat=1):
    ports = block.record.signature.quantum
    return construction((block,), ((block.record.signature.name, tuple((p.name, p.name) for p in ports)),),
                        tuple((p.name, p.width) for p in ports), repeat=repeat)


def action(circuit, initial):
    return np.asarray(Statevector(initial).evolve(circuit).data)


def test_lowering_method_context_is_lazy_local_and_not_rebuilt_on_gate_hits(monkeypatch):
    """Instrument lazy method contexts and gate-cache hits to separate local ownership from
    repeated construction.
    """
    from types import SimpleNamespace
    import qiskit
    from nwqlib.blocks import lowering
    from nwqlib.blocks.selection import SelectedBlock
    from nwqlib.core import Source

    contexts, builds = [], []
    payload = object()

    class Circuit:
        def __init__(self, *registers):
            self.qubits = tuple(bit for reg in registers for bit in reg)

        def append(self, gate, wires, *, copy):
            assert not copy and len(wires) == 1

        def find_bit(self, bit):
            return SimpleNamespace(index=self.qubits.index(bit))

    def factory():
        context = []
        contexts.append(context)
        return context

    def build(block, arguments, method_context):
        assert arguments == () and block._payload is payload
        context = method_context(factory)
        context.append(block.record.signature.name)
        builds.append(context)
        return SimpleNamespace(to_gate=lambda **kwargs: object())

    source = Source(name="test.local_constructor", version="1", domain="injected one-qubit binding",
                    reference="explicit factory callable; no common family registration")
    originals = tuple(select_zero_reflection(name, 1) for name in ("first", "second"))
    blocks = tuple(SelectedBlock.bind(block.record.revise(
        signature=block.record.signature.revise(target=source), implementation=source),
        payload=payload, constructor=build) for block in originals)
    selected = construction(blocks, tuple((block.record.signature.name,
        ((block.record.signature.quantum[0].name, "q"),)) for block in (*blocks, blocks[0])), (("q", 1),))
    monkeypatch.setattr(qiskit, "QuantumCircuit", Circuit)
    with pytest.raises(ValueError, match="max_operations"):
        lower_qiskit(selected, blocks=blocks, max_operations=1)
    assert not contexts and not builds
    lower_qiskit(selected, blocks=blocks, **LOWER)
    lower_qiskit(selected, blocks=blocks, **LOWER)
    assert len(contexts) == 2 and contexts[0] is not contexts[1]
    assert all(context == ["first", "second"] for context in contexts)
    assert builds[0] is builds[1] and builds[2] is builds[3]
    shared, gets = [], []

    def getter(factory):
        gets.append(factory)
        return shared

    definitions = ({}, {})
    for _ in range(2):
        lowering._lower_qiskit(selected, blocks=blocks, **LOWER,
                               definition_cache=definitions, method_context=getter)
    assert shared == ["first", "second"] and len(gets) == 2
    error = RuntimeError("injected constructor context failure")

    def failed(factory):
        raise error

    with pytest.raises(RuntimeError) as raised:
        lowering._lower_qiskit(selected, blocks=blocks, **LOWER, method_context=failed)
    assert raised.value is error


@pytest.mark.parametrize("custom", (None, "target", "payload", "symbolic"))
def test_selected_binding_rejects_unbound_and_incompatible_live_access_before_sdk(monkeypatch, custom):
    from dataclasses import FrozenInstanceError
    from nwqlib.blocks.selection import SelectedBlock
    from nwqlib.core import Source

    base = select_zero_reflection("base", 1)
    other = select_zero_reflection("other", 1)
    imports, calls = [], []
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "qiskit" or name.startswith("qiskit."):
            imports.append(name)
            pytest.fail("invalid binding reached the SDK import boundary")
        return original_import(name, *args, **kwargs)

    def constructor(*args):
        calls.append(args)
        pytest.fail("binding or rejected lowering executed a constructor")

    monkeypatch.setattr(builtins, "__import__", guarded)
    if custom in ("target", "payload"):
        target = Source(name="test.reflection_target", version="2", domain="custom semantic target",
                        reference="the selected full-register reflection")
        implementation = Source(name="test.reflection_builder" if custom == "target" else "reflection.positive_zero",
                                version="2", domain="method-owned constructor", reference="immutable tuple payload")
        base = SelectedBlock.bind(base.record.revise(
            signature=base.record.signature.revise(target=target), implementation=implementation),
            payload=(1,), constructor=constructor)
    elif custom == "symbolic":
        port, = base.record.signature.quantum
        base = SelectedBlock.bind(base.record.revise(signature=base.record.signature.revise(
            quantum=(port.revise(width=ExprRef(expression="width")),))),
            payload=(1,), constructor=constructor)
    transformed = transform_block("controlled", base, control=True)
    inverse = transform_block("inverse", base, adjoint=True)
    for selected in (transformed, inverse):
        assert selected.record.signature.target == base.record.signature.target
        assert selected.record.implementation == base.record.implementation
        assert selected.record.base_selection_id == base.record.content_id and selected._base is base
    if custom != "target":
        # q=1 has the selected 4-unit base plus 12-unit controlled-extension law.
        assert transformed.record.construction_work == 16
    assert inverse.record.construction_work == base.record.construction_work
    if custom is None:
        port, = base.record.signature.quantum
        for invalid in (
            base.record.revise(semantics=base.record.semantics.revise(kind="unknown")),
            base.record.revise(semantics=base.record.semantics.revise(
                basis=base.record.semantics.basis.revise(dimension=3))),
            base.record.revise(signature=base.record.signature.revise(quantum=(port.revise(width=2),))),
        ):
            with pytest.raises(ValueError, match="reflection"):
                transform_block("invalid", SelectedBlock.bind(invalid, constructor=constructor), control=True)
    for record, fields in (
        (None, {"constructor": constructor}),
        (base.record, {"constructor": object()}),
        (base.record, {}),
        (transformed.record, {"constructor": constructor}),
        (transformed.record, {"base": base, "constructor": constructor}),
        (transformed.record, {"base": object()}),
        (transformed.record, {"base": other}),
    ):
        with pytest.raises((TypeError, ValueError)):
            SelectedBlock.bind(record, **fields)
    bound = SelectedBlock.bind(base.record, payload=base._payload, constructor=constructor)
    assert bound.record.content_id == base.record.content_id
    assert bound._payload is base._payload and bound._constructor is constructor
    with pytest.raises(FrozenInstanceError):
        bound._constructor = base._constructor
    blocked = SelectedBlock.bind(base.record.revise(blocker="explicitly unbound fixture"))
    with pytest.raises(ValueError):
        lower_qiskit(one(blocked), blocks=(blocked,), **LOWER)
    with pytest.raises(ValueError, match="exact persisted selected identities"):
        lower_qiskit(one(other), blocks=(bound,), **LOWER)
    assert not imports and not calls


def supplied_expectation(circuit):
    from nwqlib import Expectation, plan
    from nwqlib.algorithms import ExpectationMethod
    from nwqlib.core import InputRef, Source
    from nwqlib.problems import bind_preparation_circuit

    q = circuit.num_qubits
    source = Source(name="supplied fixture", version="1", domain="bounded unitary premise", reference="analytic Ry and phase")
    observable = ingest_pauli((("Z"*q, 1.),), num_qubits=q)
    state = bind_preparation_circuit(circuit, reference=InputRef(identity="declared:preparation", representation="circuit", source=source),
        basis=observable.manifest.basis)
    return state, plan(Expectation(state=state, observable=observable), method=ExpectationMethod(), seed=7)


def test_supplied_selected_preparation_forward_inverse_control_and_unknown_laws():
    from qiskit import QuantumCircuit
    from nwqlib.resources import ResourceContext, estimate

    theta, phi = .43, .17
    circuit = QuantumCircuit(1, global_phase=phi)
    circuit.ry(theta, 0)
    state, selected = supplied_expectation(circuit)
    base, = selected.blocks
    assert base.record.implementation.name == "preparation.supplied"
    assert base.record.decomposition is base.record.cost_law is None
    assert base.record.resource_laws == () and base.record.semantics.epsilon is None
    assert dict((x.parameter, x.value) for x in base.record.cost_parameters)["snapshot_work"] == state.preparation.work
    inverse = transform_block("inverse", base, adjoint=True)
    controlled = transform_block("controlled", base, control=True)
    c, s = np.cos(theta/2), np.sin(theta/2)
    u = np.exp(1j*phi)*np.array([[c, -s], [s, c]])
    initial = np.array([1, 1j])/np.sqrt(2)
    for block, expected in ((base, u@initial), (inverse, u.conj().T@initial)):
        result = lower_qiskit(one(block), blocks=(block,), **LOWER)
        np.testing.assert_allclose(action(result.circuit, initial), expected, atol=3e-14, rtol=0)
    initial_control = np.array([1, 1, 0, 0])/np.sqrt(2)
    expected = np.array([1, np.exp(1j*phi)*c, 0, np.exp(1j*phi)*s])/np.sqrt(2)
    result = lower_qiskit(one(controlled), blocks=(controlled,), **LOWER)
    np.testing.assert_allclose(action(result.circuit, initial_control), expected, atol=3e-14, rtol=0)
    for block in (base, inverse, controlled):
        assert block.record.cost_law is None and block.record.resource_laws == ()
        folded = estimate(one(block), context=ResourceContext())
        assert folded.quantity("cx").fact.availability == "unknown"
    # Same declaration does not authenticate another supplied native handle.
    another = select_preparation(base.record.signature.name, state)
    assert another.record.content_id != base.record.content_id
    restored = SelectedConstruction.model_validate_json(one(base).model_dump_json())
    with pytest.raises(ValueError, match="exact persisted selected identities"):
        lower_qiskit(restored, blocks=(another,), **LOWER)


@pytest.mark.parametrize("power", [.5, -.5, 0.])
def test_supplied_annotations_reach_common_preparation_hls_once(power, monkeypatch):
    from qiskit import QuantumCircuit
    from qiskit.circuit import AnnotatedOperation, ControlModifier, InverseModifier, PowerModifier
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib._prepared_execution import Run, prepare_experiment
    import nwqlib.blocks._qiskit_intake as intake

    theta, phi, phase = .4, .12, .08
    local = QuantumCircuit(1, global_phase=phi)
    local.ry(theta, 0)
    annotation = AnnotatedOperation(local.to_gate(),
        [ControlModifier(1, 0), InverseModifier(), PowerModifier(power)])
    circuit = QuantumCircuit(2, global_phase=phase)
    circuit.append(annotation, [0, 1], copy=False)
    _, plan = supplied_expectation(circuit)
    if power == .5:
        experiment, = plan.experiments
        plan = plan.revise(experiments=(experiment, experiment.revise(name="reused")))._bind(blocks=plan.blocks)
    copies = []
    snapshot = intake.snapshot_circuit
    def counted(*args, **kwargs):
        copies.append(args[0])
        return snapshot(*args, **kwargs)
    monkeypatch.setattr(intake, "snapshot_circuit", counted)
    run = Run(plan)
    initial = np.array([1, 1, 0, 0])/np.sqrt(2)
    # U has eigenphases within (-pi,pi), so fractional power uses the stated
    # branch: the open-control block is exp(-i*p*phi) Ry(-p*theta).
    expected = np.exp(1j*phase)*np.array([
        np.exp(-1j*power*phi)*np.cos(power*theta/2), 1,
        -np.exp(-1j*power*phi)*np.sin(power*theta/2), 0])/np.sqrt(2)
    for experiment in plan.experiments:
        prepared = prepare_experiment(plan.resolve(experiment.name), run=run,
                           runtime=RuntimeOptions(seed=7))
        inspected = prepared.inspect_circuit()
        assert inspected.data[-1].operation.name == "save_expval"
        inspected.data.pop()  # Aer acquisition bookkeeping has no unitary action.
        np.testing.assert_allclose(action(inspected, initial), expected, atol=8e-14, rtol=0)
        assert prepared.record.native_operations > 0
        assert prepared.record.execution == "quantum_circuit"
    assert len(copies) == 1  # Native definition cache reuses the same admitted supplied source.
    assert run.trace.events == ()
    run.close()


def test_equivalent_choice_changes_recipe_identity_and_actual_inventory():
    state = ingest_occupation("1", num_qubits=1)
    x = select_preparation("prep", state)
    hzh = select_preparation("prep", state, choice="hzh")
    px, ph = one(x), one(hzh)
    assert px.content_id != ph.content_id
    assert [item.gate for item in x.record.decomposition] == ["x"]
    assert [item.gate for item in hzh.record.decomposition] == ["h", "z", "h"]
    for block, program, expected_counts in ((x, px, {"x": 1}), (hzh, ph, {"h": 2, "z": 1})):
        loaded = SelectedConstruction.model_validate_json(program.model_dump_json())
        circuit = lower_qiskit(loaded, blocks=(block,), **LOWER).circuit.decompose()
        assert dict(circuit.count_ops()) == expected_counts
        for initial, expected in (([1, 0], [0, 1]), ([0, 1], [1, 0])):
            np.testing.assert_allclose(action(circuit, initial), expected, atol=2e-14, rtol=0)
    repeated = lower_qiskit(one(hzh, repeat=3), blocks=(hzh,), **LOWER).circuit.decompose()
    assert dict(repeated.count_ops()) == {"h": 6, "z": 3}


def test_parameter_calls_keep_distinct_action_and_only_recent_native_specializations(monkeypatch):
    from qiskit import QuantumCircuit
    from nwqlib.blocks import BlockSemantics, SelectedDefinition
    from nwqlib.blocks import lowering
    from nwqlib.blocks.selection import SelectedBlock
    from nwqlib.core import Basis, Float64, Source
    from nwqlib.ir import Argument, BlockSignature, Constant, QuantumPort

    source = Source(name="test.ry_phase", version="1", domain="one-qubit selected action",
                    reference="exp(i*theta/3) Ry(theta)")
    record = SelectedDefinition(signature=BlockSignature(name="rotation", target=source,
        quantum=(QuantumPort(name="q", width=1),), parameters=(Parameter(name="theta", domain="real"),)),
        semantics=BlockSemantics(kind="unknown", input=None,
            basis=Basis(identity="computational", dimension=2, ordering="q0 rightmost"),
            relation="exp(i*theta/3) Ry(theta)", input_projector="identity", output_projector="identity",
            success="deterministic", workspace=0, restoration="none", epsilon=None,
            approximation_metric="operator norm", approximation_evidence="native roundoff unknown",
            inverse_legal=False, control_legal=False, phase="physical exp(i*theta/3) kept"),
        implementation=source, choice="Ry and phase", decomposition=None, cost_law=None,
        cost_context="two construction slots per chosen angle", construction_work=2)
    built = []

    def build(selected, arguments, method_context):
        theta = dict(arguments)["theta"]
        built.append(theta)
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)
        circuit.global_phase = theta/3
        return circuit

    block = SelectedBlock.bind(record, constructor=build)

    def point(a, b):
        expressions = tuple(Expression(id=name, value=Constant(value=Float64(value=value)))
                            for name, value in (("a", a), ("b", b)))
        definitions = (Definition(id="allocate", node=Allocate(wire="q")),) + tuple(
            Definition(id=name, node=BlockCall(signature="rotation", ports=(PortMap(port="q", wire="q"),),
                arguments=(Argument(parameter="theta", value=ExprRef(expression=name)),))) for name in ("a", "b"))
        definitions += (Definition(id="root", node=Sequence(children=("allocate", "a", "b"))),)
        return SelectedConstruction(program=Program(root="root", definitions=definitions,
            registers=(Register(name="q", width=1),), signatures=(record.signature,), expressions=expressions),
            selections=(record,))

    previous, counts = {}, []
    for a, b in ((.2, .7), (.2, .7), (.2, -.7), (.2, .7)):
        current = {}
        logical = lowering._lower_qiskit(point(a, b), blocks=(block,), **LOWER,
                                         specialization_cache=(previous, current))
        # Two Ry rotations add, and their physical global phases add. A
        # signature-keyed argument overwrite changes both independent values.
        expected = np.exp(1j*(a+b)/3)*np.array([np.cos((a+b)/2), np.sin((a+b)/2)])
        np.testing.assert_allclose(action(logical.circuit, [1, 0]), expected, rtol=0, atol=2e-14)
        assert logical.construction_work == 8  # Two calls and two distinct definitions, two slots each.
        assert logical.defined_selections == (record.content_id,)  # One template, not one specialization.
        assert len(current) == 2
        previous = current
        counts.append(len(built))
    assert counts == [2, 2, 3, 4]
    with pytest.raises(ValueError, match="compact native"):
        transform_block("controlled", block, control=True)
    # Follow the same varying source gates through real shared Aer preparation.
    # Its strong-reference definition cache must not keep discarded trials after
    # the logical cache drops them. No backend acquisition is needed here.
    import nwqlib
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.core.planning import Experiment, ObservationSpec, RuntimeOptions
    from nwqlib._prepared_execution import Run, prepare_experiment

    base = nwqlib.plan(nwqlib.Expectation(
        state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)),
        method=ExpectationMethod(), seed=7)
    unbound = point(.2, .7)
    program = unbound.program.revise(parameters=tuple(Parameter(name=name, domain="real") for name in ("a", "b")),
        expressions=tuple(Expression(id=name, value=ParameterRef(parameter=name)) for name in ("a", "b")))
    unbound = unbound.revise(program=program)
    plan = base.revise(construction=unbound,
        experiments=(Experiment(name="rotation", setting="rotation",
                                observation=ObservationSpec(kind="pauli_expectation", labels=("Z",))),),
        error_model=base.error_model.revise(construction_id=unbound.content_id))._bind(blocks=(block,))
    run = Run(plan)
    sizes = []
    for a, b in ((.2, .7), (.2, .7), (.2, -.7), (.2, .7)):
        actual = plan.resolve("rotation", bindings=(Binding(parameter="a", value=Float64(value=a)),
                                                    Binding(parameter="b", value=Float64(value=b))))
        prepare_experiment(actual, run=run, runtime=RuntimeOptions(seed=7))
        sizes.append(sum(len(owner._blocks) for owner in run._state["backend_context"]["aer_preparations"].values()))
    assert sizes == [2, 2, 3, 3]  # Current/previous b plus the common a; no older b object.
    assert len(run._state["specializations"]) == 2 and not run.trace.events
    run.close()


def test_controlled_negative_x_and_complex_preparation_inverse_order():
    negative = select_signed_pauli("minus_x", ingest_pauli([("X", -2)], num_qubits=1))
    ctrl = transform_block("controlled", negative, control=True)
    circuit = lower_qiskit(one(ctrl), blocks=(ctrl,), **LOWER).circuit
    # Control is q0, system q1: (|00>+|01>)/sqrt(2) -> (|00>-|11>)/sqrt(2).
    initial = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    expected = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(action(circuit, initial), expected, atol=2e-14, rtol=0)
    # Basis |10> on q0-first occupation has amplitude i and printed ket |01>.
    state = ingest_vector(np.array([0, 1j, 0, 0], dtype=complex))
    prep = select_preparation("prep", state)
    controlled = transform_block("cprep", prep, control=True)
    inverse = transform_block("inverse", prep, control=True, adjoint=True)
    program = construction((controlled, inverse), (("cprep", (("control", "control"), ("system", "system"))),
                           ("inverse", (("control", "control"), ("system", "system")))), (("control", 1), ("system", 2)))
    input_state = np.zeros(8, dtype=complex)
    input_state[:2] = 1 / np.sqrt(2)
    forward = lower_qiskit(one(controlled), blocks=(controlled,), **LOWER).circuit
    expected_state = np.zeros(8, dtype=complex)
    expected_state[0], expected_state[3] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    np.testing.assert_allclose(action(forward, input_state), expected_state, atol=3e-14, rtol=0)
    circuit = lower_qiskit(program, blocks=(controlled, inverse), **LOWER).circuit
    np.testing.assert_allclose(action(circuit, input_state), input_state, atol=3e-14, rtol=0)


def test_signed_y_identity_projected_block_and_padding():
    operator = ingest_pauli([("X", -2), ("Y", 1), ("I", 3)], num_qubits=1)
    select = select_signed_pauli("select", operator)
    prep = select_pauli_preparation("prep", select)
    inverse = transform_block("unprep", prep, adjoint=True)
    blocks = (prep, select, inverse)
    program = construction(blocks, (("prep", (("system", "index"),)),
        ("select", (("index", "index"), ("system", "system"))),
        ("unprep", (("system", "index"),))), (("index", 2), ("system", 1)))
    program = program.revise(encodings=(PauliEncoding(root="body", select="select", prepare="prep", unprepare="unprep"),))
    assert program.encoding_semantics("body").kind == "block_encoding"
    assert program.encoding_semantics("body").alpha == 6
    circuit = lower_qiskit(program, blocks=blocks, **LOWER).circuit
    # A/6 = [[3,-2-i],[-2+i,3]]/6; index-zero entries are positions 0,4.
    expected = np.array([[3, -2 - 1j], [-2 + 1j, 3]]) / 6
    for column in range(2):
        initial = np.zeros(8, dtype=complex)
        initial[column * 4] = 1
        np.testing.assert_allclose(action(circuit, initial)[[0, 4]], expected[:, column], atol=4e-14, rtol=0)
    selected_circuit = lower_qiskit(one(select), blocks=(select,), **LOWER).circuit
    for index in (3, 7):
        initial = np.zeros(8)
        initial[index] = 1
        np.testing.assert_allclose(action(selected_circuit, initial), initial, atol=3e-14, rtol=0)
    assert select.record.semantics.alpha == 6
    readout = select_pauli_readout("readout", select)
    circuit = lower_qiskit(one(readout), blocks=(readout,), **LOWER).circuit
    initial = np.zeros(8, dtype=complex)
    initial[1], initial[5] = 1 / np.sqrt(2), 1j / np.sqrt(2)  # positive Y eigenstate in label 1
    expected_readout = np.zeros(8, dtype=complex)
    expected_readout[1] = 1
    np.testing.assert_allclose(action(circuit, initial), expected_readout, atol=3e-14, rtol=0)
    padding = np.zeros(8)
    padding[3] = 1
    np.testing.assert_allclose(action(circuit, padding), padding, atol=3e-14, rtol=0)
    assert "Pi_used SELECT Pi_used" in readout.record.semantics.relation
    assert "zero weight" in readout.record.semantics.output_projector


def test_identity_zero_reflection_and_alpha_revision_boundary():
    identity = select_signed_pauli("identity", ingest_pauli([("I", -3)], num_qubits=1))
    circuit = lower_qiskit(one(identity), blocks=(identity,), **LOWER).circuit
    np.testing.assert_allclose(action(circuit, [1, 0]), [-1, 0], atol=2e-14, rtol=0)
    with pytest.raises(ValueError, match="zero Pauli"):
        select_signed_pauli("zero", ingest_pauli([("X", 0)], num_qubits=1))
    for width in (0, 1, 2):
        reflection = select_zero_reflection("reflection", width)
        circuit = lower_qiskit(one(reflection), blocks=(reflection,),
            max_qubits=width, max_clbits=0).circuit
        dimension = 1 << width
        initial = np.ones(dimension) / np.sqrt(dimension)
        expected = -initial.copy()
        expected[0] = initial[0]
        np.testing.assert_allclose(action(circuit, initial), expected, atol=3e-14, rtol=0)
        if width == 0:
            controlled = transform_block("controlled_empty", reflection, control=True)
            circuit = lower_qiskit(one(controlled), blocks=(controlled,), max_qubits=1, max_clbits=0).circuit
            np.testing.assert_allclose(action(circuit, [1, 0]), [1, 0], atol=3e-14, rtol=0)
    changed = identity.record.revise(semantics=identity.record.semantics.revise(alpha=6.0))
    program = one(identity).revise(selections=(changed,))
    with pytest.raises(ValueError, match="exact persisted"):
        lower_qiskit(program, blocks=(identity,), **LOWER)


def test_reject_before_sdk_import_and_no_repeat_expansion(monkeypatch):
    state = ingest_occupation("1", num_qubits=1)
    prep = select_preparation("prep", state)
    huge = one(prep, repeat=10**12)
    assert len(huge.program.definitions) == 5
    assert len(huge.model_dump_json()) < 20000
    original = builtins.__import__

    def forbidden(name, *args, **kwargs):
        assert not name.startswith("qiskit"), "SDK import before admission"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden)
    with pytest.raises(ValueError, match="max_operations"):
        lower_qiskit(huge, blocks=(prep,), **LOWER)
    with pytest.raises(ValueError, match="width"):
        lower_qiskit(one(prep), blocks=(prep,), max_qubits=0, max_clbits=0)
    bad_signature = prep.record.signature.revise(quantum=(prep.record.signature.quantum[0].revise(ensures="zero"),))
    with pytest.raises(ValueError, match="signature differs"):
        one(prep).revise(program=one(prep).program.revise(signatures=(bad_signature,)))


def test_generic_phase_preparation_law_is_not_exact_inventory():
    state = ingest_vector(np.array([1, 1j], dtype=complex))
    block = select_preparation("prep", state)
    assert block.record.choice == "native:magnitude_phase"
    assert block.record.decomposition is None
    assert "not an exact inventory" in block.record.cost_context
    circuit = lower_qiskit(one(block), blocks=(block,), **LOWER).circuit
    np.testing.assert_allclose(action(circuit, [1, 0]), np.array([1, 1j]) / np.sqrt(2), atol=3e-14, rtol=0)
    inverse = transform_block("inverse", block, adjoint=True)
    assert inverse.record.semantics.kind == "unitary_transform"
    assert inverse.record.semantics.base_semantics == block.record.semantics
    assert inverse.record.semantics.epsilon is None
    # For U=diag(1,i) RY(pi/2), U†|0>=(|0>-|1>)/sqrt(2), not U|0>.
    inverse_circuit = lower_qiskit(one(inverse), blocks=(inverse,), **LOWER).circuit
    np.testing.assert_allclose(action(inverse_circuit, [1, 0]), np.array([1, -1]) / np.sqrt(2), atol=3e-14, rtol=0)
    assert Counter(item.gate for item in select_preparation("basis", ingest_occupation("0", num_qubits=1)).record.decomposition) == {}


def test_product_law_scope_and_pauli_selection_byte_admission(monkeypatch):
    product = ingest_product(np.array([[1, 1j], [2, 1]], dtype=complex))
    prep = select_preparation("product", product)
    controlled = transform_block("controlled", prep, control=True)
    assert prep.record.cost_law.name == "direct_preparation_cx_bound"
    assert controlled.record.cost_law.name == "direct_preparation_controlled_cx_bound"
    assert {item.parameter: item.value for item in prep.record.cost_parameters} == {"num_qubits": 1, "multiplicity": 2}
    assert controlled.record.cost_parameters == prep.record.cost_parameters
    operator = ingest_pauli([("X", 1), ("Y", 2)], num_qubits=1)
    import nwqlib.blocks.selection as selection
    with monkeypatch.context() as patch:
        patch.setattr(selection.np, "zeros", lambda *a, **k: pytest.fail("unadmitted Pauli array allocation"))
        with pytest.raises(ValueError, match="max_bytes"):
            select_signed_pauli("select", operator, max_bytes=1)
    selected = select_signed_pauli("select", operator)
    prep = select_pauli_preparation("prep", selected)
    assert prep.record.coefficient_selection_id == selected.record.content_id


def test_generic_preparation_rejects_a_family_without_its_own_constructor_and_law():
    from nwqlib.problems.inputs import StateInput

    product = ingest_product(np.array([[1, 1j], [2, 1]], dtype=complex))
    legal = select_preparation("product", product)
    assert {binding.parameter: binding.value for binding in legal.record.cost_parameters} == {
        "num_qubits": 1, "multiplicity": 2}
    record = product.to_record()
    record["preparation"]["implementation"] = "qiskit.mps_circuit"
    with pytest.raises(ValueError, match="no native constructor and cost law"):
        select_preparation("mps", StateInput.from_record(record))


@pytest.mark.parametrize("index_qubits", range(1, 5))
def test_native_cx_envelopes_cover_uniform_preparation_and_larger_select_tables(index_qubits):
    from qiskit import transpile
    from nwqlib._preparation_laws import direct_preparation_controlled_cx_bound

    amplitudes = np.ones(1 << index_qubits, complex)
    if index_qubits > 1:
        amplitudes[-1] = 0.  # Non-power-of-two prefix uses UniformSuperpositionGate.
    prep = transform_block("controlled_prefix", select_preparation("prefix", ingest_vector(amplitudes)), control=True)
    system_qubits = (index_qubits + 1) // 2
    terms = tuple(("".join("IXYZ"[(index >> (2*bit)) & 3] for bit in range(system_qubits)),
                   (-1)**index * (index+1)) for index in range(1 << index_qubits))
    selected = select_signed_pauli("select", ingest_pauli(terms, num_qubits=system_qubits))
    controlled = transform_block("controlled_select", selected, control=True)
    for block, bound in ((prep, direct_preparation_controlled_cx_bound(index_qubits)),
                         (controlled, signed_pauli_cx_bound(index_qubits, system_qubits, controlled=True))):
        circuit = lower_qiskit(one(block), blocks=(block,), max_operations=100000,
                               max_qubits=1+index_qubits+system_qubits, max_clbits=0).circuit
        native = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
        assert native.count_ops().get("cx", 0) <= bound


def test_controlled_generic_inventory_satisfies_native_slot_bound():
    from nwqlib._preparation_laws import direct_preparation_controlled_cx_bound
    state = ingest_vector(np.array([1, 2j, 3, -4j], dtype=complex))
    controlled = transform_block("controlled", select_preparation("prep", state), control=True)
    circuit = lower_qiskit(one(controlled), blocks=(controlled,), **LOWER).circuit.decompose(reps=10)
    # Two magnitude/phase trees: six rotation slots and four CX slots;
    # control costs at most 2 CX per rotation and 6 per CX, hence 36.
    assert direct_preparation_controlled_cx_bound(2) == 36
    assert circuit.count_ops().get("cx", 0) <= 36
    assert set(circuit.count_ops()) <= {"u", "cx"}
    select = select_signed_pauli("select", ingest_pauli([("X", -1), ("Y", 2)], num_qubits=1))
    ctrl_select = transform_block("controlled_select", select, control=True)
    circuit = lower_qiskit(one(ctrl_select), blocks=(ctrl_select,), **LOWER).circuit.decompose(reps=10)
    # Two labels: three UCG CX, two core unitaries, three completion
    # rotations, and one sign rotation. Control gives <=6*3+2*(2+3+1)=30 CX.
    assert signed_pauli_cx_bound(1, 1, controlled=True) == 30
    assert circuit.count_ops().get("cx", 0) <= 30
    assert set(circuit.count_ops()) <= {"u", "cx"}


def test_lowering_reuses_native_admission_and_has_explicit_measurement_layout(monkeypatch):
    import nwqlib.operators.inputs as operators
    import nwqlib.problems.inputs as states
    block = select_preparation("prep", ingest_occupation("10", num_qubits=2))
    selected = one(block)
    root = next(item for item in selected.program.definitions if item.id == "root")
    definitions = tuple(item if item.id != "root" else root.revise(node=Sequence(children=root.node.children + ("measure",)))
                        for item in selected.program.definitions)
    program = selected.program.revise(definitions=definitions + (Definition(id="measure", node=Measure(wire="system", result="readout")),),
        classical=(ClassicalValue(name="readout", dtype="bits", width=2),))
    selected = selected.revise(program=program)

    def forbidden(*args, **kwargs):
        pytest.fail("lowering repeated admission of the input")

    # Every problems.inputs admission hashes through its own _digest binding.
    # A lazy import of the hash reads the operators.inputs attribute instead.
    for owner in (operators, states):
        monkeypatch.setattr(owner, "_digest", forbidden)
    artifact = lower_qiskit(selected, blocks=(block,), **LOWER)
    assert artifact.quantum_layout == (("system", (0, 1)),)
    assert artifact.measurement_layout == (("readout", (0, 1)),)
    measured = [(artifact.circuit.find_bit(item.qubits[0]).index, artifact.circuit.find_bit(item.clbits[0]).index)
                for item in artifact.circuit.data if item.operation.name == "measure"]
    assert measured == [(0, 0), (1, 1)]


def test_selected_experiment_reaches_lowering_and_unused_definitions_stay_unbuilt(monkeypatch):
    """Zero repeats and unused definitions must not build gates, while control/inverse share one
    realized base.
    """
    from nwqlib.blocks.selection import SelectedBlock
    from nwqlib.core import InputRef, Source
    prep = select_preparation("prep", ingest_occupation("1", num_qubits=1))
    unused = select_zero_reflection("unused", 2)
    selected = construction((prep, unused), (("prep", (("system", "system"),)),), (("system", 1),))
    source = Source(name="bounded test setting", version="1", domain="test", reference="test:setting")
    metadata = MetadataRef(format=source, data=InputRef(identity="test:readout", representation="setting", source=source))
    definitions = []
    for definition in selected.program.definitions:
        if definition.id == "repeat":
            definition = definition.revise(node=Repeat(body="body", count=ExprRef(expression="n")))
        elif definition.id == "root":
            definition = definition.revise(node=Sequence(children=definition.node.children + ("release",)))
        definitions.append(definition)
    definitions.extend((Definition(id="release", node=Release(wire="system")),
        Definition(id="batch", node=MeasurementBatch(body="root", settings=tuple(
            Setting(label="same label", bindings=(Binding(parameter="n", value=n),), metadata=metadata) for n in (0, 3)
        )))))
    program = selected.program.revise(root="batch", definitions=tuple(definitions),
        parameters=(Parameter(name="n", domain="integer", lower=0),),
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),))
    with pytest.raises(ValueError, match="select one static experiment"):
        lower_qiskit(selected.revise(program=program), blocks=(prep,), **LOWER)
    for point, expected in ((0, {}), (1, {"x": 3})):
        chosen = selected.revise(program=program.select_experiment("batch", point), selections=(prep.record,))
        artifact = lower_qiskit(chosen, blocks=(prep,), **LOWER)
        assert dict(artifact.circuit.decompose().count_ops()) == expected
        assert artifact.defined_selections == (() if point == 0 else (prep.record.content_id,))
        assert unused.record.content_id not in artifact.defined_selections
    constructor = prep._constructor
    builds = []

    def build(block, arguments, method_context):
        assert arguments == ()
        builds.append(block.record.content_id)
        return constructor(block, arguments, method_context)

    prep = SelectedBlock.bind(prep.record, payload=prep._payload, constructor=build)
    ctrl = transform_block("ctrl", prep, control=True)
    inv = transform_block("inv", prep, control=True, adjoint=True)
    pair = construction((ctrl, inv), (("ctrl", (("control", "control"), ("system", "system"))),
                                    ("inv", (("control", "control"), ("system", "system")))), (("control", 1), ("system", 1)))
    import nwqlib.subroutines.qiskit_compat as compatibility
    original = compatibility.controlled
    controls = []

    def count_control(*args, **kwargs):
        controls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(compatibility, "controlled", count_control)
    artifact = lower_qiskit(pair, blocks=(ctrl, inv), **LOWER)
    assert controls == [1]
    assert builds == [prep.record.content_id]
    assert set(artifact.defined_selections) == {prep.record.content_id, ctrl.record.content_id, inv.record.content_id}


def _dense_encoding_block():
    from nwqlib.blocks import select_block_encoding
    from nwqlib.operators import ingest_dense
    from nwqlib.subroutines.block_encoding import build_block_encoding

    matrix = np.array([[0.3, 0.1], [0.1, -0.2]], dtype=complex)
    return select_block_encoding("dense", build_block_encoding(matrix, implementation="dense_dilation"),
                                 operator=ingest_dense(matrix))


def test_controlled_dense_encoding_reserves_its_exact_synthesis():
    # Controlling a dense-dilation encoding synthesizes its unitary on the
    # ancilla and system qubits exactly, and Qiskit then controls each
    # synthesized gate. The transformed record adds both laws to its
    # construction work.
    from test_dense_synthesis import _dense_synthesis_work
    from test_synthesis_admission import _gatewise_control_work

    base = _dense_encoding_block()
    controlled = transform_block("controlled", base, control=True)
    assert controlled.record.construction_work == (base.record.construction_work + _dense_synthesis_work(2)
                                                   + _gatewise_control_work(2, 1))


def test_lowering_synthesizes_a_controlled_dense_encoding_once(monkeypatch):
    # A Program that calls the controlled block twice synthesizes it once,
    # because lowering builds the controlled base once. An adjoint without
    # control synthesizes nothing and keeps the base's construction work.
    from test_dense_synthesis import _counted_syntheses

    base = _dense_encoding_block()
    controlled = transform_block("controlled", base, control=True)
    inverse = transform_block("inverse", base, adjoint=True)
    assert inverse.record.construction_work == base.record.construction_work
    ports = tuple((port.name, port.name) for port in controlled.record.signature.quantum)
    program = construction((controlled,), (("controlled", ports), ("controlled", ports)),
                           tuple((port.name, port.width) for port in controlled.record.signature.quantum))
    calls = _counted_syntheses(monkeypatch)
    lower_qiskit(program, blocks=(controlled,), **LOWER)
    assert calls == [2]


def test_lowering_admits_a_controlled_dense_encoding_before_its_synthesis(monkeypatch):
    # Lowering charges the exact synthesis of the controlled base and
    # Qiskit's control of the synthesized gates against max_synthesis_work
    # before the synthesis starts. One unit below the two laws refuses, and
    # the laws themselves admit it.
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_synthesis_admission import _gatewise_control_work

    controlled = transform_block("controlled", _dense_encoding_block(), control=True)
    ports = tuple((port.name, port.name) for port in controlled.record.signature.quantum)
    program = construction((controlled,), (("controlled", ports),),
                           tuple((port.name, port.width) for port in controlled.record.signature.quantum))
    need = _dense_synthesis_work(2) + _gatewise_control_work(2, 1)
    calls = _counted_syntheses(monkeypatch)
    with pytest.raises(ValueError, match="max_synthesis_work"):
        lower_qiskit(program, blocks=(controlled,), max_synthesis_work=need - 1, **LOWER)
    assert calls == []
    lower_qiskit(program, blocks=(controlled,), max_synthesis_work=need, **LOWER)
    assert calls == [2]
