"""Supplied native execution snapshot, with no simulation or operator extraction."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import (AnnotatedOperation, ControlModifier, Gate, Instruction,
                           InverseModifier, Parameter, PowerModifier)
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import Isometry, StatePreparation, UnitaryGate

from nwqlib.blocks import select_block_encoding
from nwqlib.blocks._qiskit_intake import snapshot_circuit
from nwqlib.blocks.encoding import construct_block_encoding
from nwqlib.operators import ingest_dense
from nwqlib.subroutines.block_encoding.core import BlockEncoding


@pytest.mark.parametrize("first_angle", [.9, -.2])
def test_snapshot_temporary_gate_sources_keep_distinct_action_and_control_phase(first_angle):
    from qiskit.quantum_info import Operator
    from nwqlib.problems.inputs import prepare_qiskit, state_input

    def ry(angle):
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -s], [s, c]])

    circuit = QuantumCircuit(2, global_phase=.31)
    circuit.ry(first_angle, 0)
    circuit.ry(-.4, 1)
    circuit.rz(1.3, 0)
    rz = np.diag([np.exp(-.65j), np.exp(.65j)])
    expected = np.exp(.31j)*np.kron(ry(-.4), rz @ ry(first_angle))
    # CircuitInstruction can expose ephemeral Python gate objects. Compare the
    # full action, not object ids or gate names that may accidentally agree.
    first, size = snapshot_circuit(circuit)
    second, repeated_size = snapshot_circuit(first)
    assert size == repeated_size
    supplied = state_input(circuit)
    prepared = prepare_qiskit(supplied).circuit
    selected = select_block_encoding("rotations", BlockEncoding(circuit=circuit,
        alpha=1., num_ancillas=0, system_qubits=2, error_bound=0.,
        implementation="supplied", metadata={}), operator=ingest_dense(expected))
    frozen = construct_block_encoding(selected).circuit
    circuit.clear()
    for snapshot in (first, second, prepared, frozen):
        np.testing.assert_allclose(Operator(snapshot).data, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(Operator(first.inverse()).data, expected.conj().T,
                               rtol=0, atol=2e-14)
    # The extra low-order control makes physical global phase relative to the
    # inactive identity block; equivalence modulo phase would miss this defect.
    controlled = first.to_gate().control(1)
    expected_control = np.eye(8, dtype=complex)
    expected_control[1::2, 1::2] = expected
    np.testing.assert_allclose(Operator(controlled).data, expected_control,
                               rtol=0, atol=2e-14)



def test_native_selection_copies_stored_action_once_without_synthesis(monkeypatch):
    """Poison synthesis and mutate supplied gates to verify one detached shared copy including
    nested phase.
    """
    operator = ingest_dense(np.diag([1., -1.]), max_bytes=1_000_000)
    leaf = UnitaryGate(np.diag([1., -1.]).astype(complex), check_input=False)
    definition = QuantumCircuit(1, global_phase=.125)
    definition.append(leaf, [0], copy=False)
    gate = Gate("stored", 1, [])
    gate._definition = definition
    circuit = QuantumCircuit(1, global_phase=.25)
    circuit.append(gate, [0], copy=False)
    circuit.barrier()
    circuit.append(gate, [0], copy=False)
    circuit.metadata = {"ignored": object()}
    encoding = BlockEncoding(circuit=circuit, alpha=2., num_ancillas=0,
        system_qubits=1, error_bound=.03, implementation="supplied", metadata={})
    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot caused synthesis or native unitary validation")
    monkeypatch.setattr(UnitaryGate, "_define", forbidden)
    monkeypatch.setattr(UnitaryGate, "__init__", forbidden)
    selected = select_block_encoding("oracle", encoding, operator=operator, max_bytes=1_000_000)
    frozen = construct_block_encoding(selected)
    assert frozen.alpha == 2. and frozen.error_bound == .03
    assert selected.record.semantics.input == operator.manifest.reference
    assert selected.record.semantics.index_qubits == 0
    assert frozen.circuit.global_phase == .25 and not frozen.circuit.metadata
    assert len(frozen.circuit.data) == 2
    first, second = (item.operation for item in frozen.circuit.data)
    assert first is second and first is not gate
    assert first._definition.global_phase == .125
    frozen_leaf = first._definition.data[0].operation
    assert frozen_leaf._definition is None
    leaf.params[0][0, 0] = -1
    definition.clear()
    circuit.clear()
    assert frozen_leaf.params[0][0, 0] == 1
    assert len(first._definition.data) == 1 and len(frozen.circuit.data) == 2


def test_snapshot_compact_labels_and_stored_instruction_consumer():
    label = StatePreparation("r-")
    nested = Instruction("stored_instruction", 2, 0, [])
    nested._definition = QuantumCircuit(2)
    nested._definition.h(0)
    outer = Gate("outer", 2, [])
    outer._definition = QuantumCircuit(2)
    outer._definition.append(nested, [0, 1], copy=False)
    circuit = QuantumCircuit(2)
    circuit.append(label, [0, 1], copy=False)
    circuit.append(outer, [0, 1], copy=False)
    frozen, _ = snapshot_circuit(circuit, max_bytes=1_000_000)
    copied = frozen.data[0].operation
    assert copied._params_arg == "r-" and copied.params == ["r", "-"]
    assert copied._definition is None and copied.inverse()._params_arg == "r-"
    nested._definition.x(1)
    assert len(frozen.data[1].operation._definition.data[0].operation._definition.data) == 1
    assert frozen.to_gate().num_qubits == 2
    for instruction in (nested, Isometry(np.array([1., 0.]), 0, 0)):
        invalid = QuantumCircuit(instruction.num_qubits)
        invalid.append(instruction, range(instruction.num_qubits), copy=False)
        with pytest.raises(ValueError, match="top-level.*gates"):
            snapshot_circuit(invalid, max_bytes=1_000_000)


def test_snapshot_admission_precedes_copy_and_rejects_recursion(monkeypatch):
    circuit = QuantumCircuit(1)
    gate = Gate("cycle", 1, [])
    gate._definition = circuit
    circuit.append(gate, [0], copy=False)
    with pytest.raises(ValueError, match="recursive"):
        snapshot_circuit(circuit, max_bytes=1_000_000)
    def forbidden(*args, **kwargs):
        raise AssertionError("copy occurred before admission")
    monkeypatch.setattr("nwqlib.blocks._qiskit_intake.copy", forbidden)
    with pytest.raises(ValueError, match="bytes"):
        snapshot_circuit(circuit, max_bytes=1)


def test_annotation_snapshot_preserves_shared_stored_base_and_modifier_order(monkeypatch):
    base = RYGate(.6)
    modifiers = [PowerModifier(.5), InverseModifier(), ControlModifier(np.int64(1), "0")]
    annotated = AnnotatedOperation(base, modifiers)
    circuit = QuantumCircuit(2, global_phase=.2)
    circuit.append(annotated, [0, 1], copy=False)
    circuit.append(annotated, [0, 1], copy=False)
    def forbidden(*a, **k):
        raise AssertionError("intake attempted annotation synthesis")
    for name in ("control", "inverse", "power", "to_matrix"):
        monkeypatch.setattr(AnnotatedOperation, name, forbidden)
        monkeypatch.setattr(RYGate, name, forbidden)
    operator = ingest_dense(np.eye(4), max_bytes=1_000_000)
    encoding = BlockEncoding(circuit=circuit, alpha=1., num_ancillas=0,
        system_qubits=2, error_bound=0., implementation="supplied", metadata={})
    selected = select_block_encoding("annotated", encoding, operator=operator, max_bytes=1_000_000)
    copied = construct_block_encoding(selected).circuit
    first, second = (item.operation for item in copied.data)
    assert first is second and first is not annotated and first.base_op is not base
    assert [type(m) for m in first.modifiers] == [PowerModifier, InverseModifier, ControlModifier]
    modifiers[0].power = 0
    modifiers[-1].ctrl_state = 1
    base.params[0] = 1.2
    assert first.modifiers[0].power == .5 and first.modifiers[-1].ctrl_state == 0
    assert type(first.modifiers[-1].num_ctrl_qubits) is int
    assert first.base_op.params == [.6] and copied.global_phase == .2


def test_annotation_invalid_data_and_budget_reject_before_base_copy(monkeypatch):
    cases = []
    for power in (np.nan, np.inf, 1j, True, Parameter("power"), np.array([.5])):
        cases.append(AnnotatedOperation(RYGate(.2), [PowerModifier(power)]))
    for count, state in ((True, 1), (-1, 0), (1, 2), (1, "00"), (1, -1), (1, object()), (2, 0)):
        modifier = ControlModifier(1)
        modifier.num_ctrl_qubits, modifier.ctrl_state = count, state
        cases.append(AnnotatedOperation(RYGate(.2), [modifier]))
    cases += [AnnotatedOperation(Instruction("not_unitary", 1, 0, []), []),
              AnnotatedOperation(RYGate(.2), [object()])]
    circuit = QuantumCircuit(2)
    legal = AnnotatedOperation(RYGate(.2), [ControlModifier(1)])
    circuit.append(legal, [0, 1], copy=False)
    def forbidden(*a, **k):
        raise AssertionError("invalid annotation copied its base")
    monkeypatch.setattr("nwqlib.blocks._qiskit_intake.copy", forbidden)
    for invalid in cases:
        # Mutate an already stored SDK operation without asking its recursive
        # width properties to validate the invalid graph before the intake.
        legal.base_op, legal.modifiers = invalid.base_op, invalid.modifiers
        with pytest.raises(ValueError):
            snapshot_circuit(circuit, max_bytes=1_000_000)
    legal.base_op, legal.modifiers = RYGate(.2), [PowerModifier(.5)] * 50
    with pytest.raises(ValueError, match="bytes"):
        snapshot_circuit(circuit, max_bytes=512)
    legal.modifiers = []
    legal.base_op = legal
    with pytest.raises(ValueError, match="recursive"):
        snapshot_circuit(circuit, max_bytes=1_000_000)


def test_native_gate_nonfinite_parameter_rejects_before_copy(monkeypatch):
    circuit = QuantumCircuit(1)
    gate = RYGate(.2)
    circuit.append(gate, [0], copy=False)
    gate.params[0] = np.nan
    monkeypatch.setattr("nwqlib.blocks._qiskit_intake.copy",
                        lambda *a, **k: pytest.fail("invalid native parameter reached copy"))
    with pytest.raises(ValueError, match="finite"):
        snapshot_circuit(circuit, max_bytes=1_000_000)


def test_native_array_budget_precedes_finite_scan_and_copy(monkeypatch):
    circuit = QuantumCircuit(1)
    circuit.append(UnitaryGate(np.eye(2, dtype=complex)), [0], copy=False)
    # Census: circuit 32+16, global phase 16, instruction 16+8, gate 32 and
    # the 64-byte complex array, 184 bytes. The finite-check mask is freed
    # before the array is copied, so exactly 184 bytes admit the snapshot.
    _, census = snapshot_circuit(circuit, max_bytes=184)
    assert census == 184
    # One byte less rejects after the scalar objects fit, before any matrix work.
    monkeypatch.setattr("nwqlib.blocks._qiskit_intake.np.isfinite",
                        lambda *a, **k: pytest.fail("array work started before its budget"))
    with pytest.raises(ValueError, match="bytes"):
        snapshot_circuit(circuit, max_bytes=183)


def family_encoding(family,metadata):
    return BlockEncoding(circuit=QuantumCircuit(2),alpha=2.,num_ancillas=0,
        system_qubits=2,error_bound=0.,implementation=family,metadata=metadata)


def test_native_family_census_is_frozen_and_uses_only_known_scalars(monkeypatch):
    from nwqlib.backends import resources
    from nwqlib.blocks.records import SelectedDefinition
    class Unrelated:
        def __deepcopy__(self,memo):
            raise AssertionError("unrelated metadata copied")
        def __getitem__(self,key):
            raise AssertionError("unrelated metadata traversed")
    counts=dict(ucrz_multiplexor_gates=np.int64(2),ucrz_basis_cx_per_gate=3,
        control_diagonal_basis_cx=5,prep_pair_cx=7,unrelated=Unrelated())
    original=resources.block_encoding_per_query_cx
    calls=[]
    def once(detail,**kwargs):
        calls.append(dict(detail))
        return original(detail,**kwargs)
    monkeypatch.setattr(resources,"block_encoding_per_query_cx",once)
    selected=select_block_encoding("oracle",family_encoding("banded",{"periodic_gate_counts":counts}),
        operator=ingest_dense(np.eye(4),max_bytes=1_000_000),max_bytes=1_000_000)
    law,=selected.record.resource_laws
    assert law.value.value==28. # 2*2*(2-1) + 6*(2//2) + 2*3 + 5 + 7
    assert law.interpretation=="estimate" and law.evidence.kind=="user_assertion"
    assert law.bindings==selected.record.cost_parameters
    assert len(calls)==1 and len(calls[0])==4
    counts["prep_pair_cx"]=999
    assert law.value.value==28. and not construct_block_encoding(selected).metadata
    assert SelectedDefinition.model_validate_json(selected.record.model_dump_json())==selected.record


@pytest.mark.parametrize("family",["pauli_lcu","multiplexed_pauli","dense_dilation","supplied"])
def test_native_family_law_preserves_asserted_domain(family):
    from types import MappingProxyType
    metadata=MappingProxyType({}) if family=="dense_dilation" else dict(
        ucg_core_cx=2,ucg_completion_diagonal_cx=3,coefficient_diagonal_cx=5,prep_pair_cx=7)
    selected=select_block_encoding("oracle",family_encoding(family,metadata),
        operator=ingest_dense(np.eye(4),max_bytes=1_000_000),max_bytes=1_000_000)
    if family=="supplied":
        assert selected.record.resource_laws==()
    else:
        law,=selected.record.resource_laws
        # Three-qubit QSD: (23*64 - 72*8 + 64)/48 = 20,
        # including the lower-order terms rather than only 23*64/48.
        assert law.value.value==pytest.approx(20. if family=="dense_dilation" else 17.,rel=1e-15, abs=0)
        assert law.evidence.kind=="user_assertion"
        if family=="dense_dilation":
            assert "SBM arXiv:quant-ph/0406176v5 Table 1 QSD count, which the exact synthesis can exceed" in law.assumptions
            assert law.evidence.source.version == "2"


def test_native_family_unavailable_projection_and_containers_stay_executable():
    from types import MappingProxyType
    operator=ingest_dense(np.eye(4),max_bytes=1_000_000)
    for metadata in ({},MappingProxyType({}),{"periodic_gate_counts":MappingProxyType({})},
            dict(ucrz_multiplexor_gates=1<<1100,ucrz_basis_cx_per_gate=1,control_diagonal_basis_cx=0,prep_pair_cx=0)):
        selected=select_block_encoding("oracle",family_encoding("banded",metadata),operator=operator,max_bytes=1_000_000)
        assert selected.record.resource_laws==() and construct_block_encoding(selected).circuit.num_qubits==2


@pytest.mark.parametrize("invalid",[-1,1.9,True,np.bool_(True)])
def test_native_family_invalid_counts_reject_before_snapshot(monkeypatch,invalid):
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid census reached snapshot")
    monkeypatch.setattr("nwqlib.blocks._qiskit_intake.snapshot_circuit",forbidden)
    metadata=dict(ucg_core_cx=invalid,ucg_completion_diagonal_cx=3,coefficient_diagonal_cx=5,prep_pair_cx=7)
    with pytest.raises(ValueError,match="integer"):
        select_block_encoding("oracle",family_encoding("pauli_lcu",metadata),
            operator=ingest_dense(np.eye(4),max_bytes=1_000_000),max_bytes=1_000_000)


def test_native_family_conversion_callbacks_reject_before_snapshot(monkeypatch):
    class Callback:
        def __index__(self):
            raise AssertionError("untrusted conversion invoked")
        def bit_length(self):
            raise AssertionError("untrusted size invoked")
    def forbidden(*args,**kwargs):
        raise AssertionError("unadmitted census reached snapshot")
    monkeypatch.setattr("nwqlib.blocks._qiskit_intake.snapshot_circuit",forbidden)
    operator=ingest_dense(np.eye(4),max_bytes=1_000_000)
    metadata=dict(ucg_core_cx=Callback(),ucg_completion_diagonal_cx=3,coefficient_diagonal_cx=5,prep_pair_cx=7)
    with pytest.raises(ValueError,match="integer"):
        select_block_encoding("oracle",family_encoding("pauli_lcu",metadata),operator=operator,max_bytes=1_000_000)
