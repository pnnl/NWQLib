"""Supplied native execution snapshot, with no simulation or operator extraction."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import AnnotatedOperation, ControlModifier, InverseModifier, PowerModifier
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import UnitaryGate

from nwqlib.blocks import select_block_encoding
from nwqlib.blocks._qiskit_intake import snapshot_circuit
from nwqlib.blocks.encoding import construct_block_encoding
from nwqlib.operators import ingest_dense
from nwqlib.subroutines.block_encoding.core import BlockEncoding


@pytest.mark.parametrize("first_angle", [.9])
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


def test_native_family_law_preserves_asserted_domain():
    from types import MappingProxyType
    selected=select_block_encoding("oracle",family_encoding("dense_dilation",MappingProxyType({})),
        operator=ingest_dense(np.eye(4),max_bytes=1_000_000),max_bytes=1_000_000)
    law,=selected.record.resource_laws
    # Three-qubit QSD: (23*64 - 72*8 + 64)/48 = 20,
    # including the lower-order terms rather than only 23*64/48.
    assert law.value.value==pytest.approx(20.,rel=1e-15, abs=0)
    assert law.evidence.kind=="user_assertion"
    assert law.evidence.source.version == "2"
