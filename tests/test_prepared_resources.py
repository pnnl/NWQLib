"""Operation inventories of existing native circuits; no acquisition or simulation."""

import copy

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Clifford
from qiskit_aer.library import SaveStatevector
from nwqlib.backends.inspection import inspect_circuit_resources

BASIS = ("h", "cx", "measure", "reset", "save_statevector")


def inventory(circuit, **options):
    return inspect_circuit_resources(
        circuit, native_basis=BASIS, label="hand-listed circuit", **options
    )


def opaque(name, width):
    """A user-defined gate whose definition must not be counted."""
    definition = QuantumCircuit(width)
    definition.h(0)
    definition.cx(0, width - 1)
    gate = Gate(name, width, [])
    gate.definition = definition
    return gate


def small_circuit():
    """H, CX, reset, barrier, statevector save and two measurements."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.barrier()
    circuit.append(SaveStatevector(2), [0, 1])
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_ordinary_gates_directives_and_saves_are_listed_by_name():
    result = inventory(small_circuit())
    assert result["operations"] == {
        "h": 1,
        "cx": 1,
        "reset": 1,
        "barrier": 1,
        "save_statevector": 1,
        "measure": 2,
    }
    assert result["total_operations"] == 7
    assert (result["num_qubits"], result["num_clbits"]) == (2, 2)
    # Barrier and save are directives. Qubit 0 carries H, CX, reset, measure.
    assert result["depth"] == 4
    assert result["circuit"] == "hand-listed circuit; not executed by inspection"
    assert result["basis"] == "native:" + ",".join(BASIS)
    assert result["compiler"] is None


def test_mixed_circuit_lists_composites_and_control_flow_at_top_level():
    body = QuantumCircuit(2, name="controlled_body")
    body.cx(0, 1)
    controlled = body.to_gate().control(1)
    block = opaque("three_qubit_block", 3)
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    circuit = QuantumCircuit(9, 1)
    appended = []
    for operation, qubits in (
        (controlled, [0, 1, 2]),
        (block, [3, 4, 5]),
        (controlled, [6, 7, 8]),
        (Clifford(bell), [0, 8]),
    ):
        circuit.append(operation, qubits)
        appended.append(operation.name)
    circuit.ccx(6, 7, 8)
    circuit.cx(3, 4)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
        circuit.cx(1, 2)
    appended += ["ccx", "cx", "measure", "if_else"]
    expected = {}
    for name in appended:
        expected[name] = expected.get(name, 0) + 1
    result = inventory(circuit)
    assert result["operations"] == expected
    assert result["total_operations"] == len(appended) == 8
    # Neither definitions nor the if_else body contribute counts.
    assert result["operations"]["cx"] == 1 and "x" not in result["operations"]


def test_raw_inventory_neither_copies_nor_compiles_nor_runs(monkeypatch):
    import qiskit
    from qiskit_aer import AerSimulator

    circuit = small_circuit()
    original = tuple(circuit.data)

    def forbidden(*args, **kwargs):
        pytest.fail("raw inventory copied, expanded, compiled or ran the circuit")

    for owner, name in (
        (QuantumCircuit, "copy"),
        (copy, "deepcopy"),
        (QuantumCircuit, "decompose"),
        (qiskit, "transpile"),
        (AerSimulator, "run"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    assert inventory(circuit)["total_operations"] == 7
    assert tuple(circuit.data) == original
    with pytest.raises(ValueError, match="max_operations"):
        inventory(circuit, max_operations=6)
    assert inventory(circuit, max_operations=7)["total_operations"] == 7
    with pytest.raises(ValueError, match="bytes"):
        inventory(circuit, max_bytes=1)
    with pytest.raises(ValueError, match="labels"):
        inspect_circuit_resources(circuit, native_basis=BASIS, label="")
    with pytest.raises(ValueError, match="native circuit"):
        inspect_circuit_resources(object(), native_basis=BASIS, label="x")
