"""Operation inventories and native two-qubit depth of existing native circuits; no acquisition or simulation."""

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


def test_native_two_qubit_depth_keeps_wire_order_and_qualifies_only_target_operations():
    """Exact path weights from the definition: two-qubit native gates weigh 1, every
    operation orders its qubits and classical bits, and an operation without a native
    meaning on the target leaves the depth unknown.
    """
    from qiskit.circuit import Measure
    from qiskit.circuit.library import CXGate, ECRGate
    from qiskit.transpiler import Target
    from nwqlib.backends.inspection import inspect_native_depth

    target = Target(num_qubits=4)
    target.add_instruction(CXGate(), {(0, 1): None, (2, 3): None})
    # ECR has a Qiskit definition and is still a native gate of this target.
    target.add_instruction(ECRGate(), {(1, 2): None})
    target.add_instruction(Measure(), {(qubit,): None for qubit in range(4)})

    def depths(circuit):
        result = inspect_native_depth(circuit, target)
        assert result.width == 4
        if not result.qualified:
            assert result.reason and result.native_two_qubit_depth is None
            return None
        return result.native_two_qubit_depth, result.total_native_depth, result.native_two_qubit_gates

    joined = QuantumCircuit(4)
    joined.cx(0, 1)
    joined.barrier(1, 2)
    joined.cx(2, 3)
    parallel = QuantumCircuit(4)
    parallel.cx(0, 1)
    parallel.cx(2, 3)
    classical = QuantumCircuit(4, 1)
    classical.cx(0, 1)
    classical.measure(1, 0)
    classical.measure(2, 0)
    classical.cx(2, 3)
    defined = QuantumCircuit(4)
    defined.ecr(1, 2)
    # (two-qubit depth, total depth, two-qubit gates). The barrier joins the two
    # CX paths, and without it they are parallel. The shared classical bit orders
    # the two measurements, so CX(0,1), both measurements and CX(2,3) form one
    # path of total depth 4.
    assert depths(joined) == (2, 2, 2)
    assert depths(parallel) == (1, 1, 2)
    assert depths(classical) == (2, 4, 2)
    assert depths(defined) == (1, 1, 1)
    assert depths(QuantumCircuit(4)) == (0, 0, 0)

    composite = QuantumCircuit(4)
    composite.append(opaque("pair_block", 2), [0, 1])
    impostor = QuantumCircuit(4)
    impostor.append(opaque("cx", 2), [0, 1])
    unrouted = QuantumCircuit(4)
    unrouted.cx(0, 2)
    branch = QuantumCircuit(4, 1)
    branch.measure(0, 0)
    with branch.if_test((branch.clbits[0], 1)):
        branch.cx(0, 1)
    # A two-qubit width, a native name or a native body inside control flow does
    # not make an operation native on these qubits.
    for circuit in (composite, impostor, unrouted, branch):
        assert depths(circuit) is None
