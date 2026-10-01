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


def test_same_name_operations_are_summed_whatever_their_definition():
    circuit = QuantumCircuit(3, 1)
    circuit.cx(0, 1)
    circuit.append(opaque("cx", 2), [1, 2])
    circuit.append(opaque("measure", 2), [0, 2])
    circuit.measure(0, 0)
    circuit.append(opaque("barrier", 3), [0, 1, 2])
    circuit.append(opaque("my_block", 3), [0, 1, 2])
    result = inventory(circuit)
    assert result["operations"] == {"cx": 2, "measure": 2, "barrier": 1, "my_block": 1}
    assert result["total_operations"] == 6


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


def test_requested_transpile_inventories_one_auxiliary_copy(monkeypatch):
    import qiskit

    circuit = small_circuit()
    original = tuple(circuit.data)
    calls = []
    transpile = qiskit.transpile

    def counted(selected, **options):
        calls.append(options)
        assert selected is not circuit and not any(
            isinstance(i.operation, SaveStatevector) for i in selected.data
        )
        return transpile(selected, **options)

    monkeypatch.setattr(qiskit, "transpile", counted)
    result = inventory(
        circuit,
        transpile_options={
            "basis_gates": ["u", "cx"],
            "optimization_level": 0,
            "seed_transpiler": 7,
        },
    )
    assert len(calls) == 1
    # H becomes one U at level 0; the save is removed before compilation.
    assert result["operations"] == {"u": 1, "cx": 1, "reset": 1, "barrier": 1, "measure": 2}
    assert result["total_operations"] == 6
    assert result["basis"] == "transpiled:u,cx"
    assert result["circuit"].startswith("auxiliary transpiled copy of hand-listed circuit")
    compiler = result["compiler"]
    assert compiler["name"] == "qiskit.transpile" and compiler["version"] == qiskit.__version__
    assert compiler["options"]["seed_transpiler"] == 7
    assert compiler["options"]["optimization_level"] == 0
    assert compiler["options"]["approximation_degree"] == 1.0
    assert tuple(circuit.data) == original


def test_compiler_options_admit_before_work_and_record_resolved_defaults(monkeypatch):
    import qiskit
    from qiskit import user_config

    circuit = QuantumCircuit(1)
    circuit.h(0)
    calls = []
    original = qiskit.transpile

    def counted(selected, **options):
        calls.append(options)
        return original(selected, **options)

    monkeypatch.setattr(qiskit, "transpile", counted)
    with pytest.raises(ValueError, match="unsupported"):
        inventory(circuit, transpile_options={"fictional_compiler_option": True})
    cycle = {}
    cycle["recursive"] = cycle
    with pytest.raises(ValueError, match="acyclic"):
        inventory(circuit, transpile_options=cycle)
    assert calls == []
    monkeypatch.setattr(
        user_config,
        "get_config",
        lambda: {"transpile_optimization_level": 0, "transpiler_seed": 19},
    )
    monkeypatch.setenv("QISKIT_TRANSPILER_SEED", "47")
    result = inventory(circuit, transpile_options={"basis_gates": ["u", "cx"]})
    assert calls[0]["optimization_level"] == 0 and calls[0]["seed_transpiler"] == 47
    assert result["compiler"]["options"]["seed_transpiler"] == 47


def test_compiled_output_and_inventory_limits_follow_the_input_checks():
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    # One input operation; its u/cx synthesis at level 0 has more than two.
    with pytest.raises(ValueError, match="compiler output exceeds max_operations"):
        inventory(
            circuit,
            max_operations=2,
            transpile_options={"basis_gates": ["u", "cx"], "optimization_level": 0, "seed_transpiler": 7},
        )
    named = QuantumCircuit(1)
    named.append(Gate("g" * 200, 1, []), [0])
    # Input, option and basis envelopes fit 1000 bytes, while the 200-character
    # name needs 64 + 6 * 200 + 256 bytes in the returned inventory.
    with pytest.raises(ValueError, match="native operation inventory"):
        inventory(named, max_bytes=1000)
    assert inventory(named, max_bytes=2000)["operations"] == {"g" * 200: 1}


def test_option_charge_bounds_the_options_utf8_json_length():
    """The transpile-option charge is at least the options' UTF-8 JSON length.

    Control characters reach the six-byte escape per code point exactly, so a
    per-character charge below six would admit them at a limit smaller than
    their JSON length. A code point outside the Basic Multilingual Plane
    takes four UTF-8 bytes.
    """
    import json
    from nwqlib.backends.inspection import _options_snapshot

    def charge(value):
        """Smallest max_bytes that admits the snapshot of ``value``."""
        low, high = 1, 100_000
        while low < high:
            middle = (low + high) // 2
            try:
                _options_snapshot(value, max_bytes=middle)
            except ValueError:
                low = middle + 1
            else:
                high = middle
        return low

    def utf8_json(value):
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    control = "\x01" * 40
    assert charge(control) == utf8_json(control) == 6 * 40 + 2
    astral = "\U0001F600" * 10
    assert charge(astral) >= utf8_json(astral)
    options = {
        "seed": -(2**70),
        "note": "\U0001F600é\"\\\n\x1f",
        "values": [-1.2345678901234567e-308, None, True, {"nested": []}],
    }
    assert charge(options) >= utf8_json(options)
