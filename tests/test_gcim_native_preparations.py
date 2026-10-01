"""Native reference composition relations on tiny cases."""

from types import SimpleNamespace
import numpy as np
import pytest
from qiskit import QuantumCircuit
from nwqlib._limits import DEFAULT_MAX_DIRECT_AMPLITUDES
from nwqlib.algorithms.gcim import adapt_acquisition
from nwqlib.problems import bind_preparation_circuit, ingest_occupation
from nwqlib.core import InputRef, Source

MAX_BYTES = 2000000


def reference_kwargs(circuit):
    basis = ingest_occupation(
        (0,) * circuit.num_qubits, num_qubits=circuit.num_qubits, max_bytes=MAX_BYTES
    ).manifest.basis
    reference = InputRef(
        identity="explicit:test-preparation",
        representation="circuit",
        source=Source(
            name="test preparation",
            version="1",
            domain="explicit tiny unitary circuit",
            reference="test_gcim_native_preparations",
        ),
    )
    return dict(
        inputs=SimpleNamespace(pool=()),
        reference=bind_preparation_circuit(
            circuit, reference=reference, basis=basis, max_bytes=MAX_BYTES
        ),
        compiler_plans=(),
        policy=SimpleNamespace(max_bytes=MAX_BYTES),
        context=adapt_acquisition.AdaptContext(),
        max_direct_amplitudes=DEFAULT_MAX_DIRECT_AMPLITUDES,
    )


def test_preparation_inverse_reuses_controlled_definition(monkeypatch):
    from qiskit.circuit import Gate
    from qiskit.quantum_info import Operator

    preparation = QuantumCircuit(1, name="reference")
    preparation.ry(0.31, 0)
    preparation.global_phase = 0.17
    kwargs = reference_kwargs(preparation)
    forward = adapt_acquisition._native_preparation((), controlled=True, inverse=False, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("inverse constructed a second controlled reference")

    monkeypatch.setattr(Gate, "control", forbidden)
    inverse = adapt_acquisition._native_preparation((), controlled=True, inverse=True, **kwargs)
    np.testing.assert_allclose(
        Operator(inverse).data, Operator(forward).data.conj().T, atol=2e-14, rtol=0.0
    )


@pytest.mark.parametrize("kind", ("bare", "native_x", "named_x", "named_u", "named_barrier"))
def test_reference_admission_and_control_follow_gate_semantics(kind):
    from qiskit.circuit import Gate
    from qiskit.quantum_info import Operator

    reference = QuantumCircuit(1, global_phase=0.17)
    if kind == "native_x":
        reference.x(0)
        expected = np.array([[0, 1], [1, 0]], complex)
    elif kind.startswith("named_"):
        custom = Gate(kind.removeprefix("named_"), 1, [])
        custom.definition = QuantumCircuit(1, global_phase=0.13)
        custom.definition.h(0)
        reference.append(custom, [0])
        expected = np.exp(0.13j) * np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    else:
        expected = np.eye(2, dtype=complex)
    reference.barrier()
    kwargs = reference_kwargs(reference)
    expected *= np.exp(0.17j)
    prepared = adapt_acquisition._native_preparation((), controlled=False, inverse=False, **kwargs)
    controlled = adapt_acquisition._native_preparation((), controlled=True, inverse=False, **kwargs)
    np.testing.assert_allclose(Operator(prepared).data, expected, atol=1e-13, rtol=1e-13)
    expected_control = np.eye(4, dtype=complex)
    expected_control[np.ix_([1, 3], [1, 3])] = expected
    np.testing.assert_allclose(Operator(controlled).data, expected_control, atol=1e-13, rtol=1e-13)


def test_custom_native_snapshot_is_stable_for_archive_reopening():
    from qiskit.circuit import Gate
    from qiskit.quantum_info import Operator
    from nwqlib.blocks._qiskit_intake import snapshot_circuit

    custom = Gate("x", 1, [])
    custom.definition = QuantumCircuit(1, global_phase=0.23)
    custom.definition.h(0)
    nested = QuantumCircuit(1)
    nested.append(custom, [0])
    outer = Gate("u", 1, [])
    outer.definition = nested
    circuit = QuantumCircuit(1)
    circuit.append(outer, [0])
    first, _ = snapshot_circuit(circuit)
    second, _ = snapshot_circuit(first)
    assert first == second
    np.testing.assert_allclose(
        Operator(second).data,
        np.exp(0.23j) * np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        atol=1e-14,
        rtol=0,
    )
