"""Stored CX counts of multi-controlled X gates against the installed Qiskit."""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XGate

from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS
from nwqlib.subroutines.qiskit_compat import controlled


def test_multi_controlled_x_cx_table_matches_installed_qiskit_synthesis():
    # The table records the CX count of the logical circuit that NWQLib's
    # lowering emits, so the expected count comes from the installed SDK's
    # own synthesis of that gate. A failure after a Qiskit upgrade means the
    # table must be regenerated.
    for k, expected in enumerate(MCX_CX_BY_CONTROLS, start=1):
        circuit = QuantumCircuit(k + 1)
        circuit.append(controlled(XGate(), k, ctrl_state=0), range(k + 1))
        lowered = transpile(circuit, basis_gates=["cx", "u"], optimization_level=0)
        assert lowered.count_ops().get("cx", 0) == expected, k
