"""The two routes by which a construction controls a dense unitary, and the rule of ``"auto"``.

The gate-wise route synthesizes the unitary and lets Qiskit control every
synthesized gate. The whole-matrix route synthesizes the controlled matrix.
Both must give the controlled matrix to rounding at every call site, with
open control values and with the controls in superposition. ``"auto"``
takes the whole-matrix route for one control and the gate-wise route for
more, and the route is part of the Plan.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector, random_statevector, random_unitary

from nwqlib.subroutines.qiskit_compat import controlled

ROUNDING = 1e-12


def _controlled_matrix(matrix, num_controls, state):
    """The matrix that applies ``matrix`` when the controls, qubits 0 to k - 1, hold ``state``."""
    size = matrix.shape[0]
    result = np.eye(size << num_controls, dtype=complex)
    rows = [state + (index << num_controls) for index in range(size)]
    result[np.ix_(rows, rows)] = matrix
    return result


def _operator(gate):
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits)
    return Operator(circuit).data


def test_auto_takes_the_whole_matrix_route_for_one_control_only():
    from nwqlib.subroutines._dense_synthesis import AUTO_WHOLE_MATRIX_MAX_CONTROLS, select_dense_control_route

    assert AUTO_WHOLE_MATRIX_MAX_CONTROLS == 1
    assert select_dense_control_route("auto", 1) == "whole_matrix"
    assert select_dense_control_route("auto", 2) == "gatewise"
    for controls in (1, 2, 5):
        assert select_dense_control_route("gatewise", controls) == "gatewise"
        assert select_dense_control_route("whole_matrix", controls) == "whole_matrix"
    with pytest.raises(ValueError, match="dense_control_route"):
        select_dense_control_route("whole", 1)
    with pytest.raises(ValueError, match="positive number of controls"):
        select_dense_control_route("auto", 0)


@pytest.mark.parametrize(("num_qubits", "controls"), [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3), (3, 2),
                                                     (2, 3), (4, 1), (5, 1), (1, 5)])
def test_controlled_synthesis_census_bounds_the_built_whole_matrix_circuit(num_qubits, controls):
    # The whole-matrix synthesis demultiplexes the controlled matrix into a
    # multiplexed RZ and two blocks one qubit smaller. On m >= 3 qubits,
    # control included, that bounds the circuit by
    # (25/96) 4**m - 2**m + 4/3 CX and (11/16) 4**m - 2**(m+1) instructions,
    # and the kept-byte law of controlled_synthesis_size allots 256 bytes to
    # each instruction. The census and both counts are upper bounds, so a
    # synthesis that needs fewer gates still passes.
    from collections import Counter

    from test_synthesis_admission import _controlled_census
    from nwqlib.subroutines._dense_synthesis import (controlled_synthesis_gate_census, controlled_synthesis_size,
                                                     controlled_unitary_circuit)

    m = num_qubits + controls
    census = controlled_synthesis_gate_census(m)
    assert census == _controlled_census(m)
    circuit = controlled_unitary_circuit(random_unitary(1 << num_qubits, seed=64 + m).data, controls)
    built = Counter(item.operation.name for item in circuit.data)
    assert set(built) <= set(census) and all(built[kind] <= census[kind] for kind in census)
    if m >= 3:
        assert census["cx"] == (25 * 4**m - 96 * 2**m + 128) // 96
        assert len(circuit.data) <= 11 * 4**m // 16 - 2 ** (m + 1)
    assert controlled_synthesis_size(m)[2] >= 256 * len(circuit.data)


def _dense_gates():
    unitary = random_unitary(4, seed=61).data
    single = QuantumCircuit(2)
    single.append(UnitaryGate(unitary), [0, 1])
    single.global_phase = 0.4
    composite = QuantumCircuit(3)
    composite.ry(0.3, 2)
    composite.cx(2, 0)
    composite.append(UnitaryGate(unitary), [1, 0])
    composite.h(2)
    composite.global_phase = -0.7
    return {"single": single, "composite": composite}


@pytest.mark.parametrize("route", ["gatewise", "whole_matrix", "auto"])
@pytest.mark.parametrize(("controls", "state"), [(1, 0), (1, 1), (2, 1), (2, 3)])
@pytest.mark.parametrize("kind", ["single", "composite"])
def test_both_routes_give_the_controlled_matrix(kind, controls, state, route):
    # A composite with other gates around its dense unitary is controlled
    # instruction by instruction on the whole-matrix route, with its global
    # phase as a phase gate on the controls.
    circuit = _dense_gates()[kind]
    result = controlled(circuit.to_gate(), controls, ctrl_state=state, route=route)
    expected = _controlled_matrix(Operator(circuit).data, controls, state)
    assert np.max(np.abs(_operator(result) - expected)) < ROUNDING
    whole = route == "whole_matrix" or (route == "auto" and controls == 1)
    names = {item.operation.name for item in result.definition.data} | {result.name}
    assert any(name.startswith("c_dense_unitary") for name in names) == whole
    # The controls in superposition, the system in a random state.
    system = random_statevector(1 << circuit.num_qubits, seed=62)
    prepared = QuantumCircuit(controls + circuit.num_qubits)
    prepared.h(range(controls))
    prepared.append(result, prepared.qubits)
    final = system.tensor(Statevector.from_label("0" * controls)).evolve(prepared)
    plus = Statevector.from_label("+" * controls)
    expected_state = system.tensor(plus).evolve(Operator(expected))
    assert np.max(np.abs(final.data - expected_state.data)) < ROUNDING


def test_the_whole_matrix_route_keeps_a_one_qubit_unitary_and_a_gate_without_dense_unitaries():
    one = QuantumCircuit(1)
    one.append(UnitaryGate(random_unitary(2, seed=63).data), [0])
    plain = QuantumCircuit(2)
    plain.ry(0.2, 0)
    plain.cx(0, 1)
    for circuit in (one, plain):
        whole = controlled(circuit.to_gate(), 1, route="whole_matrix")
        gatewise = controlled(circuit.to_gate(), 1)
        assert type(whole) is type(gatewise) and whole.name == gatewise.name
        assert np.max(np.abs(_operator(whole) - _controlled_matrix(Operator(circuit).data, 1, 1))) < ROUNDING


def _lchs_dense_plan(num_qubits, address_bits, route, source=False):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_lchs_resource_structural_law import _dense_select_problem
    from nwqlib import plan
    from nwqlib.algorithms.lchs import LCHS, ProviderConfig

    grid = ProviderConfig(implementation="signed_binary_uniform",
                          parameters={"num_qubits": address_bits, "lsb_position": -1})
    problem = _dense_select_problem(num_qubits, source=source)
    method = LCHS(hamiltonian_evolution_backend="dense_exact", k_quadrature=grid, dense_control_route=route,
                  **({"duhamel_nodes": 1} if source else {}))
    return problem, plan(problem, method=method)


def _built_select(problem, selected):
    from nwqlib.algorithms.lchs.native import construct_select

    return construct_select(SimpleNamespace(_payload=(selected._native["native_data"], problem.elapsed_time)), (),
                            None)


@pytest.mark.parametrize("route", ["gatewise", "whole_matrix", "auto"])
@pytest.mark.parametrize("address_bits", [1, 2])
def test_lchs_dense_select_applies_each_branch_under_its_address_on_both_routes(address_bits, route):
    # The SELECT applies exp(i arg c_j) U_j when the address holds j, which
    # includes open values of the address controls. The independent reference
    # branch is scipy.linalg.expm of -i T (k_j L + H).
    from scipy.linalg import expm

    problem, selected = _lchs_dense_plan(2, address_bits, route)
    data = selected._native["native_data"]
    built = _built_select(problem, selected)
    quad = data.quadrature
    expected = np.eye(1 << built.num_qubits, dtype=complex)
    for index, (coefficient, k_value) in enumerate(zip(data.coefficient_plan.coefficients, quad.k_nodes,
                                                       strict=True)):
        unitary = expm(-1j * problem.elapsed_time * (float(k_value) * quad.l_part + quad.h_part))
        block = _controlled_matrix(np.exp(1j * np.angle(coefficient)) * unitary, address_bits, index)
        expected = block @ expected
    assert np.max(np.abs(Operator(built).data - expected)) < ROUNDING
    whole = route == "whole_matrix" or (route == "auto" and address_bits == 1)
    names = {item.operation.name for item in built.data}
    assert any(name.startswith("c_dense_unitary") for name in names) == whole


def test_lchs_source_select_is_the_same_operator_on_both_routes():
    # On the whole-matrix route a constant-source branch is one unitary, its
    # evolution times the matrix of its input preparation.
    operators = {}
    for route in ("gatewise", "whole_matrix"):
        problem, selected = _lchs_dense_plan(2, 1, route, source=True)
        built = _built_select(problem, selected)
        operators[route] = Operator(built).data
        names = {item.operation.name for item in built.data}
        assert any(name.startswith("c_dense_unitary") for name in names) == (route == "whole_matrix")
    assert np.max(np.abs(operators["gatewise"] - operators["whole_matrix"])) < ROUNDING


def _qls_query(solver, route, controls):
    import nwqlib
    from nwqlib import LinearSystem, StateVector
    from nwqlib.algorithms.qls import QLS

    matrix = np.array([[1.0, 0.2, 0.0, 0.1], [0.0, 0.8, 0.1, 0.0], [0.1, 0.0, 0.9, 0.2], [0.0, 0.1, 0.0, 0.7]])
    selected = nwqlib.plan(LinearSystem(A=matrix, b=np.ones(4)),
                           method=QLS(solver=solver, encoded_solution_norm_estimate=1.5, epsilon_inv=0.1,
                                      block_encoding_implementation="dense_dilation", dense_control_route=route),
                           output=StateVector(normalization="unit", global_phase="modulo_global_phase"),
                           execution="quantum", seed=3)
    block = next(item for item in selected.blocks if item.record.signature.name == f"original_A_query_c{controls}")
    return selected, block


@pytest.mark.parametrize(("solver", "controls"), [("shortcut_native_svp", 1), ("shortcut_dilation", 2)])
@pytest.mark.parametrize("route", ["gatewise", "whole_matrix", "auto"])
def test_qls_controlled_query_applies_the_dilation_under_its_control_value_on_both_routes(solver, controls, route):
    from nwqlib.algorithms.qls.quantum import _native_encoding, _query_circuit

    selected, block = _qls_query(solver, route, controls)
    assert block._payload.route == route
    dilation = Operator(_native_encoding(block._payload.base).circuit).data
    context = {}
    for adjoint in (0, 1):
        for state in range(1 << controls):
            built = _query_circuit(block, {"adjoint": adjoint, "control_state": state}, lambda factory: context)
            target = dilation.conj().T if adjoint else dilation
            expected = _controlled_matrix(target, controls, state)
            assert np.max(np.abs(Operator(built).data - expected)) < ROUNDING
            whole = route == "whole_matrix" or (route == "auto" and controls == 1)
            assert built.data[0].operation.name.startswith("c_dense_unitary") == whole


def test_qsp_joint_generator_is_the_same_operator_on_both_routes():
    # The combine qubit controls each branch, the diagonal rotation and the
    # dense child. On the whole-matrix route the child is synthesized with
    # the control and the rotation is controlled gate-wise.
    from nwqlib.subroutines.block_encoding import build_block_encoding
    from nwqlib.subroutines.qsp import build_control_diagonal_generator_encoding

    encoding_l = build_block_encoding(np.array([[0.5, 0.1], [0.1, 0.3]], dtype=complex),
                                      implementation="dense_dilation")
    encoding_h = build_block_encoding(np.array([[0.2, 0.05j], [-0.05j, -0.1]], dtype=complex),
                                      implementation="dense_dilation")
    options = dict(l_diagonal=[1.0, 0.5], h_diagonal=[1.0, 1.0])
    reference = build_control_diagonal_generator_encoding(encoding_l, encoding_h, dense_control_route="gatewise",
                                                          **options)
    joint = build_control_diagonal_generator_encoding(encoding_l, encoding_h, dense_control_route="whole_matrix",
                                                      **options)
    assert np.max(np.abs(Operator(joint.circuit).data - Operator(reference.circuit).data)) < ROUNDING
    names = {item.operation.name for item in joint.circuit.data}
    assert any(name.startswith("c_generator_branch") for name in names)
    assert not any(item.operation.name.startswith("c_generator_branch") for item in reference.circuit.data)


def test_the_route_is_part_of_the_plan():
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.algorithms.qls import QLS

    assert LCHS().dense_control_route == "auto" and QLS().dense_control_route == "auto"
    assert LCHS(dense_control_route="gatewise").content_id != LCHS().content_id
    assert QLS(dense_control_route="whole_matrix").content_id != QLS().content_id
    identities = {_lchs_dense_plan(2, 1, route)[1].content_id for route in ("gatewise", "whole_matrix", "auto")}
    assert len(identities) == 3
    with pytest.raises(ValueError):
        LCHS(dense_control_route="whole")


@pytest.mark.parametrize(("num_qubits", "branches"), [(3, 128), (4, 32), (5, 8), (6, 4)])
def test_auto_admits_the_stated_lchs_dense_boundaries(num_qubits, branches):
    # The largest power-of-two branch counts that the default max_select_work
    # admits for these test problems, from the spectral branch law
    # (time_independent_terms._spectral_branch_requirements) and the
    # synthesis and control laws.
    problem, selected = _lchs_dense_plan(num_qubits, branches.bit_length() - 1, "auto")
    assert selected.reconstruction.physical_branches == branches
    with pytest.raises(ValueError, match="max_select_work"):
        _lchs_dense_plan(num_qubits, branches.bit_length(), "auto")


@pytest.mark.parametrize("solver", ["qsvt_inverse", "shortcut_native_svp", "shortcut_dilation"])
def test_auto_admits_controlled_dense_queries_of_64_padded_coordinates(solver):
    # qsvt_inverse (the Hermitian dilation of a non-Hermitian A) and
    # shortcut_native_svp query with one control, which "auto" takes through
    # the controlled matrix, and shortcut_dilation with two, gate-wise.
    import nwqlib
    from nwqlib import LinearSystem, StateVector
    from nwqlib.algorithms.qls import QLS

    rng = np.random.default_rng(5)
    for size, admitted in ((64, True), (128, False)):
        matrix = np.eye(size) + 0.05 * rng.normal(size=(size, size))
        extra = {} if solver == "qsvt_inverse" else {"encoded_solution_norm_estimate": 1.0}
        method = QLS(solver=solver, block_encoding_implementation="dense_dilation", epsilon_inv=0.1, **extra)
        output = StateVector(normalization="unit", global_phase="modulo_global_phase")
        if admitted:
            nwqlib.plan(LinearSystem(A=matrix, b=np.ones(size)), method=method, output=output, execution="quantum",
                        seed=3)
        else:
            with pytest.raises(ValueError, match="QLS controlled dense-query synthesis exceeds max_work"):
                nwqlib.plan(LinearSystem(A=matrix, b=np.ones(size)), method=method, output=output,
                            execution="quantum", seed=3)
