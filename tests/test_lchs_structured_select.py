"""Independent certificate, lowering, resolution, and resource tests."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from _lchs_suzuki_reference import sorted_pauli_lie_1_reference, sorted_pauli_suzuki_2_reference
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS
from nwqlib.algorithms.lchs.native import (
    resolve_lcu_select_implementation,
)
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs.select_synthesis import (
    build_multiplexed_product_formula_select,
    build_structured_product_formula_select,
)
from nwqlib.algorithms.lchs.time_independent_terms import (
    _TrotterPauliDecomposition,
    _build_product_formula_select_plan,
    generate_lchs_product_formula_select_plan,
    generate_lchs_quadrature,
)
from nwqlib.subroutines._multiplexors import (
    affine_angle_table_values,
    product_formula_select_resource_law,
)


def _signed_options(
    address_qubits: int,
    *,
    select: str = "auto",
    preparation: str = "direct",
    resource_tier: str = "measured",
) -> LCHS:
    return LCHS(
        lchs_kernel=ProviderConfig(implementation="cauchy_density"),
        k_quadrature=ProviderConfig(
            implementation="signed_binary_uniform",
            parameters={"num_qubits": address_qubits, "lsb_position": -2},
        ),
        hamiltonian_evolution_backend="trotter",
        trotter_steps=1,
        lcu_select_implementation=select,
        lcu_state_preparation=preparation,
    )


def _mixed_problem() -> LinearDynamics:
    return LinearDynamics(
        A=np.array([[0.2, -0.1], [-0.1, 0.3]], dtype=complex),
        initial_state=np.array([1.0, 0.25], dtype=complex),
        time=0.1,
    )


def _heat_problem() -> LinearDynamics:
    # Independent finite Dirichlet second difference; no PDE toy factory.
    matrix=.1*(2*np.eye(4)-np.eye(4,k=1)-np.eye(4,k=-1))
    return LinearDynamics(A=matrix,initial_state=[1,0,0,0],time=.1)


def _signed_address_structure(address_qubits: int) -> dict[str, object]:
    return {
        "affine": True,
        "kind": "twos_complement_affine",
        "num_qubits": address_qubits,
        "sign_bit": address_qubits - 1,
        "sign_coefficient": float(-(1 << (address_qubits - 1))),
        "unsigned_terms": tuple(
            {"qubit": qubit, "coefficient": float(1 << qubit)}
            for qubit in range(address_qubits - 1)
        ),
    }


@pytest.mark.parametrize(
    ("problem_factory", "address_qubits", "order"),
    ((_mixed_problem, 2, 1), (_mixed_problem, 2, 2), (_mixed_problem, 4, 2), (_heat_problem, 3, 2)),
)
def test_structured_and_multiplexed_select_match_independent_dense_blocks(
    problem_factory: object,
    address_qubits: int,
    order: int,
) -> None:
    problem = problem_factory()
    options = _signed_options(address_qubits).revise(trotter_order=order)
    data = generate_lchs_product_formula_select_plan(
        quadrature=generate_lchs_quadrature(
            matrix=problem.A.dense_array(), final_time=problem.elapsed_time, method=options),
        final_time=problem.elapsed_time,
        method=options,
    )
    plan = data.plan
    assert plan.structure_certificate["eligible"] is True
    structured_circuit = build_structured_product_formula_select(
        plan, num_system_qubits=int(np.log2(problem.dimension)),
    )
    structured = np.asarray(Operator(structured_circuit).data)
    law = product_formula_select_resource_law(plan, implementation="structured")
    assert structured_circuit.decompose(reps=10).count_ops().get("cx", 0) == law["select_basis_cx_count"]
    multiplexed = np.asarray(
        Operator(
            build_multiplexed_product_formula_select(
                plan,
                num_system_qubits=int(np.log2(problem.dimension)),
            )
        ).data
    )
    system_dimension = problem.dimension
    machine_tolerance = 64.0 * structured.shape[0] * np.finfo(float).eps
    matrix = problem.A.dense_array()
    l_part, h_part = (matrix + matrix.conj().T) / 2, (matrix - matrix.conj().T) / (2j)
    reference = sorted_pauli_lie_1_reference if order == 1 else sorted_pauli_suzuki_2_reference
    for branch in range(plan.padded_node_count):
        indices = [branch + (basis << address_qubits) for basis in range(system_dimension)]
        # The signed address uses lsb=2^-2; the Cauchy coefficients are real
        # positive. Reconstruct from the original A and mathematical product,
        # independently of the selected occurrence schedule and angle tables.
        signed = branch if branch < 1 << (address_qubits - 1) else branch - (1 << address_qubits)
        expected = reference(l_part, h_part, k_value=signed / 4,
            final_time=problem.elapsed_time, reps=1)
        assert np.max(np.abs(structured[np.ix_(indices, indices)] - expected)) <= (
            machine_tolerance
        )
        assert np.max(np.abs(multiplexed[np.ix_(indices, indices)] - expected)) <= (
            machine_tolerance
        )
    assert np.max(np.abs(structured - multiplexed)) <= machine_tolerance
    for affine, generic in zip(
        plan.structured_generator_payload["occurrence_affine_tables"],
        plan.occurrence_angle_tables.array,
        strict=True,
    ):
        reconstructed = affine_angle_table_values(
            offset=affine.offset,
            coefficients=affine.coefficients,
        )
        assert np.max(np.abs(np.asarray(reconstructed) - np.asarray(generic))) <= (affine.tolerance)


def test_structured_eligibility_is_produced_before_resolution() -> None:
    condition = "structure certificate reconstruction does not match generic angle tables"
    decomposition = _TrotterPauliDecomposition(
        num_qubits=1,
        identity_label="I",
        l_coefficients={"X": 1.0},
        h_coefficients={"X": 0.0},
    )
    nodes = np.array([0.0, 0.75, -2.0, -1.0])
    elapsed_times = np.full(4, 0.1)
    address = _signed_address_structure(2)
    plan, _, _ = _build_product_formula_select_plan(
        decomposition=decomposition,
        k_values=nodes,
        elapsed_times=elapsed_times,
        coefficients=np.ones(4, dtype=complex),
        branch_to_node=(0, 1, 2, 3),
        method=_signed_options(2),
        address_structure=address,
    )
    certificate = plan.structure_certificate
    assert certificate["eligible"] is False
    assert any(condition in reason for reason in certificate["rejection_reasons"])
    resolution = resolve_lcu_select_implementation(
        _signed_options(2),
        select_plan=plan,
    )
    assert resolution == "multiplexor"
    with pytest.raises(ValueError, match=condition):
        resolve_lcu_select_implementation(
            _signed_options(2, select="structured"),
            select_plan=plan,
        )
