"""Independent certificate, lowering, resolution, and resource tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import qiskit
import qiskit.quantum_info as qiskit_info
import scipy.linalg
from qiskit.quantum_info import Operator

from _lchs_suzuki_reference import pauli_matrix, sorted_pauli_lie_1_reference, sorted_pauli_suzuki_2_reference
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS
from nwqlib.algorithms.lchs.native import (
    resolve_lcu_select_implementation,
)
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
import nwqlib.algorithms.lchs.select_synthesis as select_module
from nwqlib.algorithms.lchs.select_synthesis import (
    build_multiplexed_product_formula_select,
    build_structured_product_formula_select,
)
from nwqlib.algorithms.lchs.time_independent_terms import (
    _TrotterPauliDecomposition,
    _build_product_formula_select_plan,
    generate_lchs_product_formula_select_plan,
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


def _nonnormal_problem() -> LinearDynamics:
    return LinearDynamics(
        A=np.array([[0.2, 0.03j], [0.03j, 0.4]], dtype=complex),
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
    data = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=_signed_options(address_qubits).revise(trotter_order=order),
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


def test_auto_cost_comparison_tie_manual_choices_and_strict_less_than() -> None:
    """An equal CX cost must keep the generic choice, while a strict saving enables the certified
    structured route.
    """
    problem = _mixed_problem()
    tie_data = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=_signed_options(2),
    )
    tie = resolve_lcu_select_implementation(
        _signed_options(2),
        he_backend="trotter",
        candidate_costs=product_formula_select_resource_law(tie_data.plan),
        select_plan=tie_data.plan,
    )
    assert tie["lcu_select_candidate_costs"] == {
        "multiplexor_select_basis_cx_count": 14,
        "structured_select_basis_cx_count": 14,
    }
    assert tie["resolved_lcu_select_implementation"] == "multiplexor"

    cheaper_data = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=_signed_options(3),
    )
    resolutions = {
        request: resolve_lcu_select_implementation(
            _signed_options(3, select=request),
            he_backend="trotter",
            candidate_costs=product_formula_select_resource_law(cheaper_data.plan),
            select_plan=cheaper_data.plan,
        )["resolved_lcu_select_implementation"]
        for request in ("auto", "structured", "multiplexor")
    }
    assert resolutions == {
        "auto": "structured",
        "structured": "structured",
        "multiplexor": "multiplexor",
    }


def test_resolver_uses_only_plan_metadata_and_linear_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _mixed_problem()
    data = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=_signed_options(3),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resolver invoked a forbidden expensive operation")

    monkeypatch.setattr(select_module, "QuantumCircuit", forbidden)
    monkeypatch.setattr(qiskit, "transpile", forbidden)
    monkeypatch.setattr(select_module.SparsePauliOp, "to_matrix", forbidden)
    monkeypatch.setattr(qiskit_info, "Statevector", forbidden)
    resolution = resolve_lcu_select_implementation(
        _signed_options(3),
        he_backend="trotter",
        candidate_costs=product_formula_select_resource_law(data.plan),
        select_plan=data.plan,
    )
    assert resolution["resolved_lcu_select_implementation"] == "structured"




@pytest.mark.parametrize(
    ("case", "condition"),
    (
        ("affine", None),
        ("inactive_branch", "fixed-step product formula requires a uniform step count"),
        ("elapsed_times", "structured affine occurrences require a uniform elapsed time"),
        ("invalid_address", "address values are not affine in computational-basis bits"),
        ("address_width", "affine address certificate does not cover the SELECT control basis"),
        (
            "nonaffine_nodes",
            "structure certificate reconstruction does not match generic angle tables",
        ),
        ("nonreal_coefficient", "affine Pauli structure rejected unsupported terms"),
    ),
)
def test_structured_eligibility_is_produced_before_resolution(
    case: str,
    condition: str | None,
) -> None:
    decomposition = _TrotterPauliDecomposition(
        num_qubits=1,
        identity_label="I",
        l_coefficients={"X": 1.0},
        h_coefficients={"X": 0.0},
    )
    nodes = np.array([0.0, 1.0, -2.0, -1.0])
    elapsed_times = np.full(4, 0.1)
    address = _signed_address_structure(2)
    if case == "inactive_branch":
        elapsed_times[0] = 0.0
    elif case == "elapsed_times":
        elapsed_times[-1] = 0.2
    elif case == "invalid_address":
        address["sign_bit"] = 0
    elif case == "address_width":
        address = _signed_address_structure(3)
    elif case == "nonaffine_nodes":
        nodes[1] = 0.75
    elif case == "nonreal_coefficient":
        # A below-pruning-scale imaginary residual is tolerated by the
        # generic product formula, but is outside the real affine domain.
        decomposition = replace(decomposition, l_coefficients={"X": 1.0 + 1.0e-13j})
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
    assert certificate["eligible"] is (condition is None)
    if condition is None:
        resolved = resolve_lcu_select_implementation(
            _signed_options(2, select="structured"),
            he_backend="trotter",
            select_plan=plan,
        )
        assert resolved["resolved_lcu_select_implementation"] == "structured"
        assert certificate["rejection_reasons"] == ()
        return

    assert any(condition in reason for reason in certificate["rejection_reasons"])
    resolution = resolve_lcu_select_implementation(
        _signed_options(2),
        he_backend="trotter",
        select_plan=plan,
    )
    assert resolution["resolved_lcu_select_implementation"] == "multiplexor"
    with pytest.raises(ValueError, match=condition):
        resolve_lcu_select_implementation(
            _signed_options(2, select="structured"),
            he_backend="trotter",
            select_plan=plan,
        )


def test_actual_arbitrary_padding_pruning_and_unequal_step_fallbacks() -> None:
    """Padding and pruning invalidate affine structure, but the generic fallback must still match
    explicit matrix blocks.
    """
    problem = _mixed_problem()
    arbitrary_options = LCHS(
        hamiltonian_evolution_backend="trotter",
        approximation_tolerance=0.5,
    )
    arbitrary = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=arbitrary_options,
    ).plan
    assert arbitrary.structure_certificate is None
    assert (
        resolve_lcu_select_implementation(
            arbitrary_options,
            he_backend="trotter",
            candidate_costs=product_formula_select_resource_law(arbitrary),
            select_plan=arbitrary,
        )["resolved_lcu_select_implementation"]
        == "multiplexor"
    )

    decomposition = _TrotterPauliDecomposition(
        num_qubits=1,
        identity_label="I",
        l_coefficients={"X": 1.0},
        h_coefficients={"X": 0.0},
    )
    padded, _, _ = _build_product_formula_select_plan(
        decomposition=decomposition,
        k_values=np.array([0.0, 1.0, -2.0]),
        elapsed_times=np.array([0.1, 0.1, 0.1]),
        coefficients=np.ones(3, dtype=complex),
        branch_to_node=(0, 1, 2),
        method=_signed_options(2),
        address_structure=_signed_address_structure(2),
    )
    assert (
        "physical nodes must fill the address basis without padding"
        in (padded.structure_certificate["rejection_reasons"])
    )

    non_affine_pruning, _, _ = _build_product_formula_select_plan(
        decomposition=replace(decomposition, h_coefficients={"X": 5.0e-13}),
        k_values=np.array([0.0, 1.0, -2.0, -1.0]),
        elapsed_times=np.full(4, 0.1),
        coefficients=np.ones(4, dtype=complex),
        branch_to_node=(0, 1, 2, 3),
        method=_signed_options(2),
        address_structure=_signed_address_structure(2),
    )
    assert (
        "pruning does not preserve affine occurrence angle tables"
        in (non_affine_pruning.structure_certificate["rejection_reasons"])
    )
    for plan, nodes, h_coefficient in (
        (padded, (0.0, 1.0, -2.0), 0.0),
        (non_affine_pruning, (0.0, 1.0, -2.0, -1.0), 5.0e-13),
    ):
        resolution = resolve_lcu_select_implementation(
            _signed_options(2),
            he_backend="trotter",
            candidate_costs=product_formula_select_resource_law(plan),
            select_plan=plan,
        )
        assert resolution["resolved_lcu_select_implementation"] == "multiplexor"
        realized = Operator(
            build_multiplexed_product_formula_select(plan, num_system_qubits=1)
        ).data
        expected = np.zeros((8, 8), dtype=complex)
        for branch in range(4):
            projector = np.zeros((4, 4))
            projector[branch, branch] = 1.0
            block = (
                scipy.linalg.expm(-0.1j * (nodes[branch] + h_coefficient) * pauli_matrix("X"))
                if branch < len(nodes)
                else np.eye(2)
            )
            expected += np.kron(block, projector)
        # Only the k=0 branch prunes its 5e-13 X coefficient; its induced
        # error is at most 0.1*5e-13, below this matrix-product allowance.
        np.testing.assert_allclose(realized, expected, rtol=0.0, atol=1.0e-12)

    budgeted = _signed_options(3).revise(hamiltonian_evolution_backend='trotter_error_budgeted')
    resolution = resolve_lcu_select_implementation(
        budgeted,
        he_backend="trotter_error_budgeted",
    )
    assert resolution["resolved_lcu_select_implementation"] == "multiplexor"
    assert resolution["lcu_select_structure_rejection_reasons"] == [
        "fixed-step product formula is required; trotter_error_budgeted is ineligible"
    ]
    with pytest.raises(ValueError, match="trotter_error_budgeted is ineligible"):
        resolve_lcu_select_implementation(
            budgeted.revise(lcu_select_implementation='structured'),
            he_backend="trotter_error_budgeted",
        )
