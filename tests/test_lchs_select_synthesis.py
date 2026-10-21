"""Independent equivalence gates for homogeneous LCHS SELECT synthesis."""

from __future__ import annotations


import numpy as np
import pytest
import scipy.linalg
from qiskit.quantum_info import Operator

from conftest import lchs_quadrature_options
from _lchs_suzuki_reference import pauli_matrix, sorted_pauli_suzuki_2_reference
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs import LCHS
from nwqlib.algorithms.lchs.select_synthesis import (
    build_multiplexed_product_formula_select,
)
from nwqlib.algorithms.lchs.time_independent_terms import (
    generate_lchs_product_formula_select_plan,
    generate_lchs_quadrature,
)


def test_budgeted_plan_zeroes_inactive_repetitions() -> None:
    """Unequal branch step counts require zero angles only after that branch has finished its
    active repetitions.
    """
    rng = np.random.default_rng(17)
    basis, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1.0j * rng.standard_normal((4, 4)))
    l_part = basis @ np.diag([0.05, 0.11, 0.19, 0.29]) @ basis.conj().T
    h_base = rng.standard_normal((4, 4)) + 1.0j * rng.standard_normal((4, 4))
    h_part = 0.1 * (h_base + h_base.conj().T) / 2.0
    h_part *= 10.0
    options = LCHS(
        lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 0.8}),
        approximation_tolerance=0.2,
        k_quadrature=ProviderConfig(
            implementation="composite_gauss", parameters={"truncation_multiplier": 1.0}
        ),
        hamiltonian_evolution_backend="trotter_error_budgeted",
    )
    data = generate_lchs_product_formula_select_plan(
        quadrature=generate_lchs_quadrature(matrix=l_part + 1.0j * h_part, final_time=2.0, method=options),
        final_time=2.0,
        method=options,
    )
    plan = data.plan
    assert len(set(plan.branch_step_counts)) > 1
    schedule_size = len(plan.occurrence_schedule)
    repetition = 0
    for block, multiplicity in enumerate(plan.occurrence_block_repetitions):
        tables = plan.occurrence_angle_tables.array[
            block * schedule_size : (block + 1) * schedule_size
        ]
        for branch, steps in enumerate(plan.branch_step_counts):
            if repetition >= steps:
                assert all(table[branch] == 0.0 for table in tables)
            else:
                assert repetition + multiplicity <= steps
        repetition += multiplicity
    assert repetition == max(plan.branch_step_counts)


def test_generated_angle_tables_match_independent_suzuki_products() -> None:
    """Compare each address block with an independently ordered Suzuki product including its
    coefficient phase.
    """
    matrix = np.array([[0.4, 0.05], [0.05, 0.5]], dtype=complex)
    final_time = 0.8
    options = LCHS(
        **{("approximation_tolerance" if key=="epsilon" else key):value for key,value in lchs_quadrature_options().items()},
        hamiltonian_evolution_backend="trotter",
        trotter_steps=2,
    )
    quadrature = generate_lchs_quadrature(
        matrix=matrix,
        final_time=final_time,
        method=options,
    )
    data = generate_lchs_product_formula_select_plan(
        quadrature=quadrature,
        final_time=final_time,
        method=options,
    )
    realized = np.asarray(
        Operator(
            build_multiplexed_product_formula_select(
                data.plan,
                num_system_qubits=1,
            )
        ).data
    )
    address_qubits = data.plan.padded_node_count.bit_length() - 1
    for branch, (k_value, coefficient) in enumerate(
        zip(quadrature.k_nodes, data.coefficients, strict=True)
    ):
        indices = [branch, branch + (1 << address_qubits)]
        block = realized[np.ix_(indices, indices)]
        reference = sorted_pauli_suzuki_2_reference(
            quadrature.l_part,
            quadrature.h_part,
            k_value=float(k_value),
            final_time=final_time,
            reps=options.trotter_steps,
        )
        expected = np.exp(1.0j * np.angle(coefficient)) * reference
        assert np.max(np.abs(block - expected)) <= 1.0e-11


def test_budgeted_first_order_steps_follow_independent_prop9_bound() -> None:
    """Budgeted Lie-1 LCHS takes each node's smallest r with W*T**2/r <= 0.1*tolerance.

    CSTWZ (Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020)
    Prop. 9, Eq. (120), bounds one Lie
    step by (t**2/2)*sum_g1 ||sum_{g2>g1} [H_g2, H_g1]||. The selector bounds
    each tail norm by the triangle inequality, with ||[c P, c' Q]|| = 2|c c'|
    for anticommuting Pauli strings, so the error of r steps of total time T
    is bounded by W*T**2/r, with W the sum of |c c'| over anticommuting pairs
    (the r-step rule of CSTWZ doi:10.1103/PhysRevX.11.011020, Sec. V B). W is
    recomputed here
    from trace-projected coefficients and matrix anticommutators, without the
    library's Pauli decomposition or symplectic masks.
    """
    from fractions import Fraction
    from itertools import combinations, product
    from math import ceil

    from _lchs_suzuki_reference import sorted_pauli_lie_1_reference
    from nwqlib import LinearDynamics, plan, solve

    # Dyadic Pauli coefficients and the integer k nodes (0, 1, -2, -1) of
    # signed_binary_uniform with lsb_position=0 make every trace projection
    # exact, so no pruning charge enters and each step count is an exact
    # ceiling.
    l_part = pauli_matrix("II") + 0.25 * pauli_matrix("ZZ") + 0.125 * pauli_matrix("XI")
    h_part = 0.5 * pauli_matrix("YX") + 0.25 * pauli_matrix("IZ") - 0.375 * pauli_matrix("ZI")
    tolerance, time = 0.1, 0.5
    initial = np.array([1.0, 0.5j, -0.25, 0.5])
    problem = LinearDynamics(A=l_part + 1.0j * h_part, initial_state=initial, time=time)
    method = LCHS(
        hamiltonian_evolution_backend="trotter_error_budgeted",
        trotter_order=1,
        approximation_tolerance=tolerance,
        k_quadrature=ProviderConfig(
            implementation="signed_binary_uniform",
            parameters={"num_qubits": 2, "lsb_position": 0},
        ),
    )
    chosen = plan(problem, method=method, execution="quantum")
    data = chosen._native["native_data"]
    select, summary = data.select_data.plan, data.select_data.quadrature
    assert chosen.reconstruction.psd_shift == 0.0
    assert tuple(data.quadrature.k_nodes) == (0.0, 1.0, -2.0, -1.0)
    assert select.formula_order == summary["trotter_formula_order"] == 1
    # The per-node allowance is LCHS_TROTTER_EPSILON_FRACTION = 0.1 times the tolerance.
    budget = Fraction(0.1 * tolerance)
    labels = ["".join(letters) for letters in product("IXYZ", repeat=2)][1:]
    expected_steps = []
    for k_value, record in zip(
        data.quadrature.k_nodes, summary["trotter_node_records"], strict=True
    ):
        generator = h_part + k_value * l_part
        terms = [
            (pauli_matrix(label), np.trace(pauli_matrix(label) @ generator).real / 4)
            for label in labels
        ]
        terms = [(matrix, value) for matrix, value in terms if value != 0.0]
        weight = sum(
            (
                Fraction(abs(left)) * Fraction(abs(right))
                for (p, left), (q, right) in combinations(terms, 2)
                if np.array_equal(p @ q, -(q @ p))
            ),
            Fraction(0),
        )
        steps = max(1, ceil(weight * Fraction(time) ** 2 / budget))
        expected_steps.append(steps)
        structural = float(weight * Fraction(time) ** 2 / steps)
        assert record["pruned_l1_mass"] == 0.0 and record["step_count"] == steps
        assert record["pf_bound_value"] == pytest.approx(structural, rel=4 * np.finfo(float).eps)
        assert structural <= 0.1 * tolerance
        # Eq. (120) with dense tail sums in the applied (sorted-label) order
        # is at most the structural value, and the actual Lie product obeys
        # it. 1e-12 absorbs binary64 evaluation of both sides.
        tail_norms = 0.0
        for index, (p, value) in enumerate(terms):
            tail = sum((c * m for m, c in terms[index + 1 :]), np.zeros((4, 4), complex))
            tail_norms += np.linalg.norm(tail @ (value * p) - (value * p) @ tail, 2)
        dense = tail_norms * time**2 / (2 * steps)
        lie = sorted_pauli_lie_1_reference(
            l_part, h_part, k_value=k_value, final_time=time, reps=steps
        )
        exact = scipy.linalg.expm(-1.0j * time * generator)
        assert np.linalg.norm(lie - exact, 2) <= dense + 1.0e-12
        assert dense <= structural + 1.0e-12
    assert select.branch_step_counts == tuple(expected_steps)
    assert len(set(expected_steps)) > 1

    # The multiplexed SELECT applies each branch's own Lie-1 step count.
    address_qubits = select.padded_node_count.bit_length() - 1
    realized = np.asarray(
        Operator(build_multiplexed_product_formula_select(select, num_system_qubits=2)).data
    )
    expected_solution = np.zeros(4, dtype=complex)
    for branch, (k_value, coefficient) in enumerate(
        zip(data.quadrature.k_nodes, data.select_data.coefficients, strict=True)
    ):
        lie = sorted_pauli_lie_1_reference(
            l_part, h_part, k_value=k_value, final_time=time, reps=expected_steps[branch]
        )
        indices = [branch + (basis << address_qubits) for basis in range(4)]
        np.testing.assert_allclose(
            realized[np.ix_(indices, indices)],
            np.exp(1.0j * np.angle(coefficient)) * lie,
            atol=1.0e-11,
            rtol=0.0,
        )
        expected_solution += coefficient * (lie @ initial)

    # Classical execution selects the same steps and applies the same products.
    host = plan(problem, method=method, execution="classical")
    host_steps = tuple(
        node["record"].step_count for node in host._native["native_data"].host_actions["applications"][0]["positions"]
    )
    assert host_steps == tuple(expected_steps)
    np.testing.assert_allclose(solve(host).solution, expected_solution, atol=1.0e-12, rtol=0.0)


def test_dense_select_slot_cap_refuses_at_planning_before_native_construction(monkeypatch) -> None:
    """More padded SELECT slots than max_dense_select_slots is refused, not raised or built."""
    from nwqlib import LinearDynamics, plan
    from nwqlib.blocks import lowering

    def forbidden(*args, **kwargs):
        pytest.fail("slot-cap planning lowered a native circuit")

    monkeypatch.setattr(lowering, "_lower_qiskit", forbidden)
    options = {("approximation_tolerance" if key == "epsilon" else key): value
               for key, value in lchs_quadrature_options().items()}
    # A fixed two-bit k grid isolates slot-cap admission from quadrature search.
    options["k_quadrature"] = ProviderConfig(implementation="signed_binary_uniform",
        parameters={"num_qubits": 2, "lsb_position": 0})
    # Four declared grid nodes require exactly four SELECT slots.
    problem = LinearDynamics(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1.0, 0.0], time=0.1)
    with pytest.raises(ValueError, match=r"4 branches, 4 padded SELECT slots .*max_dense_select_slots=3"):
        plan(problem, method=LCHS(**options, max_dense_select_slots=3))
    assert plan(problem, method=LCHS(**options, max_dense_select_slots=4)).execution == "quantum"


# Homogeneous: 8 branches, P = 8 and a = 3 at D = 2, so the intake of
# prepare_lcu_gate_data is 16*P*(a+1) = 512 work units. The construction work
# of the dense branches is larger, 49032 units for their exponentials, their
# array operations and Qiskit's control of each branch's U gate by the three
# address bits (parameters.construction_work). With a source and one Duhamel
# node: 16 branches, P = 16 and a = 4, so 1280 and 125448.
_DENSE_INTAKE_CASES = ((None, 500, 512), ([0.2, -0.1], 1000, 1280))


def _dense_intake_plan(source, limit):
    from nwqlib import LinearDynamics, plan
    grid = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 3, "lsb_position": -1})
    problem = LinearDynamics(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1.0, 0.0], time=0.1, source=source)
    return problem, plan(problem, method=LCHS(hamiltonian_evolution_backend="dense_exact", k_quadrature=grid,
                                              duhamel_nodes=1, max_select_work=limit))


@pytest.mark.parametrize(("source", "below", "intake"), _DENSE_INTAKE_CASES)
def test_planning_refuses_a_branch_select_that_the_lcu_intake_refuses(source, below, intake) -> None:
    """Planning applies the LCU intake that native construction applies to the branch-controlled SELECT.

    A Plan whose limit lies below the intake law must be refused at planning
    by the intake, not only at construction. At the intake law the intake
    admits the Plan, and the larger work of the dense branch exponentials
    refuses it. With the default limit the Plan is admitted and its SELECT
    is built.
    """
    from types import SimpleNamespace
    from nwqlib.algorithms.lchs.native import construct_select

    with pytest.raises(ValueError, match="LCU coefficient intake exceeds max_work"):
        _dense_intake_plan(source, below)
    with pytest.raises(ValueError, match="native construction exceeds max_select_work"):
        _dense_intake_plan(source, intake)
    problem, selected = _dense_intake_plan(source, 100_000_000)
    construct_select(SimpleNamespace(_payload=(selected._native["native_data"], problem.elapsed_time)), (), None)
