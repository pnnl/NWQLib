"""Independent equivalence gates for homogeneous LCHS SELECT synthesis."""

from __future__ import annotations


import numpy as np
from nwqlib.core.records import FrozenArray
import pytest
import scipy.linalg
from qiskit.quantum_info import Operator

from conftest import lchs_quadrature_options
import nwqlib.algorithms.lchs.time_independent_terms as terms_module
from _lchs_suzuki_reference import pauli_matrix, sorted_pauli_suzuki_2_reference
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs import LCHS
from nwqlib.algorithms.lchs.native import resolve_lcu_select_implementation
from nwqlib.algorithms.lchs.select_synthesis import (
    LCHSProductFormulaSelectPlan,
    build_multiplexed_product_formula_select,
)
from nwqlib.algorithms.lchs.time_independent_terms import (
    generate_lchs_product_formula_select_plan,
    generate_lchs_quadrature,
)


def _manual_plan(*, address_qubits: int, system_qubits: int) -> LCHSProductFormulaSelectPlan:
    padded = 1 << address_qubits
    physical = padded if address_qubits == 0 else padded - 1
    labels = (
        "I" * (system_qubits - 1) + "X",
        "Z" + "I" * (system_qubits - 1),
    )
    tables = tuple(
        tuple(
            0.07 * (occurrence + 1) * (branch + 1) if branch < physical else 0.0
            for branch in range(padded)
        )
        for occurrence in range(len(labels))
    )
    return LCHSProductFormulaSelectPlan(
        pauli_labels=tuple(sorted(set(labels))),
        occurrence_schedule=labels,
        occurrence_angle_tables=FrozenArray(np.array(tables, dtype=np.float64)),
        occurrence_block_repetitions=(1,),
        branch_step_counts=(1,) * physical,
        max_step_count=1,
        identity_phases=tuple(-0.031 * (branch + 1) for branch in range(physical)),
        coefficient_phases=tuple(0.043 * (branch + 1) for branch in range(physical)),
        physical_node_count=physical,
        padded_node_count=padded,
        branch_to_node=tuple(range(physical)) + (None,) * (padded - physical),
        pruning_decisions=((),) * physical,
        pruned_l1_mass=(0.0,) * physical,
        formula_order=2,
        repetitions=1,
        structure_certificate=None,
        structured_generator_payload=None,
    )


@pytest.mark.parametrize("address_qubits", range(6))
def test_multiplexed_select_matches_explicit_branch_products_and_padding(
    address_qubits: int,
) -> None:
    """Cover one-three system qubits and zero-five address qubits."""

    system_qubits = 1 + address_qubits % 3
    plan = _manual_plan(
        address_qubits=address_qubits,
        system_qubits=system_qubits,
    )
    select = build_multiplexed_product_formula_select(
        plan,
        num_system_qubits=system_qubits,
    )
    realized = np.asarray(Operator(select).data)
    system_dimension = 1 << system_qubits
    for branch in range(plan.padded_node_count):
        indices = [branch + (basis << address_qubits) for basis in range(system_dimension)]
        block = realized[np.ix_(indices, indices)]
        expected = np.eye(system_dimension, dtype=complex)
        for label, table in zip(
            plan.occurrence_schedule,
            plan.occurrence_angle_tables.array,
            strict=True,
        ):
            expected = scipy.linalg.expm(-0.5j * table[branch] * pauli_matrix(label)) @ expected
        if branch < plan.physical_node_count:
            phase = plan.identity_phases[branch] + plan.coefficient_phases[branch]
            expected = np.exp(1.0j * phase) * expected
        assert np.max(np.abs(block - expected)) <= 1.0e-11
    if address_qubits:
        assert select.count_ops().get("nwqlib_phase_diagonal", 0) == 1
    assert select.count_ops().get("ucrz", 0) == len(plan.occurrence_angle_tables)


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
        matrix=l_part + 1.0j * h_part,
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
    assert repetition == plan.max_step_count


def test_product_formula_select_plan_rejects_unpadded_dimension_first(monkeypatch) -> None:
    monkeypatch.setattr(terms_module, "generate_lchs_quadrature",
                        lambda **_: pytest.fail("quadrature selected before dimension admission"))
    with pytest.raises(ValueError, match=r"power of two \(plan\(\) pads"):
        generate_lchs_product_formula_select_plan(
            matrix=np.diag([0.2, 0.3, 0.4]).astype(complex), final_time=0.1,
            method=LCHS(hamiltonian_evolution_backend="trotter"))


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
    data = generate_lchs_product_formula_select_plan(
        matrix=matrix,
        final_time=final_time,
        method=options,
    )
    quadrature = generate_lchs_quadrature(
        matrix=matrix,
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


def test_variable_elapsed_schedule_matches_independent_node_products() -> None:
    """Variable elapsed times and a padded branch expose wrong repetition, identity-phase and
    coefficient-phase handling.
    """
    from nwqlib.algorithms.lchs.select_synthesis import _build_product_formula_branch

    l_part = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
    h_part = np.diag([0.3, -0.3]).astype(complex)
    times = np.array([0.1, 0.4, 1.0, 0.0])
    nodes = np.array([0.5, -1.0, 1.5, 0.0])
    coefficients = np.array([0.2, 0.1j, -0.15, 0.0])
    options = LCHS(
        hamiltonian_evolution_backend="trotter_error_budgeted", approximation_tolerance=0.01
    )
    plan, records, _ = terms_module._build_product_formula_select_plan(
        decomposition=terms_module._trotter_pauli_decomposition(l_part, h_part),
        k_values=nodes,
        elapsed_times=times,
        coefficients=coefficients,
        branch_to_node=(0, 1, 2, None),
        method=options,
        address_structure={},
    )
    assert len(set(plan.branch_step_counts[:3])) > 1
    assert plan.branch_step_counts == tuple(record.step_count for record in records)
    actual = np.asarray(Operator(build_multiplexed_product_formula_select(plan, num_system_qubits=1)).data)
    for branch in range(4):
        expected = (
            sorted_pauli_suzuki_2_reference(
                l_part, h_part, k_value=nodes[branch], final_time=times[branch],
                reps=plan.branch_step_counts[branch],
            ) * np.exp(1.0j * np.angle(coefficients[branch]))
            if branch < 3 else np.eye(2)
        )
        indices = [branch, branch + 4]
        np.testing.assert_allclose(actual[np.ix_(indices, indices)], expected, atol=1.0e-12, rtol=0.0)
        branch_circuit = _build_product_formula_branch(plan, branch, num_system_qubits=1)
        branch_circuit.global_phase += (
            plan.identity_phases[branch] + plan.coefficient_phases[branch]
        )
        np.testing.assert_allclose(
            Operator(branch_circuit.decompose()).data, expected, atol=1.0e-12, rtol=0.0
        )


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


def test_lcu_select_resolution_is_strict_and_backend_specific() -> None:
    base = LCHS()
    assert (
        resolve_lcu_select_implementation(
            base,
            he_backend="dense_exact",
        )["resolved_lcu_select_implementation"]
        == "branch_controlled"
    )
    product = base.revise(hamiltonian_evolution_backend='trotter')
    assert (
        resolve_lcu_select_implementation(
            product,
            he_backend="trotter",
        )["resolved_lcu_select_implementation"]
        == "multiplexor"
    )
    # Options validation is deliberately problem-independent.  A manual
    # structured request becomes strict once the plan-aware resolver runs.
    structured = product.revise(lcu_select_implementation='structured')
    assert structured.lcu_select_implementation == "structured"
    with pytest.raises(
        ValueError,
        match="no structure certificate is available",
    ):
        resolve_lcu_select_implementation(
            structured,
            he_backend="trotter",
            select_plan=_manual_plan(address_qubits=2, system_qubits=1),
        )
    with pytest.raises(ValueError, match="elementary product-formula plan"):
        resolve_lcu_select_implementation(
            base.revise(lcu_select_implementation='multiplexor'),
            he_backend="dense_exact",
        )
    qsp = base.revise(hamiltonian_evolution_backend='qsp_block_encoding')
    assert (
        resolve_lcu_select_implementation(
            qsp,
            he_backend="qsp_block_encoding",
        )["resolved_lcu_select_implementation"]
        == "compiled_qsp"
    )
    with pytest.raises(ValueError, match="internal compiled_qsp"):
        resolve_lcu_select_implementation(
            qsp.revise(lcu_select_implementation='branch_controlled'),
            he_backend="qsp_block_encoding",
        )


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


@pytest.mark.parametrize("source", (None, [0.2, -0.1]))
def test_pauli_decomposition_and_product_formula_plan_share_one_work_limit(monkeypatch, source) -> None:
    """Homogeneous and constant-source planning charge the decomposition and the plan to one max_select_work.

    The decomposition law is time_independent_terms._admit_pauli_decomposition,
    (34*q+28)*N + 2*H with N = d**2 and H the sum of (j+1)*4**j over the
    half-label widths. At d = 2 (q = 1, widths 0 and 1) that is
    62*4 + 2*(1 + 8) = 266 units. One unit below refuses the decomposition.
    Work equal to the limit is admitted, so a limit of exactly 266 passes
    the decomposition and leaves nothing for the plan, whose own check then
    refuses. The plan must receive the rest of the limit on both routes.
    """
    from nwqlib import LinearDynamics, plan
    received = []
    original = terms_module._build_product_formula_select_plan

    def recording(*args, **kwargs):
        received.append(kwargs["max_select_work"])
        return original(*args, **kwargs)

    monkeypatch.setattr(terms_module, "_build_product_formula_select_plan", recording)
    problem = LinearDynamics(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1.0, 0.0], time=0.1, source=source)
    grid = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": 0})

    def method(limit):
        return LCHS(hamiltonian_evolution_backend="trotter", k_quadrature=grid, duhamel_nodes=1,
                    max_select_work=limit)

    decomposition = 266
    with pytest.raises(ValueError, match="LCHS Pauli decomposition needs 266 units of max_select_work, above the limit 265"):
        plan(problem, method=method(decomposition - 1))
    with pytest.raises(ValueError, match="product-formula branch selection needs \\d+ units of max_select_work, above the 0 left after 266 already charged"):
        plan(problem, method=method(decomposition))
    received.clear()
    plan(problem, method=method(10**6))
    assert received == [10**6 - decomposition]


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


def _angle_table_fixture(q, address_qubits, *, affine=False):
    """Homogeneous A = I/2 + iH with two fixed Trotter steps: generic random H, or one Z term."""
    import nwqlib

    d = 1 << q
    if affine:
        h = np.diag([0.2, -0.2]).astype(complex)
    else:
        rng = np.random.default_rng(3500 + q)
        raw = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        h = (raw + raw.conj().T) / (8 * d)
    method = LCHS(
        k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                    parameters={"num_qubits": address_qubits, "lsb_position": 0}),
        hamiltonian_evolution_backend="trotter", trotter_order=1 if affine else 2, trotter_steps=2,
        lcu_select_implementation="auto" if affine else "multiplexor")
    problem = nwqlib.LinearDynamics(A=0.5 * np.eye(d) + 1j * h, initial_state=np.ones(d, dtype=complex) / np.sqrt(d),
                                    time=0.2)
    return problem, method


def _captured_builder_call(monkeypatch, problem, method):
    """Plan once and return the keyword arguments and result of the product-formula builder."""
    import nwqlib

    calls = []
    original = terms_module._build_product_formula_select_plan

    def capture(**kwargs):
        calls.append((kwargs, original(**kwargs)))
        return calls[-1][1]

    with monkeypatch.context() as patch:
        patch.setattr(terms_module, "_build_product_formula_select_plan", capture)
        nwqlib.plan(problem, method=method, execution="quantum")
    (call,) = calls
    return call


@pytest.mark.parametrize("q, address_qubits, affine", [(3, 4, False), (4, 4, False), (1, 3, True)])
def test_adopted_angle_table_equals_the_copied_table(monkeypatch, q, address_qubits, affine):
    """The ownership transfer selects the same Plan, slot records and canonical table as the public copy."""
    kwargs, (owned, records, _) = _captured_builder_call(monkeypatch, *_angle_table_fixture(q, address_qubits,
                                                                                             affine=affine))
    with monkeypatch.context() as patch:
        patch.setattr(FrozenArray, "_from_owned_canonical_array", classmethod(lambda cls, values: cls(values)))
        copied, copied_records, _ = terms_module._build_product_formula_select_plan(**kwargs)
    assert owned == copied and records == copied_records
    assert owned.occurrence_angle_tables.to_json() == copied.occurrence_angle_tables.to_json()
    table = owned.occurrence_angle_tables.array
    assert table.dtype.str == "<f8" and table.flags.c_contiguous and not table.flags.writeable
    with pytest.raises(ValueError):
        table.flags.writeable = True


def test_owned_canonical_storage_matches_public_construction_and_refuses_shared_arrays():
    """A producer-canonicalized buffer freezes to the public constructor's bytes; borrowed arrays are refused."""
    values = np.array([-0.0, 5e-324, -1.5]) + 0.0
    assert FrozenArray._from_owned_canonical_array(values).to_json() == FrozenArray(
        np.array([-0.0, 5e-324, -1.5])).to_json()
    empty = FrozenArray._from_owned_canonical_array(np.zeros((0, 8), dtype="<f8"))
    assert empty == FrozenArray(np.zeros((0, 8))) and empty.array.shape == (0, 8)
    counts = FrozenArray._from_owned_canonical_array(np.arange(3, dtype=np.int64))
    assert counts == FrozenArray(np.arange(3))
    base = np.zeros(4)
    for borrowed in (base[::2], np.zeros(4, dtype=np.float32), np.asfortranarray(np.zeros((2, 2))).T[:, :1],
                     FrozenArray(np.zeros(2)).array):
        with pytest.raises(ValueError, match="owned canonical storage"):
            FrozenArray._from_owned_canonical_array(borrowed)


def test_angle_table_construction_holds_one_table_buffer(monkeypatch):
    """The angle table's ownership transfer fits below the table-plus-mask phase.

    The q=4, address=6 fixture has E=32576 entries. Between angle-table
    admission and Plan record construction, transfer owns one 8E-byte
    float64 buffer. A full-array Boolean mask would require at least 9E
    array bytes, and a copy at least 16E. The 8.5E threshold separates these
    phases while allowing E/2 bytes of fixed-fixture bookkeeping. On the
    qualified stack that bookkeeping is 6848 bytes, leaving 9440 bytes of
    pass margin. Requalify that margin when the allocation path or runtime
    changes. The bound concerns this construction phase, before affine checks.
    """
    import gc
    import tracemalloc
    from nwqlib.algorithms.lchs import select_synthesis
    from nwqlib.operators import access

    kwargs, (selected, _, _) = _captured_builder_call(
        monkeypatch, *_angle_table_fixture(4, 6)
    )
    entries = selected.occurrence_angle_tables.array.size
    assert entries == 32_576
    marks = {}
    check, record = access._check_bytes, select_synthesis.LCHSProductFormulaSelectPlan

    def admitted(size, limit, operation="input"):
        check(size, limit, operation)
        if operation == "product-formula angle tables":
            marks["start"] = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()

    def constructed(**fields):
        marks["peak"] = tracemalloc.get_traced_memory()[1]
        return record(**fields)

    gc.collect()
    with monkeypatch.context() as patch:
        patch.setattr(access, "_check_bytes", admitted)
        patch.setattr(select_synthesis, "LCHSProductFormulaSelectPlan", constructed)
        tracemalloc.start()
        try:
            terms_module._build_product_formula_select_plan(**kwargs)
        finally:
            tracemalloc.stop()
    growth = marks["peak"] - marks["start"]
    assert 8 * entries <= growth < 17 * entries // 2


def test_angle_fill_canonicalizes_signed_zero_and_refuses_nonfinite_angles(monkeypatch):
    """The smallest subnormal elapsed time underflows the angles of negative coefficients to -0.0.

    They are stored as +0.0, the public constructor's canonical zero.

    A NaN elapsed time gives nonfinite angles, which the fill loop refuses
    with the public constructor's message.
    """
    kwargs, _ = _captured_builder_call(monkeypatch, *_angle_table_fixture(3, 4))
    times = np.asarray(kwargs["elapsed_times"], dtype=float)
    zero, _, _ = terms_module._build_product_formula_select_plan(**{**kwargs, "elapsed_times": np.full_like(times, 5e-324)})
    table = zero.occurrence_angle_tables.array
    assert np.all(table == 0) and not np.signbit(table).any()
    assert zero.occurrence_angle_tables == FrozenArray(np.zeros(table.shape))
    with pytest.raises(ValueError, match="FrozenArray float64 elements must be finite"):
        terms_module._build_product_formula_select_plan(**{**kwargs, "elapsed_times": np.full_like(times, np.nan)})
