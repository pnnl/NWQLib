"""Independent equivalence gates for homogeneous LCHS SELECT synthesis."""

from __future__ import annotations


import numpy as np
import pytest
import scipy.linalg
from qiskit.quantum_info import Operator

from conftest import lchs_dense_quadrature, lchs_quadrature_options
from _lchs_suzuki_reference import pauli_matrix, sorted_pauli_suzuki_2_reference
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs import LCHS
from nwqlib.algorithms.lchs.select_synthesis import (
    build_multiplexed_product_formula_select,
)
from nwqlib.algorithms.lchs.time_independent_terms import (
    generate_lchs_product_formula_select_plan,
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
    quadrature, *parts = lchs_dense_quadrature(l_part + 1.0j * h_part, 2.0, options)
    data = generate_lchs_product_formula_select_plan(
        quadrature=quadrature,
        parts=tuple(parts),
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

    The second case adds 2**40*I to H. On its dyadic grid, with the shifted
    L = [[.25, .25], [.25, .25]], the identity phase -T*(2**40 + k/4) is an
    exact binary64 number, which NumPy's exp reduces against pi itself. The
    complex near_optimal_eq7 coefficient phases survive only when they are
    joined to the reduced identity phase: added to the unreduced angle they
    round at its spacing of about 1e-4, and reducing that angle with the
    rounded 2*pi is off by about 3e-5.

    Its node cutoff is 1e-12*2**40, about 1.1 (combine_aligned), so the term
    k*X/4 of L is removed at every node of this grid. The reference omits that
    term, and Duhamel's T*|k|/4 bounds the distance from the product that keeps it.
    """
    cases = (
        (np.array([[0.4, 0.05], [0.05, 0.5]], dtype=complex), 0.8, LCHS(
            **{("approximation_tolerance" if key=="epsilon" else key):value
               for key,value in lchs_quadrature_options().items()},
            hamiltonian_evolution_backend="trotter", trotter_steps=2), 0.0),
        (np.array([[0.5, 0.25], [0.25, 0.5]])+1j*2.0**40*np.eye(2), 0.75, LCHS(
            lchs_kernel=ProviderConfig(implementation="near_optimal_eq7"),
            k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                        parameters={"num_qubits": 2, "lsb_position": -2}),
            hamiltonian_evolution_backend="trotter", trotter_steps=2), 0.25),
    )
    for matrix, final_time, options, removed_x in cases:
        quadrature, l_part, h_part = lchs_dense_quadrature(matrix, final_time, options)
        data = generate_lchs_product_formula_select_plan(
            quadrature=quadrature,
            parts=(l_part, h_part),
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
            references = [
                sorted_pauli_suzuki_2_reference(
                    part,
                    h_part,
                    k_value=float(k_value),
                    final_time=final_time,
                    reps=options.trotter_steps,
                )
                for part in (l_part - removed_x * pauli_matrix("X"), l_part)
            ]
            expected, kept_all = (np.exp(1.0j * np.angle(coefficient)) * reference for reference in references)
            assert np.max(np.abs(block - expected)) <= 1.0e-11
            assert np.linalg.norm(block - kept_all, 2) <= final_time * abs(k_value) * removed_x + 2.0e-11


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


def _gauss_calls(monkeypatch):
    """Record each construction of the composite Gauss rule at its owner."""
    from nwqlib.algorithms.lchs import providers
    calls, original = [], providers.composite_gauss_grid

    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(providers, "composite_gauss_grid", counted)
    return calls


_SLOT_PROBLEM = dict(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1.0, 0.0], time=0.1)
_DENSE = dict(hamiltonian_evolution_backend="dense_exact")


def _padded(count):
    return 2**int(np.ceil(np.log2(count)))


def test_composite_gauss_slot_cap_refuses_before_the_rule_is_built(monkeypatch) -> None:
    """A rule with M nodes needs at least the power of two P >= M slots, so a limit below P refuses without building it.

    M and P come from an admitted Plan. M <= limit < P is the padded case, where the node count alone fits. Equality is
    legal and selects the same rule. The final check compares the same P (`parameters.select_parameters`).
    """
    from nwqlib import LinearDynamics, plan
    calls = _gauss_calls(monkeypatch)
    problem = LinearDynamics(**_SLOT_PROBLEM)
    full = plan(problem, method=LCHS(**_DENSE))
    nodes, slots = full.reconstruction.physical_branches, full.reconstruction.padded_branches
    assert slots == _padded(nodes) and nodes < slots and len(calls) == 1
    calls.clear()
    for limit in (slots-1, nodes, 1):
        with pytest.raises(ValueError, match=rf"max_dense_select_slots={limit}\.") as refusal:
            plan(problem, method=LCHS(**_DENSE, max_dense_select_slots=limit))
        assert calls == []
        assert f"at least {nodes} branches and {slots} padded SELECT slots" in str(refusal.value)
    # Following the refusal's remedy, a limit of P, leads past it.
    equal = plan(problem, method=LCHS(**_DENSE, max_dense_select_slots=slots))
    assert len(calls) == 1 and equal.reconstruction.content_id == full.reconstruction.content_id


@pytest.mark.parametrize(("execution", "options"), [
    ("classical", _DENSE),
    ("quantum", dict(hamiltonian_evolution_backend="trotter", trotter_steps=1)),
    ("quantum", dict(hamiltonian_evolution_backend="trotter_error_budgeted")),
])
def test_slot_cap_leaves_routes_without_a_dense_select_register(monkeypatch, execution, options) -> None:
    """Classical evaluation and product-formula branches synthesize no controlled dense exponential per slot, so the limit does not apply."""
    from nwqlib import LinearDynamics, plan
    calls = _gauss_calls(monkeypatch)
    chosen = plan(LinearDynamics(**_SLOT_PROBLEM), method=LCHS(**options, max_dense_select_slots=1), execution=execution)
    assert len(calls) == 1 and chosen.reconstruction.physical_branches > 1


def test_coefficient_table_request_has_no_slot_limit(monkeypatch) -> None:
    """The same long-time problem is refused as a dense SELECT and returned as a coefficient table beyond the default limit.

    `LCHS()` reaches the early refusal through its default backend `"auto"`, which resolves to dense_exact for a dense A.
    """
    from nwqlib import LinearDynamics, plan
    from nwqlib.algorithms.lchs import resolve_lchs_coefficient_plan
    calls = _gauss_calls(monkeypatch)
    problem = LinearDynamics(**{**_SLOT_PROBLEM, "time": 1000.0})
    with pytest.raises(ValueError, match="max_dense_select_slots=4096"):
        plan(problem, method=LCHS())
    assert calls == []
    assert len(resolve_lchs_coefficient_plan(problem).coefficients) > 4096 and len(calls) == 1


def test_source_layouts_keep_the_exact_final_slot_check(monkeypatch) -> None:
    """The early bound is one application's M slots, because a source layout can hold fewer applications than planned.

    Two applications hold 2*M slots, so a limit between P(M) and the final count builds the rule and is refused by the
    final check with its exact counts. A source-only layout holds M slots, and so does a layout in which omission removes
    the source application (Initial 2**220, source 2**-1000 at T = 2**-990, as in
    `test_selected_grid_applies_the_source_slots_that_planning_kept`). In both, the limit P(M) stays legal, so a bound
    multiplied by the provisional application count would refuse a Plan that runs.
    """
    from math import ldexp
    from nwqlib import LinearDynamics, plan
    calls = _gauss_calls(monkeypatch)
    both = LinearDynamics(**_SLOT_PROBLEM, source=[0.2, -0.1])
    full = plan(both, method=LCHS(**_DENSE, duhamel_nodes=1))
    nodes, branches, slots = len(full._native["native_data"].quadrature.k_nodes), full.reconstruction.physical_branches, full.reconstruction.padded_branches
    assert branches == 2*nodes and slots == _padded(branches) > _padded(nodes)
    calls.clear()
    with pytest.raises(ValueError, match=rf"{branches} branches, {slots} padded SELECT slots and \d+ total qubits; "
                                         rf"max_dense_select_slots={slots-1}\."):
        plan(both, method=LCHS(**_DENSE, duhamel_nodes=1, max_dense_select_slots=slots-1))
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(ValueError, match="max_dense_select_slots") as refusal:
        plan(both, method=LCHS(**_DENSE, duhamel_nodes=1, max_dense_select_slots=_padded(nodes)-1))
    assert calls == []
    assert f"at least {nodes} branches" in str(refusal.value)
    source_only = LinearDynamics(**{**_SLOT_PROBLEM, "initial_state": [0.0, 0.0]}, source=[0.2, -0.1])
    alone = plan(source_only, method=LCHS(**_DENSE, duhamel_nodes=1, max_dense_select_slots=_padded(nodes)))
    assert alone.reconstruction.physical_branches == nodes
    dropped = dict(A=np.diag([.25, .5]), initial_state=[ldexp(1., 220), 0.], source=[0., ldexp(1., -1000)], time=ldexp(1., -990))
    options = dict(**_DENSE, duhamel_nodes=1, source_drop_allowance=1e-200)
    kept = plan(LinearDynamics(**dropped), method=LCHS(**options))
    layout = kept._native["native_data"].source_layout
    size = len(kept._native["native_data"].quadrature.k_nodes)
    assert len(layout.applications) == 1 and layout.dropped_bound.mantissa > 0 and len(layout.phases) == size
    assert _padded(2*size) > _padded(size)
    limited = plan(LinearDynamics(**dropped), method=LCHS(**options, max_dense_select_slots=_padded(size)))
    assert limited.reconstruction.padded_branches == _padded(size)
    calls.clear()
    with pytest.raises(ValueError, match="max_dense_select_slots"):
        plan(LinearDynamics(**dropped), method=LCHS(**options, max_dense_select_slots=_padded(size)-1))
    assert calls == []


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
    with pytest.raises(ValueError, match=r"native construction needs \d+ work units of max_select_work, \d+ to build it"):
        _dense_intake_plan(source, intake)
    problem, selected = _dense_intake_plan(source, 100_000_000)
    construct_select(SimpleNamespace(_payload=(selected._native["native_data"], problem.elapsed_time)), (), None)


def test_select_helper_records_and_refuses_its_own_work_limit() -> None:
    """The direct helper records the same work with or without its limit and states that limit when it refuses.

    The one-qubit plan counts 266 units for the Pauli decomposition and
    5416 for the angle tables, 5682 in total. Its Method sets no limit, so
    the refusal one unit short names the helper's own value.
    """
    method = LCHS(hamiltonian_evolution_backend="trotter", approximation_tolerance=0.2)
    quadrature, *parts = lchs_dense_quadrature(np.array([[0.5, 0.125j], [0.125j, 1.0]]), 0.1, method)
    args = dict(quadrature=quadrature, parts=tuple(parts), final_time=0.1, method=method)
    records = []
    for limit in ({}, {"max_select_work": None}, {"max_select_work": 5682}):
        records.append({})
        generate_lchs_product_formula_select_plan(**args, **limit, work_ledger=records[-1])
    assert records == [{"work": 5682}] * 3
    with pytest.raises(ValueError, match=r"angle-table selection needs 5416 units of max_select_work, above the "
                                         r"5415 left after 266 already charged; this stage needs max_select_work "
                                         r"to be at least 5682 \(now 5681\)"):
        generate_lchs_product_formula_select_plan(**args, max_select_work=5681)


def test_select_construction_checks_its_circuit_bytes_before_the_first_repetition(monkeypatch) -> None:
    """A Plan with ten million product-formula steps is planned, and its SELECT is refused before any repetition.

    One system qubit, four address rows and the explicit multiplexor give
    three occurrences per step. The size formula counts the instructions
    of the built multiplexed SELECT exactly for two and three steps, and
    bounds those of the structured SELECT of the same affine table. With
    10**7 steps the
    Plan is admitted under max_bytes=1,000,000, and lowering refuses the
    SELECT circuit at its byte check, before the first repeated rotation.
    """
    from nwqlib import LinearDynamics, NormSquared, plan
    from nwqlib.algorithms.lchs import select_synthesis
    from nwqlib.blocks.lowering import lower_qiskit

    grid = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1})
    problem = LinearDynamics(A=np.array([[.4 + .2j, .1], [.1, .4 - .2j]]), initial_state=[1., 0.], time=.1)

    def selected(steps, implementation="multiplexor"):
        return plan(problem, method=LCHS(hamiltonian_evolution_backend="trotter", k_quadrature=grid,
                                         lcu_select_implementation=implementation, max_bytes=1_000_000,
                                         trotter_steps=steps), output=NormSquared(), execution="quantum", seed=13)

    for steps in (2, 3):
        select_plan = selected(steps)._native["native_data"].select_data.plan
        built = build_multiplexed_product_formula_select(select_plan, num_system_qubits=1)
        assert select_synthesis.product_formula_select_size(select_plan, implementation="multiplexor")[0] == len(
            built.data)
        select_plan = selected(steps, "structured")._native["native_data"].select_data.plan
        structured = select_synthesis.build_structured_product_formula_select(select_plan, num_system_qubits=1)
        assert select_synthesis.product_formula_select_size(select_plan, implementation="structured")[0] >= len(
            structured.data)

    def repeated(*args, **kwargs):
        pytest.fail("a SELECT repetition was appended")
    monkeypatch.setattr(select_synthesis, "_append_multiplexed_pauli_rotation", repeated)
    huge = selected(10_000_000)
    assert sum(huge._native["native_data"].select_data.plan.occurrence_block_repetitions) == 10_000_000
    _, construction = huge.resolve(huge.experiments[0].name)._selected_construction(huge)
    chosen = {selection.content_id for selection in construction.selections}
    with pytest.raises(ValueError, match="product-formula SELECT circuit needs about"):
        lower_qiskit(construction, blocks=tuple(block for block in huge.blocks if block.record.content_id in chosen))


def test_branch_controlled_select_checks_its_bytes_before_qiskit_expands_a_branch(monkeypatch) -> None:
    """A ten-million-step branch-controlled SELECT is refused before any branch is built or expanded.

    Qiskit's control unrolls each whole branch, so the controlled
    definitions of a built two-qubit SELECT with four address rows hold
    exactly the counted instructions for two and three steps of both
    orders. With 10**7 steps on one system qubit the Plan is admitted under
    max_bytes=1,000,000, and lowering refuses the SELECT at its byte check
    before any branch circuit is built or any Pauli evolution is expanded.
    """
    from types import SimpleNamespace

    from qiskit.circuit.library import PauliEvolutionGate

    from nwqlib import LinearDynamics, NormSquared, plan
    from nwqlib.algorithms.lchs import select_synthesis
    from nwqlib.algorithms.lchs.native import construct_select
    from nwqlib.blocks.lowering import lower_qiskit

    grid = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1})
    m = np.random.default_rng(3).normal(size=(4, 4)) + 1j * np.random.default_rng(4).normal(size=(4, 4))
    two = LinearDynamics(A=m @ m.conj().T / 8 + .3j * (m + m.conj().T) / 2, initial_state=[1., 0., 0., 0.], time=.1)
    one = LinearDynamics(A=np.array([[.4 + .2j, .1], [.1, .4 - .2j]]), initial_state=[1., 0.], time=.1)

    def selected(problem, steps, order=2, **limits):
        return plan(problem, method=LCHS(hamiltonian_evolution_backend="trotter", k_quadrature=grid,
                                         lcu_select_implementation="branch_controlled", trotter_order=order,
                                         trotter_steps=steps, **limits),
                    output=NormSquared(), execution="quantum", seed=13)

    for order in (1, 2):
        for steps in (2, 3):
            data = selected(two, steps, order)._native["native_data"]
            built = construct_select(SimpleNamespace(_payload=(data, two.elapsed_time)), (), None)
            emitted = sum(len(instruction.operation._definition.data) for instruction in built.data)
            assert select_synthesis.branch_controlled_select_size(data.select_data.plan, slots=4)[0] == emitted

    huge = selected(one, 10_000_000, max_bytes=1_000_000)
    assert max(huge._native["native_data"].select_data.plan.branch_step_counts) == 10_000_000

    def expanded(*args, **kwargs):
        pytest.fail("a branch was built or a Pauli evolution was expanded")
    monkeypatch.setattr(select_synthesis, "_build_product_formula_branch", expanded)
    monkeypatch.setattr(PauliEvolutionGate, "_define", expanded)
    _, construction = huge.resolve(huge.experiments[0].name)._selected_construction(huge)
    chosen = {selection.content_id for selection in construction.selections}
    with pytest.raises(ValueError, match="branch-controlled product-formula SELECT needs about"):
        lower_qiskit(construction, blocks=tuple(block for block in huge.blocks if block.record.content_id in chosen))
