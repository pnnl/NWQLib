"""Independent recursive SELECT laws and explicit saved-data resource samples."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS,ProviderConfig
from nwqlib.algorithms.lchs.select_synthesis import build_multiplexed_product_formula_select
from nwqlib.algorithms.lchs.time_independent_terms import generate_lchs_product_formula_select_plan
from nwqlib.subroutines._multiplexors import product_formula_select_resource_law
TRANSPILE_OPTIONS=dict(basis_gates=['cx','u'],optimization_level=0,seed_transpiler=7)
def _mixed_two_problem() -> LinearDynamics:
    return LinearDynamics(
        A=np.array([[0.2, -0.1], [-0.1, 0.3]], dtype=complex),
        initial_state=np.array([1.0, 0.25], dtype=complex),
        time=0.1,
    )



def _heat_problem(num_qubits):
    size=1<<num_qubits
    matrix=.1*(2*np.eye(size)-np.eye(size,k=1)-np.eye(size,k=-1))
    return LinearDynamics(A=matrix,initial_state=np.eye(size)[0],time=.1)



def _pf_options(
    *,
    tier: str,
    order: int,
    steps: int,
    epsilon: float,
    truncation_multiplier: float,
) -> LCHS:
    return LCHS(
        lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 0.8}),
        approximation_tolerance=epsilon,
        k_quadrature=ProviderConfig(
            implementation="composite_gauss",
            parameters={"truncation_multiplier": truncation_multiplier},
        ),
        hamiltonian_evolution_backend="trotter",
        trotter_order=order,
        trotter_steps=steps,
    )



PF_CASES = {
    "mixed2_lie_a3": (_mixed_two_problem,1,1,.5,1.),
    "mixed2_suzuki_a4": (_mixed_two_problem,2,1,.5,2.),
    "heat4_suzuki_steps2_a5": (lambda: _heat_problem(2),2,2,.1,2.),
}

@pytest.mark.parametrize(
    ("case", "phase"),
    [
        *((case, None) for case in PF_CASES),
        ("mixed2_lie_a3", 0.0),
        ("mixed2_lie_a3", 1.0e-11),
        ("mixed2_lie_a3", 5.0e-324),
    ],
)
def test_product_formula_law_matches_recursive_built_select(case: str, phase) -> None:
    """Successor to the dense-proxy PF law: count the built SELECT recursively."""

    problem_factory, order, steps, epsilon, multiplier = PF_CASES[case]
    problem = problem_factory()
    options = _pf_options(
        tier="analytic",
        order=order,
        steps=steps,
        epsilon=epsilon,
        truncation_multiplier=multiplier,
    )
    data = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=options,
    )
    plan = data.plan
    if phase is not None:
        # All but the last address cancel exactly between the two phase sources.
        plan = replace(
            plan,
            coefficient_phases=(0.25,) * (plan.padded_node_count - 1) + (phase,),
            identity_phases=(-0.25,) * (plan.padded_node_count - 1) + (0.0,),
        )
    law = product_formula_select_resource_law(plan)
    built = build_multiplexed_product_formula_select(
        plan,
        num_system_qubits=int(np.log2(problem.dimension)),
    )
    recursively_decomposed = built.decompose(reps=10)

    assert recursively_decomposed.count_ops().get("cx", 0) == law["select_basis_cx_count"]
    # Attribute CX gates by the built operations that own them. This catches
    # a cost moved between parity, complete-table rotations, and phases even
    # when the total stays fixed; no circuit simulation is needed.
    assert built.count_ops().get("cx", 0) == law["select_parity_network_basis_cx_count"]
    for operation_name, component in (
        ("ucrz", "select_multiplexed_rotation_basis_cx_count"),
        ("nwqlib_phase_diagonal", "select_control_diagonal_basis_cx_count"),
    ):
        counted = sum(
            instruction.operation.definition.decompose(reps=10).count_ops().get("cx", 0)
            for instruction in built.data
            if instruction.operation.name == operation_name
        )
        assert counted == law[component]
    assert law["select_branch_controlled_full_unitary_count"] == 0
    assert law["select_structured_unconditional_rotation_count"] == 0
    assert law["select_structured_controlled_rotation_count"] == 0
    assert law["select_max_structured_control_degree"] == 0
    assert law["rotation_synthesis_precision"] is None
    # Count slots from emitted operations, including no slots for an absent diagonal.
    assert law["arbitrary_rotation_count"] == sum(
        len(instruction.operation.params)
        if instruction.operation.name == "ucrz"
        else (1 << instruction.operation.num_qubits) - 1
        if instruction.operation.name == "nwqlib_phase_diagonal"
        else 0
        for instruction in built.data
    )



@pytest.mark.parametrize("phase", (0.0, 0.37, 1.0e-16, 5.0e-324))
def test_branch_controlled_phase_law_prices_only_nonzero_physical_branches(
    monkeypatch, phase
) -> None:
    from qiskit import QuantumCircuit, transpile
    from nwqlib.algorithms.lchs.native import _direct_controlled_branch
    from nwqlib.algorithms.lchs.select_synthesis import (
        _branch_controlled_product_formula_resource_law,
    )

    problem = _mixed_two_problem()
    data = generate_lchs_product_formula_select_plan(
        matrix=problem.A.dense_array(),
        final_time=problem.elapsed_time,
        method=_pf_options(
            tier="analytic", order=1, steps=1, epsilon=0.5, truncation_multiplier=1.0
        ),
    )
    # Three physical branches: cancellation, a full turn, and a phase added after
    # a full turn. Circuit assignment normalizes the identity phase first, so
    # even a subnormal coefficient phase survives the second assignment.
    # A nonzero padded address must not be charged by the branch-controlled path.
    count = data.plan.padded_node_count
    plan = replace(
        data.plan,
        physical_node_count=3,
        branch_to_node=(0, 1, 2) + (None,) * (count - 3),
        coefficient_phases=(0.25, 0.0, phase) + (0.5,) * (count - 3),
        identity_phases=(-0.25, 2 * np.pi, 2 * np.pi) + (0.0,) * (count - 3),
        branch_step_counts=(0,) * count,
    )
    controls = count.bit_length() - 1
    built = QuantumCircuit(controls + 1)
    for branch in range(3):
        node = QuantumCircuit(1)
        node.global_phase = plan.identity_phases[branch]
        node.global_phase += plan.coefficient_phases[branch]
        built.append(_direct_controlled_branch(node, controls, branch), range(controls + 1))
    actual_cx = transpile(built, **TRANSPILE_OPTIONS).count_ops().get("cx", 0)
    law = _branch_controlled_product_formula_resource_law(plan)
    # Inspect emitted sample multiplicities before resource transpilation; only
    # the tiny independently constructed physical branches need to be lowered.
    from nwqlib.algorithms.lchs.quantum_resources import _pf_samples

    payload = SimpleNamespace(
        select_data=replace(data, plan=plan), selected_select="branch_controlled"
    )
    samples = list(
        _pf_samples(
            payload,
            SimpleNamespace(num_control_qubits=controls, num_system_qubits=1),
            dict(max_qubits=13,max_bytes=64*1024**2,max_build_work=100_000_000,used_build_work=0),
        )
    )
    sampled_phases = sum(
        multiplicity for name, _, multiplicity in samples if name == "branch_phase"
    )
    assert (law["select_control_diagonal_basis_cx_count"], sampled_phases) == (
        actual_cx,
        int(phase != 0.0),
    )
    assert (actual_cx > 0) == (phase != 0.0)
    # The displayed branch-phase operation has a fixed name, not a circuit-N name.
    assert not any("circuit-" in item.operation.name
                   for name, circuit, _ in samples if name == "branch_phase" for item in circuit.data)



def test_qsp_control_projection_counts_single_qubit_and_global_phase_gates():
    from nwqlib.algorithms.lchs.compiled_selection import _census_cx, _controlled_kind_cx
    from qiskit import QuantumCircuit, transpile
    from nwqlib.subroutines.qiskit_compat import controlled

    circuit = QuantumCircuit(3)
    circuit.ry(0.2, 0)
    circuit.ry(0.3, 0)
    circuit.cx(1, 0)
    circuit.cx(1, 0)
    circuit.x(2)
    query = QuantumCircuit(4)
    query.append(controlled(circuit.to_gate(), 1), range(4))
    actual = int(transpile(query, **TRANSPILE_OPTIONS).count_ops()["cx"])
    census = {"rotation": 2, "cx": 2, "h_or_x": 1}
    # This elementary circuit has two Toffolis, two CRY gates and one CX after control.
    assert _census_cx(census, _controlled_kind_cx(1)) == actual
    phase = QuantumCircuit(1)
    phase.global_phase = 0.23
    controlled_phase = QuantumCircuit(3)
    controlled_phase.append(controlled(phase.to_gate(), 2), range(3))
    actual_phase = int(transpile(controlled_phase, **TRANSPILE_OPTIONS).count_ops()["cx"])
    assert _controlled_kind_cx(2)["global_phase"] == actual_phase


# Width, builder and applicable dense-synthesis price of each census gate.
# A None builder gives a circuit that holds only a global phase.
_CENSUS_KIND_GATES = {
    "cx": ((2, lambda circuit: circuit.cx(0, 1), "cx"),),
    "h_or_x": ((1, lambda circuit: circuit.x(0), None), (1, lambda circuit: circuit.y(0), None),
               (1, lambda circuit: circuit.z(0), None), (1, lambda circuit: circuit.h(0), "h")),
    "rotation": ((1, lambda circuit: circuit.ry(.37, 0), None), (1, lambda circuit: circuit.rz(.37, 0), "rz"),
                 (1, lambda circuit: circuit.rx(.37, 0), None)),
    "phase": ((1, lambda circuit: circuit.p(.37, 0), None), (1, lambda circuit: circuit.s(0), None),
              (1, lambda circuit: circuit.t(0), None)),
    "u": ((1, lambda circuit: circuit.u(.3, .7, -1.1, 0), "u"),),
    "global_phase": ((1, None, "global_phase"),),
}
# Census kind of each gate name in Qiskit's add_control basis.
_KIND_OF_GATE = {"cx": "cx", "x": "h_or_x", "y": "h_or_x", "z": "h_or_x", "h": "h_or_x", "ry": "rotation",
                 "rz": "rotation", "rx": "rotation", "p": "phase", "u": "u"}


def _one_gate_circuit(width, build, idle=0):
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(width + idle, global_phase=.8 if build is None else 0.)
    if build is not None:
        build(circuit)
    return circuit


def _lowered_cx(gate, *, used=False):
    """CX of one gate appended to fresh qubits, or to qubits that earlier gates already used."""
    from qiskit import QuantumCircuit, transpile
    circuit = QuantumCircuit(gate.num_qubits)
    if used:
        for qubit in circuit.qubits:
            circuit.u(.1, .2, .3, qubit)
    circuit.append(gate, circuit.qubits)
    return transpile(circuit, **TRANSPILE_OPTIONS).count_ops().get("cx", 0)


def _unrolled_kind_census(gate):
    """Count the gates of ``gate`` by census kind after Qiskit unrolls it to its add_control basis."""
    from qiskit.circuit._add_control import EFFICIENTLY_CONTROLLED_GATES, _unroll_gate
    census = {}
    for name, count in _unroll_gate(gate, EFFICIENTLY_CONTROLLED_GATES).definition.count_ops().items():
        census[_KIND_OF_GATE[name]] = census.get(_KIND_OF_GATE[name], 0) + count
    return census


@pytest.mark.parametrize("controls", range(1, 9))
def test_controlled_census_prices_bound_qiskit_for_every_kind(controls):
    """Each census kind's price is Qiskit's CX for the costliest gate of the kind, and bounds the others.

    The QSP children and the source preparations are priced per census kind
    at the number of controls Qiskit adds at once. A price below one gate of
    its kind lets the total fall below the built circuit, for example a
    Gray-code price of 4 and 8 CX for RY with two and three controls, which
    Qiskit builds with 8 and 20. SX, which the product-formula branches use
    for Y factors, is priced as a phase. Idle qubits, clean or already used,
    may serve as ancillas and must not raise a count.
    """
    from math import pi
    from nwqlib.algorithms.lchs.compiled_selection import _controlled_kind_cx
    from nwqlib.algorithms.lchs.native import _controlled_gate_cx
    from nwqlib.subroutines.qiskit_compat import controlled
    gates = {**_CENSUS_KIND_GATES, "phase": (*_CENSUS_KIND_GATES["phase"], (1, lambda circuit: circuit.sx(0), None))}
    prices = _controlled_kind_cx(controls)
    dense_prices = _controlled_gate_cx(controls)
    contexts = ((0, False), (1, False), (1, True), (3, False), (3, True))
    for kind, members in gates.items():
        price = prices[kind]
        free = []
        for width, build, dense_kind in members:
            for idle, used in contexts:
                gate = controlled(_one_gate_circuit(width, build, idle).to_gate(), controls, ctrl_state=0)
                cx = _lowered_cx(gate, used=used)
                assert cx <= price, (kind, idle, used, cx, price)
                if dense_kind is not None:
                    assert cx <= dense_prices[dense_kind], (dense_kind, idle, used)
                    if idle == 0:
                        assert cx == dense_prices[dense_kind], dense_kind
                if idle == 0:
                    free.append(cx)
        assert max(free) == price, kind
    # Qiskit's special U branches must also fit the generic U price.
    for angles in ((0., 0., .9), (.4, 0., 0.), (.4, -pi/2, pi/2)):
        for idle, used in contexts:
            circuit = _one_gate_circuit(1, lambda c: c.u(*angles, 0), idle)
            gate = controlled(circuit.to_gate(), controls, ctrl_state=0)
            assert _lowered_cx(gate, used=used) <= dense_prices["u"], (angles, idle, used)


def test_twice_controlled_kind_costs_match_qiskit():
    """A gate that Qiskit controls once, and then once more, costs the one-control price of its one-control census.

    The compiled QSP SELECT controls a generator branch on the combine qubit
    and the pass that holds it on the parity qubit. Pricing that as two
    controls at once charges a CX 14 CX where the build takes 52.
    """
    from qiskit import QuantumCircuit
    from nwqlib.algorithms.lchs.compiled_selection import _ONE_CONTROL_UNROLLED_CENSUS, _twice_controlled_kind_cx
    from nwqlib.subroutines.qiskit_compat import controlled
    price = _twice_controlled_kind_cx()
    for kind, members in _CENSUS_KIND_GATES.items():
        built = []
        for width, build, _dense_kind in members:
            inner = controlled(_one_gate_circuit(width, build).to_gate(), 1)
            middle = QuantumCircuit(inner.num_qubits)
            middle.append(inner, middle.qubits)
            built.append((_lowered_cx(controlled(middle.to_gate(), 1, ctrl_state=0)), _unrolled_kind_census(inner)))
        assert all(cx <= price[kind] for cx, _ in built), kind
        assert max(built, key=lambda item: item[0]) == (price[kind], _ONE_CONTROL_UNROLLED_CENSUS[kind])


def test_qsp_projector_flip_table_matches_installed_qiskit():
    """Each entry is the one-control price of the flip's unrolled definition, and small ones match the build."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XGate
    from nwqlib.algorithms.lchs.compiled_selection import LCHS_QSP_PROJECTOR_FLIP_CX, _controlled_kind_cx
    from nwqlib.subroutines.qiskit_compat import controlled
    once = _controlled_kind_cx(1)
    for flip_controls, recorded in enumerate(LCHS_QSP_PROJECTOR_FLIP_CX, start=1):
        flip = QuantumCircuit(flip_controls + 1)
        flip.append(controlled(XGate(), flip_controls, ctrl_state=0), flip.qubits)
        census = _unrolled_kind_census(flip.to_gate())
        assert sum(count * once[kind] for kind, count in census.items()) == recorded, flip_controls
        if flip_controls <= 5:
            assert _lowered_cx(controlled(flip.to_gate(), 1, ctrl_state=0)) == recorded


def _pauli_sum(terms):
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list(terms).to_matrix()


# Generators whose QSP children cover every child kind: a dense dilation
# alone, a dense dilation with a banded child, banded children with an
# address PREP, and three-qubit Pauli children on three address qubits
# whose projector flips have five controls.
_QSP_PROBLEMS = {
    "dense_one_child": lambda: LinearDynamics(A=np.diag([.2, .3]), initial_state=[1, 0], time=.1),
    "dense_and_banded": lambda: LinearDynamics(A=[[.3, .04j], [.04j, .5]], initial_state=[1, 1j], time=.1),
    "banded_children": lambda: LinearDynamics(
        A=.2*(2*np.eye(4)-np.roll(np.eye(4), 1, 1)-np.roll(np.eye(4), -1, 1))
        + .1j*(np.roll(np.eye(4), 1, 1)+np.roll(np.eye(4), -1, 1)), initial_state=np.eye(4)[0], time=.1),
    "pauli_children": lambda: LinearDynamics(
        A=np.eye(8) + _pauli_sum([("XXI", .2), ("ZIZ", .15), ("IYY", .1), ("ZZZ", .12), ("XIX", .08), ("IZI", .05), ("YIY", .1)])
        + 1j*_pauli_sum([("XYZ", .1), ("ZXI", .07), ("IIX", .05), ("YZY", .09), ("XIZ", .06)]),
        initial_state=np.ones(8), time=.1),
    "dense_and_banded_with_source": lambda: LinearDynamics(A=[[.3, .04j], [.04j, .5]], initial_state=[1, 1j],
                                                           source=[.2, .1j], time=.1),
}


def _qsp_plan(case, address_bits=1, route='auto'):
    from nwqlib import plan
    problem = _QSP_PROBLEMS[case]()
    method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', approximation_tolerance=.3,
                  dense_control_route=route,
                  k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                              parameters={'num_qubits': address_bits, 'lsb_position': -1}),
                  **({'duhamel_nodes': 1} if problem.source is not None else {}))
    return problem, plan(problem, method=method)


@pytest.mark.parametrize("case", sorted(_QSP_PROBLEMS))
def test_qsp_child_census_bounds_the_unrolled_child(case):
    """Every census kind bounds the matching gates of the child circuit that the SELECT controls.

    The SELECT controls the exact dense synthesis of a dense dilation and
    the unrolled gates of a Pauli or banded LCU, whose one-qubit Pauli
    factors are U gates. Each kind is compared with the child PREP pair
    taken at its largest construction.
    """
    from nwqlib._preparation_laws import _uniform_superposition_gate_slots, direct_preparation_cx_bound
    from nwqlib.algorithms.lchs.compiled_selection import _qsp_child_gate_census
    from nwqlib.algorithms.lchs.native import _build_compiled_qsp_part_encoding
    from nwqlib.subroutines.qiskit_compat import _exact_dense_definitions
    _, selected = _qsp_plan(case)
    for part in selected._native['native_data'].qsp_plan['part_plans'][:2]:
        if part.implementation == 'exact_zero':
            continue
        gate = _exact_dense_definitions(_build_compiled_qsp_part_encoding(part).circuit.to_gate(), {})
        built = _unrolled_kind_census(gate)
        census = dict(_qsp_child_gate_census(part))
        prep_qubits = census.pop("preparation_qubits", 0)
        uniform = _uniform_superposition_gate_slots(prep_qubits)
        prep = {"cx": max(direct_preparation_cx_bound(prep_qubits, complex_phases=False), uniform.get("cx", 0)),
                "rotation": max((1 << prep_qubits) - 1, uniform.get("ry", 0)), "phase": uniform.get("p", 0),
                "h_or_x": max(prep_qubits, uniform.get("h", 0) + uniform.get("x", 0))}
        for kind, count in built.items():
            assert count <= census.get(kind, 0) + 2*prep.get(kind, 0), (part.implementation, kind)


@pytest.mark.parametrize("route", ["gatewise", "whole_matrix"])
@pytest.mark.parametrize(("case", "address_bits"), [
    ("dense_one_child", 2), ("dense_and_banded", 1), ("banded_children", 1), ("pauli_children", 1),
    ("dense_and_banded_with_source", 1)])
def test_compiled_qsp_select_cx_law_bounds_the_built_select(case, address_bits, route):
    """The compiled QSP SELECT law is an upper bound on the transpiled built leaf.

    With two children the combine qubit and then the parity qubit control
    each generator gate. Pricing those as two controls at once puts the law
    below these builds by up to a factor 1.85. A one-child generator, whose
    gates are controlled once, is the case that such a price still bounds.
    The source case adds the kind_control input preparation. On the
    whole-matrix route the combine qubit controls a dense child through its
    controlled dilation, which the parity qubit controls once more.
    """
    from qiskit import transpile
    from nwqlib.algorithms.lchs.native import construct_select
    problem, selected = _qsp_plan(case, address_bits, route)
    record = next(item for item in selected.construction.selections if item.signature.name == 'lchs_select')
    law = next(item.value for item in record.resource_laws if not item.adjoint)
    built = construct_select(SimpleNamespace(_payload=(selected._native['native_data'], problem.elapsed_time)), (), None)
    assert transpile(built, **TRANSPILE_OPTIONS).count_ops().get('cx', 0) <= law


def _xx_generator_problem():
    # Pauli labels IZ, ZI, ZZ and XX: two-qubit labels with X factors, whose
    # parity CX and H gates cost more controlled than the Gray-code model.
    matrix = np.array([[.5, 0, 0, .2], [0, .4, .2, 0], [0, .2, .45, 0], [.2, 0, 0, .35]])
    return LinearDynamics(A=matrix + .1j*np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]), initial_state=np.ones(4), time=.3)


@pytest.mark.parametrize(("problem_factory", "address_bits", "order", "exact"), [
    (_xx_generator_problem, 3, 1, False), (_xx_generator_problem, 4, 1, False), (_xx_generator_problem, 3, 2, False),
    (_mixed_two_problem, 4, 1, True), (_mixed_two_problem, 3, 2, True)])
def test_branch_controlled_product_formula_law_bounds_the_built_select(problem_factory, address_bits, order, exact):
    """The branch-controlled product-formula law is an upper bound on the transpiled built leaf.

    Gray-code and phase-polynomial prices for the controlled RZ, H and
    parity CX give 0.88 and 0.84 of the build at three and four address
    bits for two-qubit X labels. One-qubit labels are single controlled
    rotations, where the law equals the build.
    """
    from qiskit import transpile
    from nwqlib import plan
    from nwqlib.algorithms.lchs.native import construct_select
    problem = problem_factory()
    grid = ProviderConfig(implementation='signed_binary_uniform', parameters={'num_qubits': address_bits, 'lsb_position': -1})
    selected = plan(problem, method=LCHS(hamiltonian_evolution_backend='trotter', trotter_steps=order, trotter_order=order,
                                         lcu_select_implementation='branch_controlled', k_quadrature=grid))
    record = next(item for item in selected.construction.selections if item.signature.name == 'lchs_select')
    law = next(item.value for item in record.resource_laws if not item.adjoint)
    built = construct_select(SimpleNamespace(_payload=(selected._native['native_data'], problem.elapsed_time)), (), None)
    actual = transpile(built, **TRANSPILE_OPTIONS).count_ops().get('cx', 0)
    assert actual <= law
    if exact:
        assert actual == law


@pytest.mark.parametrize("controls", (1, 2, 3, 4, 8))
def test_controlled_source_preparation_bounds_every_construction(controls):
    """The controlled source-input price bounds each construction the direct preparation builder takes.

    A basis state, equal amplitudes, a uniform prefix and a general complex
    state take the X, H, UniformSuperpositionGate and tree-and-diagonal
    constructions.
    """
    from nwqlib.algorithms.lchs.compiled_selection import _controlled_direct_preparation_cx
    from nwqlib.subroutines.qiskit_compat import controlled
    from nwqlib.subroutines.state_preparation.direct import _build_normalized_state_preparation
    rng = np.random.default_rng(3)
    for num_qubits in (1, 2, 3):
        dimension = 1 << num_qubits
        price = _controlled_direct_preparation_cx(num_qubits, controls)
        general = rng.normal(size=dimension) + 1j*rng.normal(size=dimension)
        states = [np.eye(dimension)[-1] * 1j, np.ones(dimension), general]
        if dimension > 2:
            states.append(np.r_[np.ones(dimension - 1), 0.])
        for state in states:
            state = np.asarray(state, dtype=complex) / np.linalg.norm(state)
            circuit = _build_normalized_state_preparation(state, input_norm=1., register_name="system").circuit
            assert _lowered_cx(controlled(circuit.to_gate(), controls, ctrl_state=0)) <= price



@pytest.mark.parametrize('backend',['dense_exact','trotter','qsp_block_encoding'])
def test_explicit_samples_reuse_archived_selection_and_preserve_scope(backend,tmp_path,monkeypatch):
    from nwqlib import plan
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    from nwqlib._choice_archive import ArchiveFiles,save_plan,load_plan
    from nwqlib.subroutines.qsp import evolution
    from nwqlib.subroutines.state_preparation import mps
    chosen=plan(LinearDynamics(A=[[.3,.04j],[.04j,.5]],initial_state=[1,1j],time=.1),
        method=LCHS(hamiltonian_evolution_backend=backend,
            k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})))
    assert sum(r.width for r in chosen.construction.program.registers)<=10
    files=ArchiveFiles(tmp_path,8_000_000)
    saved=save_plan(chosen,files)
    def forbidden(*args,**kwargs):
        pytest.fail('resource inspection repeated numerical selection')
    monkeypatch.setattr(evolution,'prepare_qsp_evolution',forbidden)
    monkeypatch.setattr(mps,'_decompose_normalized_state',forbidden)
    monkeypatch.setattr(np.linalg,'eigvalsh',forbidden)
    loaded=load_plan(saved,files)
    assert loaded._native['native_data'].method is loaded.method
    values=sample_resources(loaded,max_qubits=10)
    settings={setting.name for setting in loaded.reconstruction.settings}
    assert set(values['weighted_totals'])==settings
    assert {row['name'].removeprefix('readout_') for row in values['representatives']
        if row['name'].startswith('readout_')}==settings
    select=loaded._native['native_data'].selected_select
    for row in values['representatives']:
        inventory=row['inventory']
        assert type(row['multiplicity']) is int and row['multiplicity']>=0
        assert inventory['compiler'] is None and inventory['basis']=='selected_sample_raw'
        assert inventory['circuit'].startswith(f"LCHS {row['name']} representative (SELECT={select})")
        assert inventory['total_operations']==sum(inventory['operations'].values())
    assert all(totals['total_operations']==sum(totals['operations'].values())>0
        for totals in values['weighted_totals'].values())
    # Representative sub-circuits carry fixed names, not process-dependent circuit-N names.
    assert not any('circuit-' in name for totals in values['weighted_totals'].values()
        for name in totals['operations'])
    with pytest.raises(ValueError,match='max_qubits'):
        sample_resources(loaded,max_qubits=1)
    # An equal-looking detached numerical payload is not the actual selected
    # leaf population; reject before any sample circuit is built.
    loaded._native['native_data']=replace(loaded._native['native_data'])
    with pytest.raises(ValueError,match='exact SELECT'):
        sample_resources(loaded,max_qubits=10)


def test_actual_representative_operation_limit_allows_equality(monkeypatch):
    from nwqlib import plan
    from nwqlib.algorithms.lchs import quantum_resources
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    chosen=plan(LinearDynamics(A=np.diag([.2,.3]),initial_state=[1,0],time=.1),
        method=LCHS(k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})))
    assert sum(register.width for register in chosen.construction.program.registers)==3
    values=sample_resources(chosen,max_qubits=3)
    representatives=values['representatives']
    actual=sum(row['inventory']['total_operations'] for row in representatives)
    admitted=sample_resources(chosen,max_qubits=3,max_operations=actual)
    assert admitted==values
    with pytest.raises(ValueError,match='max_operations'):
        sample_resources(chosen,max_qubits=3,max_operations=actual-1)
    # Dense SELECT is reported by one structural bound record for its 2**2
    # physical branches and adds nothing to the raw inventory. PREP and
    # readout representatives count once.
    (bound,)=values['structural_bounds']
    assert bound['multiplicity']==4 and bound['bound']['count_kind']=='structural_upper_bound'
    others=representatives
    assert all(row['multiplicity']==1 for row in others)
    # |0> needs no initial PREP gate and the exact readout measures nothing, yet
    # both inventories are present.
    assert {row['name'] for row in others if row['inventory']['operations']=={}}=={
        'initial_prep','readout_setting_0'}
    expected={}
    for row in representatives:
        for name,count in row['inventory']['operations'].items():
            expected[name]=expected.get(name,0)+row['multiplicity']*count
    others_total=sum(row['inventory']['total_operations'] for row in others)
    assert values['weighted_totals']=={'setting_0':dict(operations=expected,
        total_operations=others_total)}
    # A setting whose readout representative is missing has no weighted total.
    built=quantum_resources._representatives
    monkeypatch.setattr(quantum_resources,'_representatives',lambda *args:(
        row for row in built(*args) if not row[0].startswith('readout_')))
    with pytest.raises(ValueError,match='no readout inventory for setting setting_0'):
        sample_resources(chosen,max_qubits=3)


def test_readout_counts_join_only_their_own_setting():
    from nwqlib import plan,NormalizedExpectation
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    # Z+X/2 on the one system qubit needs an X and a Z setting. Each readout
    # measures the two success bits and the system bit once; X first applies H.
    chosen=plan(LinearDynamics(A=np.diag([.2,.3]),initial_state=[1,0],time=.1),
        method=LCHS(k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})),
        output=NormalizedExpectation(observable=[[1,.5],[.5,-1]]),shots=64)
    rec=chosen.reconstruction
    labels={setting.name:setting.label for setting in rec.settings}
    assert sorted(labels.values())==['X','Z'] and len(rec.success_bits)+len(rec.system_bits)==3
    values=sample_resources(chosen,max_qubits=3)
    readout={row['name'].removeprefix('readout_'):row['inventory']['operations']
        for row in values['representatives'] if row['name'].startswith('readout_')}
    assert {labels[name]:operations for name,operations in readout.items()}=={'X':{'h':1,'measure':3},'Z':{'measure':3}}
    core={}
    for row in values['representatives']:
        if not row['name'].startswith('readout_'):
            for name,count in row['inventory']['operations'].items():
                core[name]=core.get(name,0)+row['multiplicity']*count
    for setting,label in labels.items():
        totals=values['weighted_totals'][setting]
        assert totals['operations']['measure']==core.get('measure',0)+3
        assert totals['operations'].get('h',0)==core.get('h',0)+(label=='X')
        assert totals['total_operations']==sum(core.values())+3+(label=='X')


def test_branch_controlled_trotter_sample_names_are_fixed():
    from nwqlib import plan
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    chosen=plan(LinearDynamics(A=[[.3,.04j],[.04j,.5]],initial_state=[1,1j],time=.1),
        method=LCHS(hamiltonian_evolution_backend='trotter',lcu_select_implementation='branch_controlled',
            k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})))
    first,second=(sample_resources(chosen,max_qubits=10) for _ in range(2))
    assert first==second
    names={name for totals in first['weighted_totals'].values() for name in totals['operations']}
    assert not any('circuit-' in name for name in names)
    # The Hermitian part gives X and the diagonal dissipative part gives Z, so
    # each Pauli rotation class appears under one fixed label.
    assert sorted(name.rsplit('pauli_rotation_',1)[1] for name in names if 'pauli_rotation_' in name)==['X','Z']


def _dense_select_problem(q, source=False, seed=17):
    # A PSD Hermitian part and a nonzero anti-Hermitian part, so every k-node
    # branch is a generic dense unitary rather than the identity.
    rng=np.random.default_rng(seed+q)
    d=2**q
    m=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    g=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    A=.3*(m@m.conj().T)/d+.2j*(g+g.conj().T)/d
    return LinearDynamics(A=A,initial_state=rng.normal(size=d)+0j,time=.3,
        source=rng.normal(size=d)+0j if source else None)


@pytest.mark.parametrize('route',('gatewise','whole_matrix'))
@pytest.mark.parametrize('q',(1,2,3,4))
@pytest.mark.parametrize('a',(1,2,3))
def test_dense_exact_select_cx_law_bounds_the_built_select(q,a,route):
    """The dense_exact SELECT law is an upper bound on the transpiled built leaf.

    On the gate-wise route Qiskit adds the address controls gate by gate to
    the exact synthesis of each branch, so the law prices the synthesis gate
    census at the actual control count. For one system qubit, and for two
    with one address bit, a generic branch reaches the bound exactly. On the
    whole-matrix route a branch on two or more system qubits is one
    controlled synthesis on q + a qubits, at most
    (25/96) 4**(q+a) - 2**(q+a) + 4/3 CX, whose work exceeds the default
    max_select_work at q = 4, a = 3.
    """
    from qiskit import transpile
    from nwqlib import plan
    from nwqlib.algorithms.lchs.native import construct_select
    grid=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':a,'lsb_position':-1})
    problem=_dense_select_problem(q)
    selected=plan(problem,method=LCHS(hamiltonian_evolution_backend='dense_exact',k_quadrature=grid,
                                      dense_control_route=route,max_select_work=10**9))
    record=next(item for item in selected.construction.selections if item.signature.name=='lchs_select')
    law=next(item.value for item in record.resource_laws if not item.adjoint)
    built=construct_select(SimpleNamespace(_payload=(selected._native['native_data'],problem.elapsed_time)),(),None)
    counts=transpile(built,**TRANSPILE_OPTIONS).count_ops()
    actual=counts.get('cx',0)
    assert actual<=law
    physical=selected.reconstruction.physical_branches
    # The sampled structural record prices every physical branch with the
    # same value-independent owner bound, in the CX/U basis of this built leaf.
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    (bound,)=sample_resources(selected,max_build_work=10**9)['structural_bounds']
    assert bound['multiplicity']==physical and bound['basis_cx_upper_bound_total']==law
    if bound['bound']['basis_u_upper_bound'] is not None:
        assert counts.get('u',0)<=physical*bound['bound']['basis_u_upper_bound']
    if route=='whole_matrix' and q>=2:
        m=q+a
        assert law==physical*((25*4**m-96*2**m+128)//96)
        return
    if q==1 or (q,a)==(2,1):
        assert actual==law
    from nwqlib.algorithms.lchs.native import dense_branch_select_cx
    assert law==physical*dense_branch_select_cx(q,a)


@pytest.mark.parametrize('route',('gatewise','whole_matrix'))
def test_dense_exact_select_cx_law_bounds_a_source_layout(route):
    """The dense_exact SELECT law bounds a constant-source layout on both routes.

    The gate-wise route controls each branch's input preparation and dense
    unitary. The whole-matrix route synthesizes one controlled matrix per
    branch, its evolution times the preparation's matrix.
    """
    from qiskit import transpile
    from nwqlib import plan
    from nwqlib.algorithms.lchs.native import construct_select
    grid=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})
    problem=_dense_select_problem(3,source=True)
    selected=plan(problem,method=LCHS(hamiltonian_evolution_backend='dense_exact',k_quadrature=grid,duhamel_nodes=1,
                                      dense_control_route=route))
    record=next(item for item in selected.construction.selections if item.signature.name=='lchs_select')
    law=next(item.value for item in record.resource_laws if not item.adjoint)
    built=construct_select(SimpleNamespace(_payload=(selected._native['native_data'],problem.elapsed_time)),(),None)
    assert transpile(built,**TRANSPILE_OPTIONS).count_ops().get('cx',0)<=law
    if route=='whole_matrix':
        # The preparation is inside each branch's whole-matrix synthesis.
        m=3+selected.reconstruction.padded_branches.bit_length()-1
        assert law==selected.reconstruction.physical_branches*((25*4**m-96*2**m+128)//96)


def test_dense_synthesis_census_bounds_every_emitted_gate_kind():
    from scipy.linalg import expm
    from scipy.stats import unitary_group
    from nwqlib.subroutines._dense_synthesis import dense_synthesis_gate_census, dense_unitary_circuit
    rng=np.random.default_rng(11)
    for q in (2,3,4):
        d=2**q
        g=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
        u0,u1=unitary_group.rvs(d//2,random_state=rng),unitary_group.rvs(d//2,random_state=rng)
        cases=[unitary_group.rvs(d,random_state=rng),expm(-1e-4j*(g+g.conj().T)),
               np.block([[u0,np.zeros_like(u0)],[np.zeros_like(u0),u1]]),np.eye(d)[rng.permutation(d)]]
        bound=dense_synthesis_gate_census(q)
        for unitary in cases:
            counts=dense_unitary_circuit(unitary).count_ops()
            assert set(counts)<={'cx','u','rz','h'}
            assert all(counts.get(name,0)<=bound[name] for name in bound)
    # The generic Haar case reaches the RZ and H bounds, which have no A.2 saving.
    counts=dense_unitary_circuit(unitary_group.rvs(16,random_state=rng)).count_ops()
    assert (counts['rz'],counts['h'])==(dense_synthesis_gate_census(4)['rz'],dense_synthesis_gate_census(4)['h'])


@pytest.mark.parametrize(('q','a','source','route'),[(2,2,False,'auto'),(3,1,True,'auto'),(1,2,False,'auto'),
                                                  (2,1,False,'auto'),(3,1,True,'whole_matrix')])
def test_dense_exact_select_charges_the_syntheses_it_makes(monkeypatch,q,a,source,route):
    """Planning charges every exact branch synthesis of the built dense SELECT and Qiskit's control of it.

    Controlling a branch on its address bits synthesizes the branch's q-qubit
    unitary, once per physical branch, and Qiskit then controls every
    synthesized gate with the a address bits. A one-qubit branch keeps
    Qiskit's exact definition, one U gate, which Qiskit controls too. The
    construction work is the spectral branch law
    (time_independent_terms._spectral_branch_requirements) of the node
    eigensystems and branch products the build computes, recorded here,
    plus the control law per physical branch and the synthesis law of each
    call counted during the build, and
    max_select_work one unit below it refuses the Plan. On the whole-matrix
    route, which "auto" takes for one address bit, each branch on two or
    more system qubits is one whole-matrix synthesis on q + a qubits instead,
    and a constant-source branch first multiplies its evolution by the
    matrix of its input preparation, 16 D**3 units for each of the two
    input kinds and D**3 for each branch.
    """
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_synthesis_admission import _controlled_synthesis_work, _gatewise_control_work
    from nwqlib import plan
    from nwqlib.algorithms.lchs import time_independent_terms
    from nwqlib.algorithms.lchs.native import construct_select
    grid=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':a,'lsb_position':-1})
    problem=_dense_select_problem(q,source=source)
    options=dict(hamiltonian_evolution_backend='dense_exact',k_quadrature=grid,dense_control_route=route,
                 **({'duhamel_nodes':1} if source else {}))
    selected=plan(problem,method=LCHS(**options))
    calls=_counted_syntheses(monkeypatch)
    eigensystems,exponentials=[],[]
    eigensystem,branch=time_independent_terms._node_eigensystem,time_independent_terms._branch_unitary
    def recorded_eigensystem(k_value,l_part,h_part):
        eigensystems.append(k_value)
        return eigensystem(k_value,l_part,h_part)
    def recorded_branch(system,elapsed,dimension):
        exponentials.append(elapsed)
        return branch(system,elapsed,dimension)
    monkeypatch.setattr(time_independent_terms,'_node_eigensystem',recorded_eigensystem)
    monkeypatch.setattr(time_independent_terms,'_branch_unitary',recorded_branch)
    data=selected._native['native_data']
    construct_select(SimpleNamespace(_payload=(data,problem.elapsed_time)),(),None)
    physical=selected.reconstruction.physical_branches
    # A constant-source layout can add address bits for its input kinds.
    address=selected.reconstruction.padded_branches.bit_length()-1
    whole=q>=2 and (route=='whole_matrix' or address==1)
    assert calls==([q+address if whole else q]*physical if q>=2 else []) and len(exponentials)==physical
    d=1<<q
    # One eigensystem per distinct node, reused by every application of it.
    assert len(eigensystems)==len(set(eigensystems))
    law,_=time_independent_terms._spectral_branch_requirements(d,eigensystems=len(eigensystems),
                                                               branches=len(exponentials))
    if whole:
        work=law+physical*_controlled_synthesis_work(q+address)+(2*16*d**3+physical*d**3 if source else 0)
    else:
        work=law+physical*_gatewise_control_work(q,address)+sum(_dense_synthesis_work(width) for width in calls)
    record=next(item for item in selected.construction.selections if item.signature.name=='lchs_select')
    assert record.construction_work==work
    with pytest.raises(ValueError,match='max_select_work'):
        plan(problem,method=LCHS(**options,max_select_work=work-1))
    plan(problem,method=LCHS(**options,max_select_work=work))


@pytest.mark.parametrize(('case','address_bits','route'),[('dense_one_child',2,'auto'),('dense_and_banded',1,'gatewise'),
                                                         ('dense_and_banded',1,'auto'),('banded_children',1,'auto')])
def test_compiled_qsp_select_charges_the_syntheses_of_its_dense_children(monkeypatch,case,address_bits,route):
    """Planning charges the exact syntheses that the built compiled QSP SELECT makes, and Qiskit's control of them.

    A lone dense child is synthesized forward and adjoint by the parity
    control of each pass, which controls the synthesized gates at every
    query of the pass. A dense child beside a banded one is synthesized and
    controlled once, in its controlled generator branch, and the parity
    control of each pass controls that output again at every query. On
    the whole-matrix route, which "auto" takes for the combine qubit's one
    control, that child is one whole-matrix synthesis on its qubits and the
    control, and the parity control unrolls that circuit at every query.
    Banded children hold no dense unitary.
    """
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_synthesis_admission import (_controlled_census, _controlled_synthesis_work, _gatewise_control_work,
                                          _twice_controlled_work)
    from nwqlib.algorithms.lchs.native import construct_select
    problem,selected=_qsp_plan(case,address_bits,route)
    native=selected._native['native_data']
    calls=_counted_syntheses(monkeypatch)
    construct_select(SimpleNamespace(_payload=(native,problem.elapsed_time)),(),None)
    assert bool(calls)==case.startswith('dense')
    expansion=native.qsp_plan['expansion']
    queries=expansion.cos_degree+expansion.sin_degree
    work=sum(native.qsp_plan['structural_law'].values())
    if case=='dense_and_banded' and route=='auto':
        work+=_controlled_synthesis_work(calls[0])+_gatewise_control_work(0,1,queries,census=_controlled_census(calls[0]))
    else:
        work+=sum(_dense_synthesis_work(width) for width in calls)
        if case=='dense_one_child':
            work+=_gatewise_control_work(calls[0],1,queries)
        elif case=='dense_and_banded':
            work+=_gatewise_control_work(calls[0],1)+queries*_twice_controlled_work(calls[0])
    record=next(item for item in selected.construction.selections if item.signature.name=='lchs_select')
    assert record.construction_work==work
    if calls:
        from nwqlib.algorithms.lchs.compiled_selection import (compiled_select_controlled_syntheses,
                                                               compiled_select_dense_syntheses)
        assert sorted(calls)==sorted(compiled_select_dense_syntheses(native.qsp_plan,route=route)
                                     +compiled_select_controlled_syntheses(native.qsp_plan,route=route))


@pytest.mark.parametrize(('case','route'),[('dense_one_child','auto'),('dense_and_banded','auto'),
                                          ('banded_children','auto')])
def test_representative_sampling_charges_its_dense_syntheses(monkeypatch,case,route):
    """Explicit sampling adds the synthesis and control laws of each controlled dense QSP child to its build work.

    Each law of the route inflated past max_build_work refuses the dense
    children before they are built and leaves the banded QSP children
    unaffected. On the whole-matrix route, which "auto" takes for one
    control, a dense child beside another is charged the whole-matrix
    synthesis law instead.
    """
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    from nwqlib.subroutines import _dense_synthesis
    _,selected=_qsp_plan(case,route=route)
    sample_resources(selected)
    whole=route=='auto' and case!='dense_one_child'
    for law in ('controlled_synthesis_size',) if whole else ('dense_synthesis_size','gatewise_control_size'):
        with monkeypatch.context() as patched:
            patched.setattr(_dense_synthesis,law,lambda *counts:(10**9,0,0),raising=False)
            if case=='banded_children':
                sample_resources(selected)
                continue
            with pytest.raises(ValueError,match='max_build_work'):
                sample_resources(selected)


def test_dense_representative_sampling_reports_its_structural_bound_without_building(monkeypatch):
    """A dense exact SELECT is reported by its value-independent owner bound, not by built representatives.

    Sampling forms no node eigensystem, branch unitary or dense synthesis
    (quantum_resources.dense_representative_envelope) and adds the bound to
    no raw inventory; the record names its basis, route and count kind.
    """
    from nwqlib import plan
    from nwqlib.algorithms.lchs import time_independent_terms
    from nwqlib.algorithms.lchs.native import dense_branch_select_cx
    from nwqlib.algorithms.lchs.quantum_resources import sample_resources
    from nwqlib.subroutines import _dense_synthesis
    grid=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':1,'lsb_position':-1})
    selected=plan(_dense_select_problem(2),method=LCHS(hamiltonian_evolution_backend='dense_exact',k_quadrature=grid))
    for name in ('_node_eigensystem','_branch_unitary'):
        monkeypatch.setattr(time_independent_terms,name,lambda *a,**k:pytest.fail('representative matrix formed'))
    monkeypatch.setattr(_dense_synthesis,'dense_unitary_circuit',lambda *a,**k:pytest.fail('representative synthesized'))
    values=sample_resources(selected)
    (record,)=values['structural_bounds']
    assert record['bound']['count_kind']=='structural_upper_bound'
    assert record['multiplicity']==selected.reconstruction.physical_branches==2
    assert record['basis_cx_upper_bound_total']==2*dense_branch_select_cx(2,1,selected.method.dense_control_route)
    assert not any(row['name'].startswith('dense_') for row in values['representatives'])
