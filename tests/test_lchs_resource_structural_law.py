"""Independent recursive SELECT laws and explicit saved-data resource samples."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS,DenseExact,ProviderConfig,QSPBlockEncoding,Trotter
from nwqlib.algorithms.lchs.select_synthesis import build_multiplexed_product_formula_select
from nwqlib.algorithms.lchs.time_independent_terms import generate_lchs_product_formula_select_plan
from conftest import lchs_dense_quadrature
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
        approx_tol=epsilon,
        k_quadrature=ProviderConfig(
            implementation="composite_gauss",
            parameters={"truncation_multiplier": truncation_multiplier},
        ),
        ham_evo=Trotter(order=order, steps=steps),
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
    quadrature, *parts = lchs_dense_quadrature(problem.A.dense_array(), problem.elapsed_time, options)
    data = generate_lchs_product_formula_select_plan(
        quadrature=quadrature,
        parts=tuple(parts),
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
    # H = -.3 I + .1 (S + S^-1) - .2 S^2 on four points: a paired band child
    # with a half shift and two negative self-inverse coefficients.
    "paired_band_half_shift_signs": lambda: LinearDynamics(
        A=.2*(2*np.eye(4)-np.roll(np.eye(4), 1, 1)-np.roll(np.eye(4), -1, 1))
        + 1j*(-.3*np.eye(4) + .1*(np.roll(np.eye(4), 1, 1)+np.roll(np.eye(4), -1, 1)) - .2*np.roll(np.eye(4), 2, 1)),
        initial_state=np.eye(4)[0], time=.1),
}


def _qsp_plan(case):
    from nwqlib import plan
    problem = _QSP_PROBLEMS[case]()
    method = LCHS(ham_evo='qsp_block_encoding', approx_tol=.3,
                  k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                              parameters={'num_qubits': 1, 'lsb_position': -1}),
                  **({'duhamel_nodes': 1} if problem.source is not None else {}))
    return problem, plan(problem, method=method)


@pytest.mark.parametrize("case", list(_QSP_PROBLEMS))
def test_coherent_lcu_cx_law_bounds_the_built_circuit(case):
    """The coherent LCU block's CX law bounds its transpiled built circuit, and is absent when a child cost is unknown.

    The law is the selected row's attempt CX: the factorized PREP and its
    inverse, the slot phases, the input preparation R, and the evolution
    priced by the generator's child laws and Qiskit's multi-controlled X
    counts without ancillas (coherent_selection, module docstring). A
    dense-dilation child is an opaque query whose controlled CX is unknown,
    so its Plan declares no law rather than an incomplete count. The source
    case adds the kind-controlled input preparations. Every case is at
    most ten qubits wide.
    """
    from qiskit import transpile
    from nwqlib.algorithms.lchs.native import construct_coherent_lcu
    problem, selected = _qsp_plan(case)
    width = sum(register.width for register in selected.construction.program.registers)
    assert width <= 10
    record = next(item for item in selected.construction.selections if item.signature.name == 'lchs_coherent_lcu')
    data = selected._native['native_data']
    opaque = any(child is not None and child.implementation == 'dense_dilation'
                 for name, child in data.qsp_plan.generator.components.items() if name in ('l_child', 'h_child'))
    laws = [item.value for item in record.resource_laws if not item.adjoint]
    assert (not laws) == opaque
    if laws:
        built = construct_coherent_lcu(SimpleNamespace(_payload=data), (), None)
        assert built.num_qubits == width
        assert transpile(built, **TRANSPILE_OPTIONS).count_ops().get('cx', 0) <= laws[0]


def _xx_generator_problem():
    # Pauli labels IZ, ZI, ZZ and XX: two-qubit labels with X factors, whose
    # parity CX and H gates cost more controlled than the Gray-code model.
    matrix = np.array([[.5, 0, 0, .2], [0, .4, .2, 0], [0, .2, .45, 0], [.2, 0, 0, .35]])
    return LinearDynamics(A=matrix + .1j*np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]]), initial_state=np.ones(4), time=.3)


def test_branch_controlled_product_formula_law_bounds_the_built_select():
    """The branch-controlled product-formula law is an upper bound on the transpiled built leaf.

    Gray-code and phase-polynomial prices for the controlled RZ, H and
    parity CX give 0.88 of the build at three address bits for two-qubit X
    labels.
    """
    from qiskit import transpile
    from nwqlib import plan
    from nwqlib.algorithms.lchs.native import construct_select
    problem = _xx_generator_problem()
    grid = ProviderConfig(implementation='signed_binary_uniform', parameters={'num_qubits': 3, 'lsb_position': -1})
    selected = plan(problem, method=LCHS(ham_evo=Trotter(steps=1, order=1),
                                         lcu_select='branch_controlled', k_quadrature=grid))
    record = next(item for item in selected.construction.selections if item.signature.name == 'lchs_select')
    law = next(item.value for item in record.resource_laws if not item.adjoint)
    built = construct_select(SimpleNamespace(_payload=(selected._native['native_data'], problem.elapsed_time)), (), None)
    actual = transpile(built, **TRANSPILE_OPTIONS).count_ops().get('cx', 0)
    assert actual <= law


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
    selected=plan(problem,method=LCHS(ham_evo=DenseExact(control_route=route),k_quadrature=grid,
                                      max_select_work=10**9))
    record=next(item for item in selected.construction.selections if item.signature.name=='lchs_select')
    law=next(item.value for item in record.resource_laws if not item.adjoint)
    built=construct_select(SimpleNamespace(_payload=(selected._native['native_data'],problem.elapsed_time)),(),None)
    counts=transpile(built,**TRANSPILE_OPTIONS).count_ops()
    actual=counts.get('cx',0)
    assert actual<=law
    physical=selected.reconstruction.physical_branches
    # The structural bound prices every physical branch with the same
    # value-independent owner bound, in the CX/U basis of this built leaf.
    from nwqlib.algorithms.lchs.quantum_resources import dense_representative_envelope
    bound=dense_representative_envelope(q,a,route)
    assert physical*bound['basis_cx_upper_bound']==law
    if bound['basis_u_upper_bound'] is not None:
        assert counts.get('u',0)<=physical*bound['basis_u_upper_bound']
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
    selected=plan(problem,method=LCHS(ham_evo=DenseExact(control_route=route),k_quadrature=grid,
                                      duhamel_nodes=1))
    record=next(item for item in selected.construction.selections if item.signature.name=='lchs_select')
    law=next(item.value for item in record.resource_laws if not item.adjoint)
    built=construct_select(SimpleNamespace(_payload=(selected._native['native_data'],problem.elapsed_time)),(),None)
    assert transpile(built,**TRANSPILE_OPTIONS).count_ops().get('cx',0)<=law
    if route=='whole_matrix':
        # The preparation is inside each branch's whole-matrix synthesis.
        m=3+selected.reconstruction.padded_branches.bit_length()-1
        assert law==selected.reconstruction.physical_branches*((25*4**m-96*2**m+128)//96)


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
    options=dict(ham_evo=DenseExact(control_route=route),k_quadrature=grid,
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


def test_dense_select_breakdown_reports_its_structural_bound_without_building(monkeypatch):
    """A dense exact SELECT's resource breakdown reports its value-independent CX bound and builds nothing.

    The breakdown forms no node eigensystem, branch unitary or dense
    synthesis. It gives the SELECT one call per shot with the record's CX
    law, the owner bound (quantum_resources.dense_representative_envelope)
    times the two physical branches.
    """
    from nwqlib import plan
    from nwqlib.algorithms.lchs import time_independent_terms
    from nwqlib.algorithms.lchs.native import dense_branch_select_cx
    from nwqlib.algorithms.lchs.quantum_resources import resource_breakdown
    from nwqlib.subroutines import _dense_synthesis
    grid=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':1,'lsb_position':-1})
    selected=plan(_dense_select_problem(2),method=LCHS(ham_evo='dense_exact',k_quadrature=grid))
    for name in ('_node_eigensystem','_branch_unitary'):
        monkeypatch.setattr(time_independent_terms,name,lambda *a,**k:pytest.fail('branch matrix formed'))
    monkeypatch.setattr(_dense_synthesis,'dense_unitary_circuit',lambda *a,**k:pytest.fail('branch synthesized'))
    for entry in resource_breakdown(selected):
        (select,)=[row for row in entry['blocks'] if row['block']=='lchs_select']
        (law,)=select['laws']
        assert select['calls']==1 and selected.reconstruction.physical_branches==2
        assert law.value==2*dense_branch_select_cx(2,1,selected.method.ham_evo.control_route)


def test_coherent_lcu_completes_each_dense_child_once(monkeypatch):
    """The emitter completes a dense-dilation child once for all of its queries, with
    one full SVD, and the construction work counts that completion, 9 d**3 for d = 2."""
    from dataclasses import replace
    from types import MappingProxyType
    from nwqlib.algorithms.lchs import parameters
    from nwqlib.algorithms.lchs.native import construct_coherent_lcu
    from nwqlib.subroutines.block_encoding.joint import _complete_dense_children, build_access_query
    _, selected = _qsp_plan("dense_one_child")
    data = selected._native['native_data']
    generator = data.qsp_plan.generator
    dense = [name for name, child in generator.components.items()
             if getattr(child, 'implementation', None) == 'dense_dilation']
    assert len(dense) == 1 and generator.components[dense[0]].system_qubits == 1
    original, full = np.linalg.svd, []

    def svd(matrix, *args, **kwargs):
        if kwargs.get('compute_uv', True):
            full.append(np.shape(matrix))
        return original(matrix, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, 'svd', svd)
        construct_coherent_lcu(SimpleNamespace(_payload=data), (), None)
        # The other full SVDs are the batched 4 x 4 ones of the two-qubit synthesis.
        assert full.count((2, 2)) == 1
        # This route queries one form. Three forms of the completed generator
        # still use one completion.
        forms = ({}, {'num_controls': 1}, {'num_controls': 1, 'adjoint': True})
        full.clear()
        source = _complete_dense_children(generator, max_bytes=None, max_work=None)
        for options in forms:
            build_access_query(source, **options)
        assert full.count((2, 2)) == 1
    # Without the synthesis and table-query terms, the dense child differs
    # from a non-dense stand-in by its completion alone.
    monkeypatch.setattr(parameters, '_dense_child_syntheses', lambda prepared: (0, 0))
    monkeypatch.setattr(parameters, '_table_queries', lambda prepared, max_bytes: (0, 0, None))
    stand_in = SimpleNamespace(implementation='banded', system_qubits=1)
    other = replace(data, qsp_plan=replace(data.qsp_plan, generator=replace(
        generator, components=MappingProxyType({**generator.components, dense[0]: stand_in}))))
    work = parameters._coherent_construction_work(data, method=data.method)[0]
    assert work - parameters._coherent_construction_work(other, method=data.method)[0] == 9 * 2**3


def test_coherent_planning_counts_the_table_queries_that_its_select_builds(monkeypatch):
    """Planning counts each compact-child and node-table query of the coherent SELECT at its built controls.

    On three qubits the gqet walk builds the singly controlled generator query, whose node table and two
    multiplexed-Pauli children are queried with two controls: 96 * 2**4 + 2 * 288 * 2**3 dominant table
    work. The construction work includes that work, and the Plan records it with the planning of the
    children as select_work. One unit less, which covers the construction alone, is refused at planning,
    with that total, and at the recorded total the Plan records no more than its limit and preparation
    builds the SELECT, admitting the same table queries that planning listed.
    """
    from collections import Counter
    from nwqlib import plan, prepare
    from nwqlib.algorithms.lchs import parameters
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.subroutines.block_encoding import core, nodes

    a = ingest_pauli([("III", 1.+.3j), ("XII", .25), ("IZI", .4j), ("IIY", -.1), ("ZZI", .05j)], num_qubits=3)
    problem = LinearDynamics(A=a, initial_state=ingest_occupation("000", num_qubits=3), time=.2)
    options = dict(ham_evo=QSPBlockEncoding(route="gqet"), duhamel_nodes=2,
                   lchs_kernel=ProviderConfig(implementation="cauchy_density"),
                   k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                               parameters={"num_qubits": 2, "lsb_position": -1}))
    admitted = []

    def observe(original):
        def admit(bits, unit_bytes, unit_work, max_bytes, max_work, operation):
            admitted.append((operation, bits, unit_work << bits))
            return original(bits, unit_bytes, unit_work, max_bytes, max_work, operation)
        return admit

    monkeypatch.setattr(core, "_admitted_power", observe(core._admitted_power))
    monkeypatch.setattr(nodes, "_admitted_power", observe(nodes._admitted_power))
    free = plan(problem, method=LCHS(**options))
    data, total = free._native["native_data"], free.reconstruction.select_work
    work = parameters._coherent_construction_work(data, method=data.method)[0]
    admitted.clear()
    parameters._table_queries(data.qsp_plan, max_bytes=data.method.max_bytes)
    planned = Counter(admitted)
    assert planned == Counter({("diagonal angle table", 4, 96 << 4): 1, ("block-encoding query tables", 3, 288 << 3): 2})
    assert total > work >= sum(value * count for (*_, value), count in planned.items())
    with pytest.raises(ValueError, match=rf"needs {total} work units of max_select_work, {work} to build it"):
        plan(problem, method=LCHS(**options, max_select_work=total - 1))
    capped = plan(problem, method=LCHS(**options, max_select_work=total))
    assert capped.reconstruction.select_work == total
    admitted.clear()
    with prepare(capped, progress=False).run:
        pass
    assert Counter(admitted) == planned
