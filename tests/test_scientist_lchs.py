"""Direct scientific LCHS inputs, numerical selection and circuit consumers."""

from dataclasses import replace

import numpy as np
import pytest

from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs.providers import resolve_lchs_coefficient_plan
from nwqlib.subroutines.state_preparation.mps import analyze_mps_state_compression


def test_coefficient_selection_uses_original_spectrum_and_encoded_zero(monkeypatch):
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.algorithms.lchs.time_independent_terms import generate_lchs_quadrature

    calls = []
    original = np.linalg.eigvalsh

    def record(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", record)
    # Original L=-I shifts to the PSD roundoff window; the extra zero
    # eigenvalue shifts to 1+window. Its encoded norm includes that dummy.
    matrix = -np.eye(3, dtype=complex)
    selected = generate_lchs_quadrature(matrix=matrix, final_time=.1,
        method=LCHS(approximation_tolerance=.01), padded_dimension=4)
    # The spectrum is solved once on the original 3x3 L, never on the padded
    # 4x4 matrix. A node count other than 3 or 4 keeps leggauss's own Jacobi
    # eigensolve distinguishable from either system eigensolve.
    assert selected.node_count not in (3, 4)
    assert calls.count((3, 3)) == 1 and (4, 4) not in calls
    shift = 1 + 1e-12
    assert selected.conversion["psd_shift"] == shift
    assert selected.l_norm == shift
    np.testing.assert_array_equal(selected.l_part, np.diag([shift-1]*3 + [shift]))
    np.testing.assert_array_equal(selected.h_part, np.zeros((4, 4)))


def test_coefficient_mps_targets_magnitudes_and_allows_identical_scaled_input(monkeypatch):
    problem = LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1, 0], time=.1)
    selected = resolve_lchs_coefficient_plan(problem)
    # Independent rank-one tensor G=(1,2,2,4)/5. Positive scaling changes
    # alpha, but leaves this exact numerical PREP tensor and its ranks fixed.
    coefficients = (1, 4j, -4, -16j)
    first = replace(selected, coefficients=coefficients, coefficient_l1_norm=25.)
    second = replace(first, coefficients=tuple(4*c for c in coefficients), coefficient_l1_norm=100.)
    np.testing.assert_array_equal(first.prep_amplitudes(), np.array([1, 2, 2, 4])/5)
    mps = first.decompose_mps(max_bond_dim=1)
    monkeypatch.setattr(np.linalg, "svd", lambda *a, **k: pytest.fail("unexpected second SVD"))
    assert second.decompose_mps(max_bond_dim=1, reuse=mps) is mps
    analysis = analyze_mps_state_compression(mps)
    assert analysis.estimated_raw_l2 < 1e-14
    reordered = replace(first, coefficients=(1, 4j, -16j, -4))
    with pytest.raises(ValueError, match="same normalized tensor"):
        reordered.decompose_mps(max_bond_dim=1, reuse=mps)


def test_coefficient_limits_precede_spectral_work(monkeypatch):
    problem = LinearDynamics(A=np.eye(3), initial_state=[1, 0, 0], time=.1)
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda *a, **k: pytest.fail("eigensolve before admission"))
    with pytest.raises(ValueError, match="bytes"):
        resolve_lchs_coefficient_plan(problem, max_bytes=16)


def test_source_phase_and_original_coordinates_reach_both_actual_consumers():
    from nwqlib import plan, solve
    from nwqlib.algorithms import LCHS
    # Complex scaling is physical; the unused padded coordinate is not part
    # of the solution. Equal initial/source handles still have distinct roles.
    diagonal = np.array([.1+.05j,.2-.04j,.3+.02j])
    initial = .5*np.exp(1j*np.pi/3)*np.array([1.,1j,-.25j])
    problem = LinearDynamics(A=np.diag(diagonal),initial_state=initial,source=initial,time=.1)
    # This witness tests source/phase/original-coordinate association, not
    # automatic quadrature accuracy. Keep its selected native width bounded.
    from nwqlib.algorithms.lchs import ProviderConfig
    method = LCHS(duhamel_nodes=2, k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits":2, "lsb_position":-1}))
    chosen = plan(problem,method=method)
    assert sum(w.width for w in chosen.construction.program.registers)<=8
    selected = chosen._native['native_data']
    quad = selected.quadrature
    # Independent diagonal exponentials and coherent Duhamel sum. No SELECT,
    # Pauli/PF builder or result field supplies the expected native action.
    def evolution(elapsed):
        return sum(c*np.exp(-1j*elapsed*(k*diagonal.real+diagonal.imag))
            for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
    expected = evolution(.1)*initial
    for node,weight in zip(chosen.reconstruction.source_nodes,chosen.reconstruction.source_weights,strict=True):
        expected += weight*evolution(.1-node)*initial
    quantum = solve(chosen)
    classical = solve(problem,method=method,execution='classical')
    assert quantum.solution.shape==classical.solution.shape==(3,)
    # <=9q native synthesis roundoff; no assertion about finite-grid accuracy.
    np.testing.assert_allclose(quantum.solution,expected,rtol=0,atol=2e-9)
    np.testing.assert_allclose(classical.solution,expected,rtol=0,atol=2e-13)
    arguments = [{item.parameter:item.value for item in app.arguments} for app in classical.applications]
    assert [row['source_application'] for row in arguments]==[0,1,1]
    for row,weight in zip(arguments,(1.,.05,.05),strict=True):
        assert row['weight'].value==pytest.approx(weight*np.sqrt(33)/8,rel=2e-15, abs=0)
        assert row['recovery_scale'].value==1.  # Positive L needs no PSD shift.


@pytest.mark.parametrize('backend,tolerance', [('dense_exact',1e-13),('trotter',1e-13),('qsp_block_encoding',2e-5)])
def test_exact_zero_l_removes_k_quadrature_and_keeps_physical_phase(backend,tolerance):
    from nwqlib import plan, solve
    from nwqlib.algorithms import LCHS
    problem = LinearDynamics(A=1j*np.diag([.2,.3]),initial_state=[1,1j],time=.1)
    selected = plan(problem,method=LCHS(hamiltonian_evolution_backend=backend))
    assert selected.reconstruction.mode=='unitary'
    assert selected.reconstruction.physical_branches==1
    assert selected.reconstruction.kernel_approximation_bound==selected.reconstruction.quadrature_bound==0
    assert sum(w.width for w in selected.construction.program.registers)<=13
    result = solve(selected)
    expected = np.exp(-.1j*np.array([.2,.3]))*np.array([1,1j])
    # QSP has its actual finite polynomial/phase approximation; direct/PF
    # commute on this diagonal H and have only native roundoff.
    np.testing.assert_allclose(result.solution,expected,rtol=0,atol=tolerance)


def test_zero_solution_needs_no_evolution(monkeypatch):
    from nwqlib import solve, NormSquared, StateVector
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import time_independent_terms
    monkeypatch.setattr(time_independent_terms,'generate_lchs_quadrature',
        lambda **kw: pytest.fail('algebraic zero selected an evolution'))
    zero = LinearDynamics(A=np.eye(3),initial_state=[0,0,0],source=[0,0,0],time=1)
    assert solve(zero,method=LCHS(),output=NormSquared()).value==0
    unit = solve(zero,method=LCHS(),output=StateVector(normalization='unit'))
    assert unit.unavailable is not None


def test_mps_archive_reuses_selected_cores_and_original_complex_result(tmp_path,monkeypatch):
    from nwqlib import plan, solve, load_result
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs.provider_config import ProviderConfig
    from nwqlib.subroutines.state_preparation import mps
    problem = LinearDynamics(A=np.diag([.2,.3]),initial_state=[1,1j],time=.1)
    method = LCHS(k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
        parameters={'num_qubits':2,'lsb_position':0}),lcu_state_preparation='mps_circuit',
        lcu_mps_max_bond_dim=1,mps_num_layers=1)
    selected = plan(problem,method=method)
    original = resolve_lchs_coefficient_plan(selected).mps_decomposition
    monkeypatch.setattr(mps,'_decompose_normalized_state',lambda *a,**kw: pytest.fail('selected TT-SVD repeated'))
    result = solve(selected)
    loaded = load_result(result.save(tmp_path/'mps'))
    restored = resolve_lchs_coefficient_plan(loaded.plan)
    assert restored.decompose_mps(max_bond_dim=1) is restored.mps_decomposition
    for before,after in zip(original.cores,restored.mps_decomposition.cores,strict=True):
        np.testing.assert_array_equal(before,after)
        assert not after.flags.writeable
    np.testing.assert_array_equal(result.solution,loaded.solution)
    assert loaded.plan==selected
    # Reporting consumes saved rank/layer/cost parameters, not another MPS
    # construction, simulation, or decomposition.
    monkeypatch.setattr('nwqlib.algorithms.lchs.native.construct_preparation',
        lambda *a,**kw: pytest.fail('report constructed an MPS circuit'))
    summary=loaded.report()['summary']
    assert 'MPS layers=1' in summary and 'max stored bond=1' in summary and 'SDK workspace unknown' in summary


def test_periodic_selected_repeat_matches_independent_strang_matrix():
    from nwqlib import plan, solve, NormSquared
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import ProviderConfig
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_occupation
    q,steps,elapsed,mass,diffusion,potential = 4,3,.1,.1,.1,.05
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=q,mass=mass,diffusion=diffusion,potential=potential),
        initial_state=ingest_occupation((0,)*q,num_qubits=q),time=elapsed)
    chosen = plan(problem,method=LCHS(hamiltonian_evolution_backend='trotter',
        lcu_select_implementation='structured',trotter_steps=steps,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':3,'lsb_position':-1})),output=NormSquared())
    coefficients = resolve_lchs_coefficient_plan(chosen)
    d = 1<<q
    identity = np.eye(d,dtype=complex)
    x = identity[np.arange(d)^1]
    shift = np.roll(identity,1,axis=0)  # S|j> = |j+1 mod d>.
    z = 1-2*(np.arange(d)&1)
    rz = np.diag(np.exp(-.5j*elapsed/steps*potential*z))
    expected = np.zeros(d,dtype=complex)
    for k,c in zip(coefficients.nodes,coefficients.coefficients,strict=True):
        angle = -elapsed/steps*k*diffusion
        half = np.cos(angle/2)*identity-1j*np.sin(angle/2)*x
        full = np.cos(angle)*identity-1j*np.sin(angle)*x
        step = rz@half@shift@full@shift.conj().T@half@rz
        expected += c*np.exp(-1j*elapsed*k*(mass+2*diffusion))*np.linalg.matrix_power(step,steps)[:,0]
    # This protects Strang action at each actual node; automatic quadrature
    # accuracy is independent. Eight explicit addresses keep native width seven.
    assert sum(w.width for w in chosen.construction.program.registers)==7
    result = solve(chosen)
    # Construction identity against analytic Pauli rotations/permutations,
    # independent of the native QFT/multiplexor emitters; not exact-PDE error.
    assert result.norm_squared==pytest.approx(float(np.vdot(expected,expected).real),rel=0,abs=2e-11)


def test_periodic_hundred_qubit_estimate_keeps_symbolic_system(monkeypatch):
    from nwqlib import plan, estimate, NormSquared
    from nwqlib.algorithms import LCHS
    from nwqlib.operators.inputs import OperatorInput, PeriodicStencil
    from nwqlib.problems.inputs import StateInput, ingest_occupation
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=100,mass=.1,diffusion=.1,potential=.05),
        initial_state=ingest_occupation((0,)*100,num_qubits=100),time=.1)
    def forbidden(*args,**kwargs):
        pytest.fail('symbolic periodic work materialized a system input')
    monkeypatch.setattr(OperatorInput,'dense_array',forbidden)
    monkeypatch.setattr(StateInput,'physical_vector',forbidden)
    chosen = plan(problem,method=LCHS(hamiltonian_evolution_backend='trotter',lcu_select_implementation='structured'),
        output=NormSquared())
    result = estimate(chosen)
    assert result.construction_id==chosen.construction.content_id
    assert len(chosen.construction.selections)==5


def test_dense_selected_branch_control_keeps_phase_adjoint_and_padding(monkeypatch):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from nwqlib.algorithms.lchs.native import build_branch_select
    from nwqlib.subroutines.lcu.core import prepare_lcu_gate_data
    from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate
    # Noncommuting complex branches with distinct physical phases; slot3 is
    # unused. A generic branch must synthesize its actual definition before
    # control instead of materializing each full controlled UnitaryGate.
    x = np.array([[0,1],[1,0]],dtype=complex)
    z = np.diag([1.,-1.])
    matrices = (np.cos(.3)*np.eye(2)-1j*np.sin(.3)*x,
        np.exp(.2j)*z, np.array([[1,1j],[1j,1]])/np.sqrt(2))
    coefficients = (1j,-2.,np.exp(-.4j))
    branches = []
    for value in matrices:
        circuit = QuantumCircuit(1)
        circuit.append(UnitaryGate(value), [0])
        branches.append(circuit)
    monkeypatch.setattr(UnitaryGate, 'control', lambda *a,**kw: pytest.fail('full controlled-matrix validation'))
    selected = build_branch_select(prepare_lcu_gate_data(coefficients,system_dimension=2),tuple(branches))
    assert selected.num_qubits == 3
    expected = np.eye(8,dtype=complex)
    for slot,(coefficient,value) in enumerate(zip(coefficients,matrices,strict=True)):
        indices = np.array([slot,slot+4])
        expected[np.ix_(indices,indices)] = coefficient/abs(coefficient)*value
    np.testing.assert_allclose(Operator(selected).data,expected,rtol=0,atol=2e-13)
    gate = selected.to_gate()
    np.testing.assert_allclose(Operator(inverse_realized_gate(gate)).data,expected.conj().T,rtol=0,atol=2e-13)
    outer = Operator(controlled(gate,1)).data
    np.testing.assert_allclose(outer[1::2,1::2],expected,rtol=0,atol=3e-13)
    np.testing.assert_allclose(outer[::2,::2],np.eye(8),rtol=0,atol=3e-13)


@pytest.mark.parametrize('num_qubits,time', [(2, 1e-6), (3, 1e-8), (2, 1.)])
def test_dense_select_branches_match_exact_controlled_exponentials(num_qubits, time):
    from scipy.linalg import expm
    from qiskit.quantum_info import Operator
    from nwqlib import plan
    from nwqlib.algorithms import LCHS
    from nwqlib.blocks.lowering import _local_method_context
    # Qiskit 2.5.2 synthesized a near-identity branch with entry errors of
    # about 1e-6 on two system qubits at time 1e-6 and 2e-10 to 1e-9 on three
    # at time 1e-8. Time 1 gives generic branches. The expected branch is
    # SciPy's expm of the saved Cartesian parts, independent of the branch
    # constructor. An eight-node grid keeps three address qubits. 1e-11 is
    # about 80 times the largest rounding error of these controlled
    # constructions, 1.2e-13.
    rng = np.random.default_rng(31 + num_qubits)
    d = 1 << num_qubits
    x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    y = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    matrix = x @ x.conj().T / d + .5j * (y + y.conj().T)
    initial = np.zeros(d)
    initial[0] = 1
    grid = {'implementation': 'signed_binary_uniform', 'parameters': {'num_qubits': 3, 'lsb_position': -1}}
    chosen = plan(LinearDynamics(A=matrix, initial_state=initial, time=time), method=LCHS(k_quadrature=grid))
    select = {block.record.signature.name: block for block in chosen.blocks}['lchs_select']
    data, elapsed = select._payload
    assert data.method.hamiltonian_evolution_backend == 'dense_exact'
    circuit = select._constructor(select, (), _local_method_context())
    nodes = np.asarray(data.quadrature.k_nodes)
    coefficients = np.asarray(data.coefficient_plan.coefficients)
    address = circuit.num_qubits - num_qubits
    assert address == 3
    for branch in range(len(nodes)):
        exact = expm(-1j * elapsed * (nodes[branch] * data.quadrature.l_part + data.quadrature.h_part))
        expected = np.eye(1 << circuit.num_qubits, dtype=complex)
        selected = [(row << address) | branch for row in range(d)]
        expected[np.ix_(selected, selected)] = coefficients[branch] / abs(coefficients[branch]) * exact
        actual = Operator(circuit.data[branch].operation).data
        assert np.abs(actual - expected).max() <= 1e-11, (branch, nodes[branch])


@pytest.mark.parametrize('source',[None,[0,0,0],[.1j,-.2,.3]])
def test_zero_qsp_generator_is_actual_identity_with_source_and_archive(source,tmp_path,monkeypatch):
    from nwqlib import plan, solve, load_result
    from nwqlib.algorithms import LCHS
    from nwqlib.subroutines.qsp import evolution
    monkeypatch.setattr(evolution,'prepare_qsp_evolution',lambda **kw: pytest.fail('zero generator fitted phases'))
    initial = np.array([1j,-.5,.25j])
    problem = LinearDynamics(A=np.zeros((3,3)),initial_state=initial,source=source,time=.3)
    chosen = plan(problem,method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding',duhamel_nodes=2))
    assert chosen.reconstruction.selected_select=='identity_evolution'
    assert chosen._native['native_data'].qsp_plan is None
    if source is not None:
        gauss=next(value.fact for value in chosen.facts if value.fact.quantity=='duhamel_quadrature')
        assert gauss.value.value==0  # Actual physical A=0: constant source integrand.
    assert sum(register.width for register in chosen.construction.program.registers)<=5
    result = solve(chosen)
    expected = initial if source is None else initial+.3*np.asarray(source)
    np.testing.assert_allclose(result.solution,expected,rtol=0,atol=3e-13)
    loaded = load_result(result.save(tmp_path/'identity'))
    assert loaded.plan._native['native_data'].method is loaded.plan.method
    np.testing.assert_array_equal(loaded.solution,result.solution)


def test_invalid_method_and_source_preparation_reject_before_selection(monkeypatch):
    from nwqlib import plan
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import selection
    monkeypatch.setattr(selection,'select_dense',lambda *a: pytest.fail('invalid input consumed numerical selection'))
    # Both product-formula backends take Lie order 1. The other backends apply
    # no product formula, so an order other than 2 is rejected there.
    for backend in ('dense_exact','qsp_block_encoding'):
        with pytest.raises(ValueError,match='use no product formula'):
            LCHS(trotter_order=1,hamiltonian_evolution_backend=backend)
    for backend in ('trotter','trotter_error_budgeted'):
        assert LCHS(trotter_order=np.int64(1),hamiltonian_evolution_backend=backend).trotter_order==1
    with pytest.raises(ValueError,match='selects its own step count'):
        LCHS(hamiltonian_evolution_backend='trotter_error_budgeted',trotter_steps=2)
    problem = LinearDynamics(A=np.eye(3),initial_state=[1,0,0],source=[0,1,0],time=.1)
    with pytest.raises(ValueError,match='direct PREP'):
        plan(problem,method=LCHS(lcu_state_preparation='mps_circuit'))


@pytest.mark.parametrize('backend,revised', [('dense_exact', False), ('qsp_block_encoding', True)])
def test_trotter_steps_without_product_formula_warns_at_caller_and_plans_the_default(backend, revised):
    import warnings
    from nwqlib import plan
    from nwqlib.algorithms import LCHS
    problem = LinearDynamics(A=[[.4,.15],[.05,.25]],initial_state=[1,0],time=.1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        base = LCHS(hamiltonian_evolution_backend=backend,
            k_quadrature={'implementation': 'signed_binary_uniform',
                          'parameters': {'num_qubits': 2, 'lsb_position': -1}})
        default_method = base.revise() if revised else base
        method = base.revise(trotter_steps=5) if revised else LCHS(
            **{**base.model_dump(exclude_computed_fields=True), 'trotter_steps': 5})
        default = plan(problem,method=default_method,seed=7)
    assert [str(item.message) for item in record] == []
    with pytest.warns(UserWarning,match="applies only to the 'trotter' backend") as record:
        ignored = plan(problem,method=method,seed=7)
    assert [item.filename for item in record] == [__file__]
    assert ignored.reconstruction == default.reconstruction
    assert ignored.method is method
    assert ignored.content_id == default.content_id
    assert ignored.construction.content_id == default.construction.content_id
    assert method.parent_id == default_method.parent_id
    # Ignored values cannot violate a step cap; copies and loads are canonical
    # records with no pending caller diagnostic.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert LCHS(hamiltonian_evolution_backend=backend,
                    trotter_steps=5, max_trotter_steps=2).trotter_steps == 1
        for canonical in (method.model_copy(), LCHS.model_validate_json(method.model_dump_json())):
            assert plan(problem, method=canonical, seed=7).content_id == default.content_id
    assert not record


@pytest.mark.parametrize('backend',['trotter','trotter_error_budgeted'])
def test_source_pf_padding_and_original_readout_follow_recorded_steps(backend):
    from nwqlib import plan, solve, StateVector, NormSquared, NormalizedExpectation, Samples
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs.provider_config import ProviderConfig
    from _lchs_suzuki_reference import sorted_pauli_suzuki_2_reference
    matrix = np.array([[.3,.04+.01j,0],[.04-.03j,.4,.02],[0,.02,.2]])
    initial,source = np.array([1j,-.2,.1]),np.array([.1,.2j,-.1j])
    problem = LinearDynamics(A=matrix,initial_state=initial,source=source,time=.2)
    # The budgeted backend selects its own steps and refuses a fixed count.
    fixed = {'trotter_steps':2} if backend=='trotter' else {}
    method = LCHS(hamiltonian_evolution_backend=backend,**fixed,duhamel_nodes=3,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen = plan(problem,method=method)
    data = chosen._native['native_data']
    layout,pf = data.source_layout,data.select_data.plan
    assert layout.branch_count==16 and len(layout.coefficients)==32
    assert any(slot is None for slot in layout.branch_to_node)
    assert sum(register.width for register in chosen.construction.program.registers)==7
    expected = np.zeros(4,dtype=complex)
    for slot,node in enumerate(layout.branch_to_node):
        if node is None:
            continue
        direction = data.initial_direction if layout.branch_kinds[node]=='initial' else data.source_direction
        action = sorted_pauli_suzuki_2_reference(data.quadrature.l_part,data.quadrature.h_part,
            k_value=layout.k_values[slot],final_time=layout.elapsed_times[slot],reps=pf.branch_step_counts[slot])
        expected += layout.coefficients[slot]*(action@direction)
    actual = solve(chosen)
    np.testing.assert_allclose(actual.solution,expected[:3],rtol=0,atol=3e-10)
    unit = solve(problem,method=method,output=StateVector(normalization='unit'))
    np.testing.assert_allclose(unit.state_vector,expected[:3]/np.linalg.norm(expected[:3]),rtol=0,atol=3e-10)
    mass = np.linalg.norm(expected[:3])**2
    observable = np.diag([1.,-2.,.5])
    exact = solve(problem,method=method,output=NormalizedExpectation(observable=observable))
    expected_value = float(np.vdot(expected[:3],observable@expected[:3]).real/mass)
    assert exact.value==pytest.approx(expected_value,abs=3e-10)
    counts = solve(problem,method=method,output=NormSquared(),shots=4096,seed=11)
    alpha = chosen.reconstruction.recovery.as_float()
    # Independent Bernoulli physical success mass, with its actual recovery;
    # six standard deviations checks readout wiring rather than promising accuracy.
    p = mass/alpha**2
    assert abs(counts.value-mass)<6*alpha**2*np.sqrt(p*(1-p)/4096)
    samples = solve(problem,method=method,output=Samples(),shots=128,seed=12)
    assert all(index<3 for index in samples.samples.indices.array)


@pytest.mark.parametrize('dim',[2,3])
def test_budgeted_source_bounds_follow_nodes_in_compact_and_padded_layouts(dim):
    from math import fsum
    from nwqlib import plan
    from nwqlib.algorithms import LCHS
    A = np.diag(np.linspace(.2,.4,dim))+.05j*(np.eye(dim,k=1)+np.eye(dim,k=-1))
    problem = LinearDynamics(A=A,initial_state=np.ones(dim)/np.sqrt(dim),source=.1*np.ones(dim),time=.2)

    def selected(select):
        chosen = plan(problem,method=LCHS(hamiltonian_evolution_backend='trotter_error_budgeted',
            lcu_select_implementation=select,approximation_tolerance=.1,duhamel_nodes=2),execution='quantum',seed=7)
        data = chosen._native['native_data']
        trotter = next(item.fact for item in chosen.facts if item.fact.quantity=='trotter_synthesis')
        return data.source_layout,data.select_records,data.select_data.quadrature,trotter

    # branch_controlled keeps only physical branches; multiplexor interleaves
    # zero-coefficient padding slots, so one node occupies different slots.
    compact,compact_records,compact_summary,compact_fact = selected('branch_controlled')
    padded,padded_records,padded_summary,padded_fact = selected('multiplexor')
    assert compact.compact and not padded.compact
    assert len(compact_records)==len(compact.coefficients)==compact.branch_count
    assert len(padded_records)==len(padded.coefficients)>padded.branch_count
    bounds = []
    for (*_,compact_slots),(*_,padded_slots) in zip(compact.applications,padded.applications,strict=True):
        for c,p in zip(compact_slots,padded_slots,strict=True):
            assert compact.coefficients[c]==padded.coefficients[p]
            assert (compact_records[c].step_count,compact_records[c].combined_bound_value)==(
                padded_records[p].step_count,padded_records[p].combined_bound_value)
            bounds.append(compact_records[c].combined_bound_value)
    # Distinct node bounds make a shifted or padded association change the sum.
    assert len(set(bounds))>1 and all(bound>=0 for bound in bounds)
    expected = fsum(abs(compact.coefficients[slot])*compact_records[slot].combined_bound_value
        for slot in range(compact.branch_count))
    # fsum is correctly rounded, so equal nonzero products give equal sums;
    # padding slots contribute exact zeros.
    assert compact_summary['trotter_synthesis_error_bound']==expected==padded_summary['trotter_synthesis_error_bound']>0
    assert compact_fact.availability=='concrete' and compact_fact.value==padded_fact.value
    # The fact weights each application's bound over its own layout slots; the
    # summary sums the slot products directly. Both add the same positive terms
    # in different orders, so 1e-14 covers their rounding.
    assert compact_fact.value.value==pytest.approx(compact_summary['trotter_synthesis_error_bound'],rel=1e-14,abs=0.0)


def test_shifted_identity_source_keeps_nonconstant_duhamel_error_unknown():
    from nwqlib import plan
    from nwqlib.algorithms import LCHS
    # The positive shift rounds to exactly1 at this admitted tiny PSD window:
    # the selected shifted generator is0, but physical recovery is exp(T-t).
    chosen=plan(LinearDynamics(A=-np.eye(2),initial_state=[1,0],source=[0,1],time=.1),
        method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding',psd_tolerance=5e-324,duhamel_nodes=2))
    assert chosen.reconstruction.selected_select=='identity_evolution'
    assert chosen.reconstruction.psd_shift==1.
    gauss=next(value.fact for value in chosen.facts if value.fact.quantity=='duhamel_quadrature')
    assert gauss.availability=='unknown'
