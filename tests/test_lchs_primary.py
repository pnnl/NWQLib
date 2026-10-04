"""Natural LCHS input admission, acquired populations and numerical work ownership."""

import weakref
import numpy as np
import pytest
from nwqlib import LinearDynamics,QuadraticForm,plan,solve
from nwqlib.algorithms.lchs import LCHS,ProviderConfig
from nwqlib.problems.inputs import StateInput
from nwqlib.operators.inputs import OperatorInput


@pytest.mark.parametrize("execution", ("classical", "quantum"))
def test_lchs_archive_preserves_selected_numerical_data(tmp_path, execution):
    from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
    from nwqlib.algorithms.lchs.parameters import selected_identity

    method = LCHS(k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1}))
    chosen = plan(LinearDynamics(A=np.diag([.2, .4]), initial_state=[1., 0.], time=.1),
                  method=method, execution=execution)
    files = ArchiveFiles(tmp_path, 4_000_000)
    saved = save_plan(chosen, files)
    restored = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    assert restored.content_id == chosen.content_id
    assert selected_identity(restored._native["native_data"], restored.problem, method) == selected_identity(
        chosen._native["native_data"], chosen.problem, method)


def test_selected_identity_encoding_distinguishes_mapping_nesting():
    from hashlib import sha256
    from nwqlib.algorithms.lchs.parameters import _feed_selected_value

    def encoded(value):
        digest = sha256()
        _feed_selected_value(digest, value)
        return digest.hexdigest()

    # Moving the key "y" into the nested mapping changes the selected data. A
    # stream that lists mapping entries without their count feeds the same
    # bytes for both values.
    assert encoded({"x": {"IZ": 1.0}, "y": 2.0}) != encoded({"x": {"IZ": 1.0, "y": 2.0}})
    # Legal preservation: insertion order and NumPy scalar types are not
    # selected data, so equal values keep one identity.
    assert encoded({"a": 1.0, "b": 2.0}) == encoded({"b": 2.0, "a": 1.0})
    assert encoded((np.float64(1.5), np.int64(3))) == encoded((1.5, 3))


def test_selected_mps_summary_preserves_unevaluated_error_without_simulation(monkeypatch, tmp_path):
    from qiskit.quantum_info import Statevector
    from nwqlib import load_result

    # Exact unitary dynamics avoids quadrature; one product PREP remains a
    # legal MPS selection even though no fidelity calculation was requested.
    method = LCHS(initial_state_preparation="mps_circuit")
    chosen = plan(LinearDynamics(A=1j*np.diag([1., -1.]), initial_state=[1., 0.], time=.1),
        method=method)
    prep = next(b for b in chosen.construction.selections if b.choice == "mps_circuit")
    assert prep.semantics.epsilon is None
    assert "not evaluated" in prep.semantics.preparation.approximation
    result = solve(chosen)
    np.testing.assert_allclose(result.solution, [np.exp(-.1j), 0.], rtol=0, atol=2e-13)
    with monkeypatch.context() as guard:
        guard.setattr(Statevector, "from_instruction",
            lambda *a, **k: pytest.fail("MPS report/load launched simulation"))
        restored = load_result(result.save(tmp_path / "mps"))
        for value in (result, restored):
            assert "Layered MPS circuit error: not evaluated" in str(value)
            assert "TT-SVD discarded weight is a separate compression quantity" in str(value)


@pytest.mark.parametrize('role',['generator','initial','source','observable'])
def test_descriptive_data_rejects_before_quadrature(role,monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    problem=LinearDynamics(A=np.eye(3),initial_state=[1,0,0],source=[0,1j,0],time=.1)
    observable=problem.A
    fields=dict(A=problem.A,initial_state=problem.initial_state,source=problem.source,time=.1)
    if role=='generator':
        fields['A']=OperatorInput.from_record(problem.A.to_record())
    elif role=='initial':
        fields['initial_state']=StateInput.from_record(problem.initial_state.to_record())
    elif role=='source':
        fields['source']=StateInput.from_record(problem.source.to_record())
    else:
        observable=OperatorInput.from_record(observable.to_record())
    monkeypatch.setattr(terms,'generate_lchs_quadrature',lambda **kw: pytest.fail('descriptive input consumed spectrum'))
    with pytest.raises(ValueError,match='data is unavailable|data unavailable|descriptive'):
        plan(LinearDynamics(**fields),method=LCHS(),output=QuadraticForm(observable=observable))


def test_dense_host_streams_only_requested_operator_and_result_reanalysis_is_free(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    from scipy.linalg import expm
    matrix=np.array([[-.2,.03j],[.03j,.4]])
    problem=LinearDynamics(A=matrix,initial_state=[1,1j],time=.1)
    method=LCHS(k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(problem,method=method,execution='classical')
    quad=chosen._native['native_data'].quadrature
    expected=sum(c*expm(-.1j*(k*quad.l_part+quad.h_part)) for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
    expected*=np.exp(.1*quad.conversion['psd_shift'])
    # The host applies each node evolution to the vector from one Hermitian
    # eigensystem per node, streamed: at most one eigenvector matrix is alive.
    original=terms._node_eigensystem
    refs=[]
    peak=0
    def node(*args):
        nonlocal peak
        values,vectors=original(*args)
        refs.append(weakref.ref(vectors))
        peak=max(peak,sum(ref() is not None for ref in refs))
        return values,vectors
    monkeypatch.setattr(terms,'_node_eigensystem',node)
    result=solve(chosen)
    np.testing.assert_allclose(result.solution,expected@np.array([1,1j]),rtol=0,atol=2e-14)
    assert len(refs)==4 and peak<=2 and not any(ref() is not None for ref in refs)
    monkeypatch.setattr(terms,'_node_eigensystem',lambda *a: pytest.fail('reanalysis repeated evolution'))
    repeated=result.analyze()
    np.testing.assert_array_equal(repeated.solution,result.solution)


@pytest.mark.parametrize('source',[[0,0],[1j,-.2]])
def test_source_branch_population_has_only_active_physical_applications(source):
    from scipy.linalg import expm
    problem=LinearDynamics(A=[[.2,.04j],[.04j,.3]],initial_state=[0,0],source=source,time=.1)
    result=solve(problem,method=LCHS(duhamel_nodes=2),execution='classical')
    if not np.any(source):
        assert result.applications==()
        np.testing.assert_array_equal(result.solution,[0,0])
    else:
        assert len(result.applications)==2
        assert all(next(arg.value for arg in app.arguments if arg.parameter=='source_application')==1 for app in result.applications)
        quad=result.plan._native['native_data'].quadrature
        expected=sum(w*c*expm(-1j*(.1-t)*(k*quad.l_part+quad.h_part))@np.asarray(source)
            for t,w in zip(result.plan.reconstruction.source_nodes,result.plan.reconstruction.source_weights,strict=True)
            for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
        np.testing.assert_allclose(result.solution,expected,rtol=0,atol=2e-14)


def _price_out_dense_route(monkeypatch, host_pf, when=lambda rotations: True):
    """Make the dense route unaffordable where ``when`` holds, so a test of the vector action runs it.

    On a two-dimensional system the single-label action law
    (operators._pauli.pauli_action_requirements) makes the dense route the
    cheaper one; route choice itself is tested by
    test_pauli_action_route_uses_actual_repetitions.
    """
    law = host_pf._route_requirements
    def priced(dimension, steps, rotations, route):
        work, size, products = law(dimension, steps, rotations, route)
        return (10**15 if route == "dense_power" and when(rotations) else work), size, products
    monkeypatch.setattr(host_pf, "_route_requirements", priced)


@pytest.mark.parametrize('order',[1,2])
@pytest.mark.parametrize('with_source',[False,True])
def test_selected_pauli_vector_action_preserves_order_phase_source_and_replay(order,with_source,tmp_path,monkeypatch):
    from nwqlib import load_result
    from nwqlib.algorithms.lchs import host_pf
    _price_out_dense_route(monkeypatch, host_pf)
    from qiskit.quantum_info import Operator
    x=np.array([[0.,1.],[1.,0.]])
    z=np.diag([1.,-1.])
    initial=np.array([1j,.3])
    source=np.array([.2,.1j]) if with_source else None
    matrix=np.diag([-.2,.3])+1j*(.4*x+.17*np.eye(2))
    method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=3,trotter_order=order,duhamel_nodes=2,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(LinearDynamics(A=matrix,initial_state=initial,source=source,time=.2),method=method,execution='classical')
    data=chosen._native['native_data']
    assert all(node['route']=='vector' for action in data.host_actions['applications'] for node in action['positions'])
    expected=np.zeros(2,dtype=complex)
    for action in data.host_actions['applications']:
        duration=.2-action['start']
        vector=source if action['source'] else initial
        for k,c in zip(data.quadrature.k_nodes,data.quadrature.coefficients,strict=True):
            # Analytic exponentials of X,Z, with original identity and the
            # selected PSD shift. This is the literal ordered r=3 product.
            a=.4*duration/3
            b=-.25*k*duration/3
            rx=np.cos(a)*np.eye(2)-1j*np.sin(a)*x
            rz=np.cos(b)*np.eye(2)-1j*np.sin(b)*z
            if order==2:
                half=np.cos(a/2)*np.eye(2)-1j*np.sin(a/2)*x
                step=half@rz@half
            else:
                step=rz@rx
            phase=np.exp(-1j*duration*(.17+k*(.05+chosen.reconstruction.psd_shift)))
            expected+=action['weight']*c*np.exp(chosen.reconstruction.psd_shift*duration)*phase*(step@step@step@vector)
    def forbidden(*args,**kwargs):
        raise AssertionError('selected vector action extracted a full circuit operator or replanned')
    monkeypatch.setattr(Operator,'__init__',forbidden)
    executed=[]
    rotate=host_pf.apply_terms
    monkeypatch.setattr(host_pf,'apply_terms',lambda *a,**k: executed.append(a[0]) or rotate(*a,**k))
    result=solve(chosen)
    # At most12 selected node actions,9 rotations each and tiny matrix sums;
    # 256eps bounds this well-conditioned two-coordinate roundoff witness.
    np.testing.assert_allclose(result.solution,expected,rtol=256*np.finfo(float).eps,atol=256*np.finfo(float).eps)
    # One initial and two Duhamel applications over k=(0,.5,-1,-.5). H+kL keeps
    # X and -k/4 Z, so k=0 has one rotation per step and the other nodes have
    # two (Lie) or three (Suzuki); three steps each, all on the vector route.
    applications=3 if with_source else 1
    rotations=applications*3*(1+3*(1+order))
    assert tuple(data.quadrature.k_nodes)==(0.,.5,-1.,-.5)
    work=dict(chosen.reconstruction.classical_work)
    assert (work['dimension'],work['operator_applications'],work['evolution_calls'],work['pauli_decompositions'],
        work['commutator_checks'],work['matrix_power_products'],work['dense_step_rotations'])==(2,applications,4*applications,2,0,0,0)
    assert len(executed)==work['vector_rotations']==rotations
    kernel,=chosen.construction.kernels
    law,=kernel.resource_laws
    assert law.metric=='classical_work' and kernel.construction_work==work['selection_work']
    # Each executed rotation updates both amplitudes of the host vector.
    assert kernel.invocation_work==law.value==work['size_units']-work['selection_work']>=2*rotations
    line=f"Classical work proxy: {work['size_units']}; vector rotations={rotations}; matrix-power products=0"
    assert line in str(result)
    result.save(tmp_path/'result')
    monkeypatch.setattr(host_pf,'select_actions',forbidden)
    monkeypatch.setattr(host_pf,'apply_node',forbidden)
    loaded=load_result(tmp_path/'result')
    assert loaded.plan.reconstruction.classical_work==chosen.reconstruction.classical_work
    assert loaded.report()['plan']==result.report()['plan'] and line in str(loaded)
    np.testing.assert_array_equal(loaded.analyze().solution,result.solution)
    assert len(executed)==rotations


def test_pauli_action_route_uses_actual_repetitions(monkeypatch):
    from nwqlib.algorithms.lchs import host_pf
    # Vector work r*R*(W_action + 2*D) with the single-label action law
    # W_action = 5D + q + 6: at D=4, r=R=1 the vector route costs 36 and dense
    # powering 32, so dense powering is selected. At D=2, r=3, R=3 dense costs
    # 32 against 189. At r=2**40 the vector route is inadmissible and the
    # dense power uses 40 squarings. At D=8, r=R=1 the vector route (65)
    # beats dense (128).
    assert host_pf._action_route(
        4, 1, 1, max_work=1000, max_bytes=10_000,
    ) == ("dense_power", 32, 2048, 0)
    assert host_pf._action_route(
        2, 3, 3, max_work=1000, max_bytes=10_000,
    ) == ("dense_power", 32, 640, 2)
    route, work, size, products = host_pf._action_route(
        2, 2**40, 3, max_work=1000, max_bytes=10_000,
    )
    assert (route, work, size, products) == ("dense_power", 336, 640, 40)
    assert host_pf._action_route(8, 1, 1, max_work=1000, max_bytes=10_000) == ("vector", 65, 2136, 0)
    with pytest.raises(ValueError,match='max_'):
        host_pf._action_route(2,2**40,3,max_work=10,max_bytes=10_000)
    def forbidden(*args,**kwargs):
        raise AssertionError('large-r route expanded rotations')
    monkeypatch.setattr(host_pf,'apply_terms',forbidden)
    class ReachedPower(Exception):
        pass
    def power(step,r):
        assert r==2**40
        raise ReachedPower
    monkeypatch.setattr(np.linalg,'matrix_power',power)
    from types import SimpleNamespace
    node=dict(route=route,record=SimpleNamespace(step_count=2**40),sequence=(('X',.01),),phase=0.)
    with pytest.raises(ReachedPower):
        host_pf.apply_node(node["sequence"],node["phase"],node["record"].step_count,node["route"],np.array([1.,0.]))


def test_mixed_pauli_routes_record_executed_rotations_and_power_products(monkeypatch):
    from nwqlib.algorithms.lchs import host_pf
    # The one-rotation k=0 node takes the vector route.
    _price_out_dense_route(monkeypatch, host_pf, when=lambda rotations: rotations == 1)
    x=np.array([[0.,1.],[1.,0.]])
    method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=10,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(LinearDynamics(A=np.diag([-.2,.3])+1j*(.4*x+.17*np.eye(2)),initial_state=[1j,.3],time=.2),
        method=method,execution='classical')
    vector,dense,powers=[],[],[]
    rotate,masks,power=host_pf.apply_terms,host_pf._pauli_masks,np.linalg.matrix_power
    monkeypatch.setattr(host_pf,'apply_terms',lambda *a,**k: vector.append(a[0]) or rotate(*a,**k))
    monkeypatch.setattr(host_pf,'_pauli_masks',lambda label: dense.append(label) or masks(label))
    monkeypatch.setattr(np.linalg,'matrix_power',lambda step,r: powers.append(r) or power(step,r))
    result=solve(chosen)
    # The dense route is priced out for one-rotation nodes, so k=0, which
    # keeps X alone, takes ten vector steps of one rotation. The three X,Z nodes
    # form one three-rotation Suzuki step each, raised to r=10=0b1010 with
    # three squarings and one product.
    routes=[node['route'] for action in chosen._native['native_data'].host_actions['applications'] for node in action['positions']]
    assert routes==['vector']+['dense_power']*3
    work=dict(chosen.reconstruction.classical_work)
    assert len(vector)==work['vector_rotations']==10
    assert len(dense)==work['dense_step_rotations']==9 and powers==[10]*3
    assert work['matrix_power_products']==3*4
    assert f"Classical work proxy: {work['size_units']}; vector rotations=10; matrix-power products=12" in str(result)


def test_zero_l_omits_eigensolve_and_its_work_admission(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    def forbidden(*args,**kwargs):
        raise AssertionError('exact zero L requested spectral work')
    monkeypatch.setattr(np.linalg,'eigvalsh',forbidden)
    data=terms.generate_lchs_quadrature(matrix=1j*np.diag([.2,.3]),final_time=.1,
        method=LCHS(),max_spectral_work=1)
    assert data.l_norm==0 and np.count_nonzero(data.l_part)==0
    with pytest.raises(ValueError,match='max_spectral_work'):
        terms.generate_lchs_quadrature(matrix=np.diag([.2,.3]),final_time=.1,
            method=LCHS(),max_spectral_work=1)


@pytest.mark.parametrize('steps,route',[(3,'vector'),(10,'dense_power')])
def test_pauli_action_matches_independent_y_rotation_and_selected_native_step(steps,route):
    from nwqlib.algorithms.lchs import host_pf, time_independent_terms as terms
    x=np.array([[0.,1.],[1.,0.]])
    y=np.array([[0.,-1j],[1j,0.]])
    z=np.diag([1.,-1.])
    l=.3*np.eye(2)+.1*z
    h=.4*x+.2*y+.17*np.eye(2)
    method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=steps)
    from _lchs_suzuki_reference import fixed_trotter_term_unitary_with_record
    original,record=fixed_trotter_term_unitary_with_record(final_time=.2,k_value=1.,
        method=method,trotter_paulis=terms._trotter_pauli_decomposition(l,h))
    sequence=host_pf._rotation_sequence((('X',.4),('Y',.2),('Z',.1)),.2,steps,2)
    node=dict(sequence=sequence,phase=-.2*.47,record=record,route=route)
    vector=np.array([1.,1j])/np.sqrt(2)
    def rotation(pauli,angle):
        return np.cos(angle)*np.eye(2)-1j*np.sin(angle)*pauli
    rx=rotation(x,.4*.2/(2*steps))
    ry=rotation(y,.2*.2/(2*steps))
    rz=rotation(z,.1*.2/steps)
    step=rx@ry@rz@ry@rx
    expected=np.exp(-1j*.2*.47)*np.linalg.matrix_power(step,steps)@vector
    np.testing.assert_allclose(host_pf.apply_node(node["sequence"],node["phase"],node["record"].step_count,node["route"],vector),expected,rtol=0,atol=3e-14)
    np.testing.assert_allclose(original@vector,expected,rtol=0,atol=3e-14)
    # The opposite admitted implementation obeys the same selected relation.
    other=dict(node,route='dense_power' if route=='vector' else 'vector')
    np.testing.assert_allclose(host_pf.apply_node(other["sequence"],other["phase"],other["record"].step_count,other["route"],vector),expected,rtol=0,atol=3e-14)


def test_budgeted_host_keeps_node_specific_steps_and_scalar_projection(monkeypatch):
    from nwqlib import NormSquared
    from nwqlib.algorithms.lchs import host_pf
    _price_out_dense_route(monkeypatch, host_pf)
    from _lchs_suzuki_reference import sorted_pauli_suzuki_2_reference
    problem=LinearDynamics(A=np.diag([.2,.7])+1j*np.array([[0.,.4],[.4,0.]]),
        initial_state=[1.,1j],source=[.1j,.2],time=.8)
    method=LCHS(hamiltonian_evolution_backend='trotter_error_budgeted',duhamel_nodes=2,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':0}))
    chosen=plan(problem,method=method,execution='classical',output=NormSquared())
    data=chosen._native['native_data']
    steps={node['record'].step_count for action in data.host_actions['applications'] for node in action['positions']}
    assert len(steps)>1
    expected=np.zeros(2,dtype=complex)
    for action in data.host_actions['applications']:
        vector=data.source if action['source'] else data.initial
        for k,c,node in zip(data.quadrature.k_nodes,data.quadrature.coefficients,action['positions'],strict=True):
            native=sorted_pauli_suzuki_2_reference(data.quadrature.l_part,data.quadrature.h_part,
                k_value=k,final_time=.8-action['start'],reps=max(1,node['record'].step_count))
            expected+=action['weight']*c*(native@vector)
    executed=[]
    rotate=host_pf.apply_terms
    monkeypatch.setattr(host_pf,'apply_terms',lambda *a,**k: executed.append(a[0]) or rotate(*a,**k))
    result=solve(chosen)
    assert result.value==pytest.approx(float(np.vdot(expected,expected).real),rel=3e-13,abs=3e-14)
    assert result.artifact is None
    # k=(0,1,-2,-1) leaves X alone at k=0 and X,Z elsewhere, in three
    # applications. One ordered structure over the union {X,Z} serves all
    # four distinct nodes: one pair test and one nested test, performed once
    # (P=1, N=1, E=F=1), and each node pays its own contraction V=E+F. The
    # fixed-step selection of the same problem differs by that shared census,
    # its p*q label packing and the 16 scalar-selection units per
    # application/node (time_independent_terms._node_bound_coefficients).
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.subroutines.trotterization.error_budget import census_work, evaluate_trotter_bound
    from nwqlib.algorithms.lchs.time_independent_terms import _combined_trotter_terms,_trotter_pauli_decomposition
    assert tuple(data.quadrature.k_nodes)==(0.,1.,-2.,-1.)
    work=dict(chosen.reconstruction.classical_work)
    fixed=dict(plan(problem,method=method.revise(hamiltonian_evolution_backend='trotter'),execution='classical',
        output=NormSquared()).reconstruction.classical_work)
    assert work['commutator_checks']==2 and work['dense_step_rotations']==0
    assert work['census_structures']==1 and work['census_contractions']==work['combined_nodes']==4
    shared=census_work(2,1,order=2,variant='exact_census',pairs=1,nested=1,triples=1)+3*2
    assert work['selection_work']-fixed['selection_work']==2*1+shared+16*3*4
    # The fixed-step selection alone is the decomposition W_dec, the one-time
    # support alignment W_align = (q+4)*s + (q+2)*u and each distinct node's
    # combination and streamed dropped mass K*(2*m_L + 23*u + 8)
    # (time_independent_terms._coefficient_preparation_work). Here
    # L = .45I - .25Z and H = .4X: q = 1, m_L = 2, s = 3, u = 3 and K = 4.
    from nwqlib.algorithms.lchs.time_independent_terms import _pauli_decomposition_requirements, _trotter_budget_quadrature_terms
    assert fixed['selection_work']==_pauli_decomposition_requirements(2)[0]+(5*3+3*3)+4*(2*2+23*3+8)
    # Reused selections perform no new commutation test: summed over every
    # application's summary, the tests equal the one shared structure's
    # P + N, and a single record carries them.
    summaries=[_trotter_budget_quadrature_terms([node['record'] for node in action['positions']],
        coefficients=data.quadrature.coefficients,method=data.method)['trotter_algebraic_work_counts']
        for action in data.host_actions['applications']]
    assert sum(s['pair_commutation_checks']+s['nested_commutation_checks'] for s in summaries)==work['commutator_checks']==2
    counts=[node['record'].algebraic_work_counts for action in data.host_actions['applications'] for node in action['positions']]
    assert sum(1 for c in counts if c['pair_commutation_checks'] or c['nested_commutation_checks'])==1
    # Each node's reused coefficient gives the bound of a separate census of
    # its own kept terms at every application, including k=0's smaller set.
    decomposition=_trotter_pauli_decomposition(data.quadrature.l_part,data.quadrature.h_part)
    for action in data.host_actions['applications']:
        for k,node in zip(data.quadrature.k_nodes,action['positions'],strict=True):
            kept=_combined_trotter_terms(decomposition,k_value=float(k)).traceless_terms
            record=node['record']
            assert record.pf_bound_value==evaluate_trotter_bound(SparsePauliOp.from_list(kept),
                time=.8-action['start'],steps=record.step_count,order=2)
    assert len(executed)==work['vector_rotations']
    kernel,=chosen.construction.kernels
    assert kernel.construction_work==work['selection_work']
    assert kernel.invocation_work==kernel.resource_laws[0].value==work['size_units']-work['selection_work']
    assert f"vector rotations={len(executed)}; matrix-power products=0" in str(result)


_SPD = [[.3,.1],[.1,.5]]
# Hermitian part with eigenvalue -.2, so planning shifts L and every
# application carries the recovery exp(shift*(T-s)).
_SHIFTED = [[-.2,.1,0],[.1,.4,.05j],[0,-.05j,.3]]


@pytest.mark.parametrize('matrix,options,source,unknown',[
    (_SPD,dict(hamiltonian_evolution_backend='trotter_error_budgeted'),None,{}),
    (_SPD,dict(hamiltonian_evolution_backend='trotter_error_budgeted',trotter_order=1),None,{}),
    (_SPD,dict(),None,{}),
    (_SPD,dict(hamiltonian_evolution_backend='trotter',trotter_steps=3),None,
     {'trotter_synthesis':'missing_stage:trotter_synthesis'}),
    (_SPD,dict(hamiltonian_evolution_backend='trotter_error_budgeted',duhamel_nodes=2,approximation_tolerance=.3),[.1,-.2j],
     {'duhamel_quadrature':'missing_stage:duhamel_quadrature'}),
    (_SPD,dict(duhamel_nodes=2),[.1,-.2j],{'duhamel_quadrature':'missing_stage:duhamel_quadrature'}),
    (_SHIFTED,dict(hamiltonian_evolution_backend='trotter_error_budgeted'),None,{}),
])
def test_classical_plan_publishes_the_component_facts_its_result_reports(matrix,options,source,unknown):
    """Planning fixes every component input, so the Plan states what solve records.

    The kernel and k-quadrature stages weight the provider bounds by each
    application's physical input weight |w_a|*||u_a|| and PSD recovery
    exp(shift*(T-s_a)), and trotter_synthesis sums |c_j| times each node's
    stored product-formula and pruning bound. None of them needs the host
    action. The Duhamel remainder needs spectral norms that only explicit
    refinement evaluates, and a fixed-step formula has no selected synthesis
    bound, so those stay unknown with their stage reason in both records.
    """
    from math import exp, fsum
    initial = [1]+[0]*(len(matrix)-1)
    problem = LinearDynamics(A=matrix,initial_state=initial,time=.2,source=source)
    chosen = plan(problem,method=LCHS(**options),execution='classical')
    planned = {item.fact.quantity:item for item in chosen.facts}
    assert planned == {term.name:term.fact for term in chosen.error_model.terms if term.role=='listed_only'}
    result = solve(chosen)
    assert {item.fact.quantity:item for item in result.facts if item.fact.quantity in planned} == planned
    assert {name:item.fact.reason for name,item in planned.items() if item.fact.availability=='unknown'} == unknown
    # A concrete stage is sum_a |w_a|*||u_a||*exp(shift*(T-s_a)) times its stored bound.
    rec,data = chosen.reconstruction,chosen._native['native_data']
    assert (rec.psd_shift > 0) == (matrix is _SHIFTED)
    inputs = [(0.,rec.initial_scale.as_float())]+[(s,abs(w)*rec.source_scale.as_float())
                                                   for s,w in zip(rec.source_nodes,rec.source_weights,strict=True)]
    factors = [scale*exp(rec.psd_shift*(problem.elapsed_time-start)) for start,scale in inputs]
    for name,bound in (('kernel_approximation',rec.kernel_approximation_bound),('k_quadrature',rec.quadrature_bound)):
        assert planned[name].fact.value.value == pytest.approx(fsum(f*bound for f in factors),rel=4*np.finfo(float).eps)
    if options.get('hamiltonian_evolution_backend') == 'trotter_error_budgeted':
        assert [action['start'] for action in data.host_actions['applications']] == [start for start,_ in inputs]
        stored = fsum(f*fsum(abs(c)*node['record'].combined_bound_value
                             for c,node in zip(data.quadrature.coefficients,action['positions'],strict=True))
                      for f,action in zip(factors,data.host_actions['applications'],strict=True))
        assert planned['trotter_synthesis'].fact.value.value == pytest.approx(stored,rel=4*np.finfo(float).eps)
        assert stored > 0


@pytest.mark.parametrize('options',[
    dict(hamiltonian_evolution_backend='trotter_error_budgeted'),
    dict(hamiltonian_evolution_backend='trotter',trotter_steps=3),
    dict(duhamel_nodes=16),
])
def test_classical_source_receipts_are_reserved_with_their_plan_at_default_scale(options):
    """Known host receipts are funded by the reservation, not by the provider allowance.

    The default grid has 204 k-nodes. With a constant source the product
    formulas have nine applications (eight Duhamel nodes) and the dense case
    seventeen. Their receipts are fixed by the Plan, so the kernel declares
    their exact JSON bytes. Per-node steps, rotations and route work stay in
    the Plan's host actions, which selected_grid reads.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.algorithms.lchs.selected_grid import host_applications
    problem = LinearDynamics(A=[[.3,.1],[.1,.5]],initial_state=[1,0],time=.2,source=[.1,-.2j])
    chosen = plan(problem,method=LCHS(**options),execution='classical')
    data = chosen._native['native_data']
    assert len(data.quadrature.k_nodes) == 204
    result = solve(chosen)
    kernel, = chosen.construction.kernels
    assert len(result.applications) == 1+len(chosen.reconstruction.source_nodes)
    assert kernel.application_bytes == _json_bound(result.applications,10**9) > 0
    assert not any(item.parameter.startswith('node_') for application in result.applications
                   for item in application.arguments)
    if data.host_actions is not None:
        steps = [app.steps for app in host_applications(chosen,result,data)]
        assert steps == [tuple(node['record'].step_count for node in action['positions']) for action in data.host_actions['applications']]


def test_undeclared_completion_bytes_spend_the_metadata_allowance(monkeypatch):
    """Declared receipt bytes are reserved with the outputs, and every other byte stays capped.

    The declared receipts have a JSON size bound of about 51 kB, yet an 8 KiB
    allowance admits the solve, because only the undeclared metadata (the
    event, the chunk envelope and the array publication, about 4.4 kB)
    spends it. The allowance still rejects an undeclared 8 KiB argument name
    added to each receipt. It also rejects 16 KiB of undeclared text outside
    shrunken receipts, because the credit is the smaller of the declared and
    the returned receipt bytes, so a large declaration cannot hide it.
    """
    from dataclasses import replace
    from nwqlib.execution import ExecutionLimits
    from nwqlib.algorithms.lchs import host
    from nwqlib.ir import Binding
    problem = LinearDynamics(A=[[.3,.1],[.1,.5]],initial_state=[1,0],time=.2,source=[.1,-.2j])
    chosen = plan(problem,method=LCHS(hamiltonian_evolution_backend='trotter_error_budgeted'),execution='classical')
    kernel, = chosen.construction.kernels
    limits = ExecutionLimits(max_completion_metadata_bytes=8192)
    assert kernel.application_bytes > limits.max_completion_metadata_bytes
    assert solve(chosen,limits=limits).solution is not None
    application,execute = host._application,host._execute

    def padded(receipt):
        return receipt.revise(arguments=receipt.arguments+(Binding(parameter='x'*8192,value=0),))
    with monkeypatch.context() as patch:
        patch.setattr(host,'_application',lambda *args,**kwargs: padded(application(*args,**kwargs)))
        with pytest.raises(ValueError,match='completion metadata exceeds max_completion_metadata_bytes'):
            solve(chosen,limits=limits)

    def shrunken(*args,**kwargs):
        output = execute(*args,**kwargs)
        return replace(output,applications=tuple(item.revise(facts=()) for item in output.applications),
                       physical_scale=None,physical_scale_unavailable='x'*16384)
    monkeypatch.setattr(host,'_execute',shrunken)
    with pytest.raises(ValueError,match='completion metadata exceeds max_completion_metadata_bytes'):
        solve(chosen,limits=limits)


@pytest.mark.parametrize('options',[dict(hamiltonian_evolution_backend='trotter_error_budgeted'),dict(duhamel_nodes=32)])
def test_durable_classical_source_run_reopens_from_a_small_manifest(options,tmp_path):
    """Plan-size data lives in its own archive files, so run.json stays a small manifest.

    The default budgeted source case stores host actions for 204 k-nodes and
    nine applications, and the dense case has 33 applications. load_run
    reads run.json under a one MiB cap before the journal supplies the data
    allowance. run.json therefore only names the selection file, and the
    LCHS selection in turn only names the file of its native numerical data.
    What stays in the selection is the Plan record, the input manifests and
    the Method, which this test bounds at 128 KiB.
    """
    import json
    from nwqlib import load_run
    from nwqlib._prepared_execution import Run
    problem = LinearDynamics(A=[[.3,.1],[.1,.5]],initial_state=[1,0],time=.2,source=[.1,-.2j])
    chosen = plan(problem,method=LCHS(**options),execution='classical')
    path = tmp_path/'run'
    with Run(chosen,directory=path) as run:
        result = run.wait()
    assert (path/'run.json').stat().st_size < 64*1024
    manifest = json.loads((path/'run.json').read_text())
    selection = path/manifest['selection']
    assert selection.stat().st_size < 128*1024
    native = json.loads(selection.read_text())['selected']['native']
    assert type(native) is str and (path/native).is_file()
    with load_run(path,backend=None) as reopened:
        # A source Plan binds no homogeneous coefficient plan, at planning or on loading.
        assert reopened.plan._native['coefficient_plan'] is None
        again = reopened.wait()
    assert again.content_id == result.content_id
    np.testing.assert_array_equal(again.solution,result.solution)


def test_source_plan_coefficient_table_fields_follow_the_selected_table():
    """Classical source Plans report the dimensionless kernel table, quantum ones the source branch layout."""
    source = np.array([.2, -.1])
    problem = LinearDynamics(A=[[.6, .1], [.1, .3]], initial_state=[1., 0.], time=.1, source=source)
    method = LCHS(approximation_tolerance=.1, duhamel_nodes=2)
    classical = plan(problem, method=method, execution="classical", seed=1).reconstruction
    quantum = plan(problem, method=method, seed=1).reconstruction
    assert quantum.psd_shift == classical.psd_shift == 0
    kernel_nodes = classical.physical_branches
    assert quantum.physical_branches == 3*kernel_nodes  # The initial application and two Duhamel nodes.
    assert quantum.padded_branches >= quantum.physical_branches > classical.physical_branches
    # Each quantum coefficient carries the initial norm 1 or a weight times ||b||.
    expected = classical.coefficient_l1_norm*(1+np.linalg.norm(source)*sum(quantum.source_weights))
    assert quantum.coefficient_l1_norm == pytest.approx(expected, rel=1e-14)


def test_dense_host_admits_every_node_exponential_before_planning_succeeds(monkeypatch):
    # Each dense node evolution comes from one Hermitian eigensystem of
    # k L + H, 8 D**3 units each (time_independent_terms.
    # _spectral_host_requirements, which also prices the generator passes,
    # projections and phase actions). A max_select_work of 8 D**3 per node
    # therefore lies below the admitted law and must refuse the Plan. The
    # default limit admits it.
    l_part = np.array([[.4, .1, 0, 0], [.1, .3, .1, 0], [0, .1, .35, .05], [0, 0, .05, .3]])
    h_part = np.diag([.1, .1, .1], 1) + np.diag([.1, .1, .1], -1)
    problem = LinearDynamics(A=l_part + 1j*h_part, initial_state=[1., 0., 0., 0.], time=.1)
    grid = ProviderConfig(implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": 0})
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    admitted = plan(problem, method=LCHS(k_quadrature=grid), execution="classical")
    work = dict(admitted.reconstruction.classical_work)
    formed = []
    original = terms._node_eigensystem
    monkeypatch.setattr(terms, "_node_eigensystem", lambda *a: formed.append(a[0]) or original(*a))
    solve(admitted)
    # The two-qubit grid has four k nodes, one eigensystem each and no
    # exponential, and planning charges every eigensystem that the host forms.
    assert work["eigh_calls"] == len(formed) == 4 and work["dense_expm_calls"] == 0
    with pytest.raises(ValueError, match="max_select_work"):
        plan(problem, method=LCHS(k_quadrature=grid, max_select_work=8*4**3*work["eigh_calls"]), execution="classical")


def _aer_run_counter(monkeypatch):
    import qiskit_aer
    calls = []
    original = qiskit_aer.AerSimulator.run
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(qiskit_aer.AerSimulator, "run", counted)
    return calls


def _trotter_method(bits=2):
    return LCHS(hamiltonian_evolution_backend="trotter", k_quadrature=ProviderConfig(
        implementation="signed_binary_uniform", parameters={"num_qubits": bits, "lsb_position": -1}))


@pytest.mark.parametrize("route", ["dense", "periodic"])
def test_exact_lchs_scalar_acquires_one_simulation_and_reloads_without_replay(route, tmp_path, monkeypatch):
    """An exact L = 4 Pauli observable needs one Aer simulation, and reanalysis after load_run none."""
    import nwqlib
    from nwqlib import NormalizedExpectation
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import Run
    from nwqlib.operators import ingest_pauli
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_occupation
    terms = (("ZI", .5), ("IZ", .3), ("XX", .2), ("YY", -.1))
    output = NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2))
    if route == "dense":
        problem, method = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1), _trotter_method()
    else:
        problem = LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=1., diffusion=.25, potential=.3),
                                 initial_state=ingest_occupation((0, 1), num_qubits=2), time=.5)
        method = LCHS(hamiltonian_evolution_backend="trotter", lchs_kernel="cauchy_density", trotter_steps=None,
                      trotter_synthesis_tolerance=1e-4, k_quadrature=ProviderConfig(
                          implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": 0}))
    chosen = plan(problem, method=method, output=output, execution="quantum", seed=7)
    assert [experiment.name for experiment in chosen.experiments] == ["projected_moments"]
    calls = _aer_run_counter(monkeypatch)
    with Run(chosen, directory=tmp_path / "run", progress=False) as run:
        result = chosen.method.execute(chosen, run=run)
    assert len(calls) == 1 and result.reduction is not None and result.value is not None
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        again = reopened.plan.method.execute(reopened.plan, run=reopened)
    assert len(calls) == 1 and again.value == result.value and again.reduction == result.reduction
    solved = solve(chosen)
    solved.save(tmp_path / "result")
    assert len(calls) == 2 and solved.value == result.value
    assert nwqlib.load_result(tmp_path / "result").reduction == solved.reduction and len(calls) == 2
    # The Aer receipt lists only "amplitude-derived masses", and resolving
    # that label leaves a qualified saved-state budget.
    from nwqlib._quantum_readout import PROJECTED_MASS_EXCLUSIONS
    [receipt] = solved.data.receipts
    assert receipt.probability_window_exclusions == PROJECTED_MASS_EXCLUSIONS
    assert receipt.saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0] is not None


def test_saved_exact_lchs_result_loads_in_a_fresh_process(tmp_path):
    """load_result in a new interpreter resolves the saved reduction's reducer before any LCHS module loads."""
    import json
    import os
    import subprocess
    import sys
    import nwqlib
    from nwqlib import NormalizedExpectation
    from nwqlib.operators import ingest_pauli
    output = NormalizedExpectation(observable=ingest_pauli((("ZI", .5), ("XX", .2)), num_qubits=2))
    problem = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1)
    result = solve(plan(problem, method=_trotter_method(), output=output, execution="quantum", seed=7))
    assert result.reduction is not None and result.value is not None
    result.save(tmp_path / "result")
    fields = ("value", "norm_squared", "numerator", "numerator_frame", "unavailable", "content_id")
    source = (f"import json, nwqlib\nloaded = nwqlib.load_result({str(tmp_path / 'result')!r})\n"
              f"print(json.dumps([nwqlib.__file__] + [getattr(loaded, name) for name in {fields!r}]))")
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=dict(os.environ),
                               timeout=300)
    assert completed.returncode == 0, completed.stderr
    origin, *loaded = json.loads(completed.stdout.splitlines()[-1])
    assert origin == nwqlib.__file__
    assert loaded == [getattr(result, name) for name in fields]


@pytest.mark.parametrize("kind", ["solution", "state_vector", "unrepresentable"])
def test_saved_quantum_lchs_vector_publishes_its_array_or_reason_and_loads(tmp_path, kind):
    """A quantum Solution or StateVector publishes its amplitude array, or the reason it has none, and loads.

    The growth problem ``A = -720 I``, ``u0 = e_0``, ``T = 1`` has the
    physical solution ``exp(720) e_0``, about 4.9e312, which complex128 cannot
    represent. Its quantum Solution publishes no array, no norm squared and
    the acquisition's reason; the other two rows publish their array and no
    reason. Each legal Result loads with its identity.
    """
    import nwqlib
    from nwqlib import Solution, StateVector

    if kind == "unrepresentable":
        output, problem = Solution(), LinearDynamics(A=[[-720., 0.], [0., -720.]], initial_state=[1., 0.], time=1.)
    else:
        output = Solution() if kind == "solution" else StateVector()
        problem = LinearDynamics(A=np.diag([.1, .25, .4]), initial_state=[1., 2., 3.], time=.3)
    result = solve(plan(problem, method=_trotter_method(), output=output, execution="quantum", seed=7))
    if kind == "unrepresentable":
        assert result.artifact is None and result.norm_squared is None
        assert result.unavailable == "selected physical vector is not representable in complex128"
    else:
        assert result.artifact is not None and result.unavailable is None
    result.save(tmp_path / "result")
    assert nwqlib.load_result(tmp_path / "result").content_id == result.content_id


def _reference_slice(chosen):
    """The bound Plan circuit in Statevector, with the physical prefix by explicit bit enumeration."""
    from qiskit.quantum_info import Statevector
    from nwqlib.blocks.lowering import lower_qiskit
    _, construction = chosen.resolve("projected_moments")._selected_construction(chosen)
    selected = {record.content_id for record in construction.selections}
    circuit = lower_qiskit(construction, blocks=tuple(block for block in chosen.blocks
                                                      if block.record.content_id in selected)).circuit
    state = Statevector(circuit).data
    rec = chosen.reconstruction
    success, coordinates = rec.success_bits, rec.system_bits
    algorithm = physical = 0.
    vector = np.zeros(rec.encoded_dimension, dtype=complex)
    for index, amplitude in enumerate(state):
        if any((index >> bit) & 1 for bit in success):
            continue
        algorithm += abs(amplitude)**2
        coordinate = sum(((index >> bit) & 1) << place for place, bit in enumerate(coordinates))
        if coordinate < rec.dimension:
            vector[coordinate] = amplitude
            physical += abs(amplitude)**2
    return vector, algorithm, physical


def _pauli_matrix(label):
    one = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    matrix = np.eye(1)
    for axis in label:  # qubit zero is the rightmost letter, the last Kronecker factor
        matrix = np.kron(matrix, one[axis])
    return matrix


@pytest.mark.parametrize("dimension", [3, 4])
def test_exact_lchs_outputs_match_an_independent_physical_slice_reference(dimension):
    """Projected p, p_alg and q against the same bound circuit in Statevector.

    The receipts exclude supplied-state operations, so this uses the unit-scale
    regression contract atol=2e-12, rtol=0 for raw masses and q with C <= 1,
    propagated to the normalized value by the quotient rule and to physical
    outputs by the recovery. It is not a native error certificate.
    """
    from nwqlib import NormalizedExpectation, NormSquared
    from nwqlib._quantum_readout import pair_value
    matrix = np.diag([.3, -.2, .1, .15][:dimension]) + .05 * (np.eye(dimension, k=1) + np.eye(dimension, k=-1))
    generator = np.diag([.1, .25, .4, .55]) + .6 * (np.eye(4, k=1) + np.eye(4, k=-1))
    problem = LinearDynamics(A=generator[:dimension, :dimension], initial_state=[1., 2., 3., 4.][:dimension], time=.8)
    method = _trotter_method()
    tolerance = 2e-12
    for output in (NormalizedExpectation(observable=matrix), QuadraticForm(observable=matrix), NormSquared()):
        chosen = plan(problem, method=method, output=output, execution="quantum", seed=7)
        result = solve(chosen)
        vector, algorithm, physical = _reference_slice(chosen)
        rec = chosen.reconstruction
        terms = rec.terms if not isinstance(output, NormSquared) else ()
        coefficient_norm = sum(abs(c) for _, c in terms)
        assert coefficient_norm <= 1
        observable = sum((c * _pauli_matrix(label) for label, c in terms), np.zeros((rec.encoded_dimension,) * 2))
        numerator = float(np.vdot(vector, observable @ vector).real)
        stats = result.reduction
        p, q = pair_value(stats.physical_mass), pair_value(stats.numerator)
        assert abs(float(p) - physical) <= tolerance and abs(float(pair_value(stats.success_mass)) - algorithm) <= tolerance
        assert abs(float(q) - numerator) <= tolerance and physical > .1
        gamma2 = rec.recovery.as_float()**2
        if dimension == 3:
            assert algorithm - physical > 1e-11  # a nonzero dummy population is excluded from p
        if isinstance(output, NormalizedExpectation):
            bound = (tolerance + abs(numerator) / physical * tolerance) / (physical - tolerance) + 2**-53 * abs(result.value)
            assert abs(result.value - numerator / physical) <= bound
        elif isinstance(output, QuadraticForm):
            assert abs(result.value - gamma2 * numerator) <= gamma2 * tolerance * (1 + 1e-12)
        else:
            assert abs(result.value - gamma2 * physical) <= gamma2 * tolerance * (1 + 1e-12)


def test_scaled_reduction_publishes_zero_physical_mass_as_unavailable_normalization():
    """A state with no physical-slice mass gives zero p and q; the normalized quotient is unavailable."""
    from nwqlib._quantum_readout import pair_ratio, reduce_scaled, _pauli_table
    native = np.zeros(8, dtype=complex)
    native[3] = 1.  # success bit 0 reads one
    table = _pauli_table((("ZI", .5), ("XX", .25)), 2)
    result = reduce_scaled(native, coordinates=(1, 2), success=((0, 0),), conditions=(), dimension=4, table=table,
                           window=1e-12)
    assert result["physical"] == (0.0, 0) and result["numerator"] == (0.0, 0) and result["complete"] == (0.5, 1)
    assert result["success"] == (0.0, 0)
    assert pair_ratio(result["numerator"], result["physical"]) is None


def test_reducer_mass_window_rejects_lost_complete_norm_and_keeps_subpopulations():
    """The complete native norm is validated before publication; masses are subpopulations, not ones."""
    from nwqlib._quantum_readout import (fraction_pair as pair, pair_value, reduce_scaled, validate_saved_masses,
                                         _pauli_table)
    validate_saved_masses(pair(1), pair(.9), pair(.7), (8, 4, 3), window=1e-12)
    with pytest.raises(ValueError, match="outside the producing window"):
        validate_saved_masses(pair(.99), pair(.9), pair(.7), (8, 4, 3), window=1e-12)
    with pytest.raises(ValueError, match="population relation"):
        validate_saved_masses(pair(1), pair(.7), pair(.9), (8, 4, 3), window=1e-12)
    native = np.array([.6, .0, .8, 0, 0, 0, 0, 0], dtype=complex)
    table = _pauli_table((("ZI", .5),), 2)
    accepted = reduce_scaled(native * (1 + 1e-13), coordinates=(1, 2), success=((0, 0),), conditions=(), dimension=3,
                             table=table, window=1e-12)
    # The raw accepted values are kept, not renormalized to one.
    assert accepted["complete"] == accepted["success"]
    assert pair_value(accepted["complete"]) != 1 and abs(float(pair_value(accepted["complete"])) - (1+1e-13)**2) < 1e-15
    with pytest.raises(ValueError, match="outside the producing window"):
        reduce_scaled(native * np.sqrt(.99), coordinates=(1, 2), success=((0, 0),), conditions=(), dimension=3,
                      table=table, window=1e-12)


def test_sampled_lchs_groups_share_bases_and_decode_parities_from_their_counts():
    """G qubit-wise commuting groups (+1 unrotated mass setting when padded), with their recorded moments."""
    import math
    from nwqlib import NormalizedExpectation
    from nwqlib._quantum_readout import physical_moment, weighted_group_moments
    from nwqlib.operators import ingest_pauli
    terms = (("ZI", .5), ("IZ", .3), ("ZZ", -.2), ("XX", .2), ("XI", -.15), ("YY", .07))
    method = _trotter_method()
    for dimension, output in ((4, NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2))),
                              (3, NormalizedExpectation(observable=np.diag([.3, -.2, .1]) + .05 * np.eye(3, k=1) + .05 * np.eye(3, k=-1)))):
        problem = LinearDynamics(A=np.diag([.1, .25, .4, .55][:dimension]), initial_state=[1., 2., 3., 4.][:dimension], time=.3)
        chosen = plan(problem, method=method, output=output, execution="quantum", shots=3000, seed=7)
        rec = chosen.reconstruction
        labels = [label for label, c in rec.terms if c != 0 and set(label) != {"I"}]
        padded = dimension != 4
        # Each selected one-site basis change U maps its axis to Z: U X U^dagger = Z for H,
        # U Y U^dagger = Z for S^dagger then H. A Y site rotated by H alone gives -Y.
        pauli = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
        gates = {"h": np.array([[1, 1], [1, -1]]) / np.sqrt(2), "sdg": np.diag([1, -1j])}
        rotated = set()
        for record in chosen.construction.selections:
            if record.signature.name.startswith("rotate_"):
                unitary = np.eye(2)
                for step in record.decomposition:
                    unitary = gates[step.gate] @ unitary
                axis = record.signature.name[len("rotate_")].upper()
                np.testing.assert_allclose(unitary @ pauli[axis] @ unitary.conj().T, pauli["Z"], rtol=0, atol=1e-15)
                rotated.add(axis)
        assert rotated == {axis for label in labels for axis in label if axis in "XY"}
        assert len(chosen.experiments) == len(rec.groups) + padded < len(labels) + padded
        assert sorted(label for group in rec.groups for label in group) == sorted(labels)
        assert 0 < rec.grouping_comparisons <= len(labels) * len(rec.groups)
        result = solve(chosen)
        ns = len(rec.success_bits)
        for moments in result.groups:
            chunk = next(c for c in result.data.observations.chunks if c.setting == moments.name)
            histogram = chunk.histogram()
            assert moments.returned_shots == chunk.returned_shots == 3000 and histogram.width == ns + 2
            if moments.population == "physical_prefix":
                continue
            bits, counts = histogram.indices(), histogram.weights
            success = (bits & np.uint64((1 << ns) - 1)) == 0
            coefficient = dict((label, c) for label, c in rec.terms)
            masks = [(int(label[::-1].replace("I", "0").replace("X", "1").replace("Y", "1").replace("Z", "1")[::-1], 2)) << ns
                     for label in moments.labels]
            weights = [coefficient[label] for label in moments.labels]
            if padded and moments.name == "group_0":
                masks, weights = masks + [0], weights + [coefficient.get("II", 0.)]
            expected = (weighted_group_moments(bits[success], counts[success], 1., masks, weights) if not padded
                        else weighted_group_moments(bits, counts, success.astype(float), masks, weights))
            assert (moments.mean, moments.second_moment, moments.variance) == expected
            assert moments.selected_shots == int(counts[success].sum())
        parts = [item for item in result.groups if item.population != "physical_prefix"]
        if not padded:
            assert result.value == sum(item.mean for item in parts) + dict(rec.terms).get("II", 0.)
            mass = parts[0].selected_shots/parts[0].returned_shots
        else:
            (prefix,) = [item for item in result.groups if item.population == "physical_prefix"]
            assert prefix.selected_shots > 0 and result.value == math.fsum(item.mean for item in parts)/prefix.mean
            mass = prefix.mean
        assert result.norm_squared == physical_moment(mass, rec.recovery)
    form = solve(plan(LinearDynamics(A=np.diag([.1, .25, .4]), initial_state=[1., 2., 3.], time=.3), method=method,
                      output=QuadraticForm(observable=np.diag([.3, -.2, .1]) + .05 * np.eye(3, k=1) + .05 * np.eye(3, k=-1)),
                      execution="quantum", shots=3000, seed=7))
    assert form.value == physical_moment(math.fsum(item.mean for item in form.groups), form.plan.reconstruction.recovery)


def test_lchs_samples_are_sorted_physical_arrays_that_survive_save_and_load(tmp_path):
    """Samples keep original-coordinate indices and counts as int64 arrays summing to selected_shots."""
    import nwqlib
    from nwqlib import Samples
    problem = LinearDynamics(A=np.diag([.1, .25, .4]), initial_state=[1., 2., 3.], time=.3)
    result = solve(plan(problem, method=_trotter_method(), output=Samples(), execution="quantum", shots=4000, seed=9))
    samples = result.samples
    assert samples.indices.array.dtype == np.int64 and np.all(samples.indices.array < 3)
    assert result.selected_shots == samples.total == sum(map(int, samples.counts.array))
    result.save(tmp_path / "result")
    loaded = nwqlib.load_result(tmp_path / "result")
    assert loaded.samples == samples and loaded.content_id == result.content_id


def test_exact_reduction_and_sampled_grouping_obey_the_readout_work_limit():
    """Exact reduction and sampled grouping refuse work above max_readout_work."""
    import json
    from nwqlib import NormalizedExpectation
    from nwqlib._quantum_readout import projected_requirements
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1)

    def chosen(limit, terms=(("ZI", .5), ("XX", .2)), shots=None):
        method = _trotter_method().revise(max_readout_work=limit)
        output = NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2))
        return plan(problem, method=method, output=output, execution="quantum", shots=shots, seed=7)

    probe = chosen(2_000_000_000)
    point = probe.experiments[0].observation.positions[0]
    width = len(probe.reconstruction.success_bits)+len(probe.reconstruction.system_bits)
    _, work = projected_requirements(json.loads(point.parameters), width)
    # Construction keeps its own ledger; the exact route records no grouping comparison.
    assert probe.reconstruction.grouping_comparisons == 0 and probe.reconstruction.select_work > 1554
    assert solve(chosen(work)).value is not None
    with pytest.raises(ValueError, match=r"LCHS\.max_readout_work"):
        solve(chosen(work - 1))
    # ZI, XX and YY pairwise fail qubit-wise commutation: three groups after 1 + 2 comparisons.
    labels = (("ZI", .5), ("XX", .2), ("YY", .1))
    sampled = chosen(3, labels, shots=100)
    assert sampled.reconstruction.grouping_comparisons == 3 and len(sampled.reconstruction.groups) == 3
    # The 15 two-qubit labels also exceed a single grouping comparison.
    labels = tuple((a + b, 1 / (1 + i)) for i, (a, b) in enumerate(
        (a, b) for a in "IXYZ" for b in "IXYZ" if a + b != "II"))
    with pytest.raises(ValueError, match=r"LCHS\.max_readout_work=1\b"):
        chosen(1, labels, shots=100)


def test_wide_exact_readout_obeys_its_work_limit_before_acquisition():
    """A width-20 periodic Plan with 128 observable terms fits the default readout cap.

    At an explicit ``max_readout_work`` of 100,000,000 the hook refuses before
    acquisition. The check uses the planned readout without allocating a state.
    """
    import json
    from types import SimpleNamespace
    from nwqlib import NormalizedExpectation
    from nwqlib._quantum_readout import projected_requirements
    from nwqlib.operators import ingest_pauli
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_occupation
    n = 18
    method = LCHS(k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
        parameters={"num_qubits": 2, "lsb_position": 0}), lchs_kernel="cauchy_density",
        trotter_steps=None, trotter_synthesis_tolerance=1e-4, hamiltonian_evolution_backend="trotter")
    terms = tuple(("".join("IXYZ"[(j >> (2*k)) & 3] for k in reversed(range(n))), .5) for j in range(1, 129))
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=n, mass=1., diffusion=.25, potential=.3),
        initial_state=ingest_occupation((0,) * n, num_qubits=n), time=.5)
    chosen = plan(problem, method=method, output=NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=n)),
                  execution="quantum", seed=7)
    observation = chosen.experiments[0].observation
    point = chosen.resolve(chosen.experiments[0].name)
    run = SimpleNamespace(observations=SimpleNamespace(chunks=()))
    work = projected_requirements(json.loads(observation.positions[0].parameters), 20)[1]
    assert 100_000_000 < work < method.max_readout_work
    assert method.reduction_allowance(chosen, point, observation=observation, width=20, run=run) == 2_000_000_000
    narrow = method.revise(max_readout_work=100_000_000)
    with pytest.raises(ValueError, match=r"LCHS\.max_readout_work"):
        narrow.reduction_allowance(chosen, point, observation=observation, width=20, run=run)


def test_saved_mass_branches_disagree_on_a_near_unit_complete_norm():
    """The delta and window branches of ``validate_saved_masses`` can disagree.

    A complete squared norm of 1 - 5e-13 on an eight-entry state passes the
    window convention omega = 1e-12 and fails delta = 0, whose host allowance
    is of order 1e-15 (``validate_saved_masses``). Which branch acquisition
    and publication take is checked on a receipt in
    ``test_observation_schedule.py``
    (``test_projected_reducer_resolves_only_the_amplitude_mass_label_at_every_mass_checkpoint``).
    """
    import math
    import nwqlib._quantum_readout as readout
    near = math.frexp(1 - 5e-13)
    readout.validate_saved_masses(near, near, near, (8, 4, 4), window=1e-12)
    readout.validate_saved_masses(near, near, near, (8, 4, 4), delta=1e-12, window=1e-12)
    with pytest.raises(ValueError, match="outside the producing window"):
        readout.validate_saved_masses(near, near, near, (8, 4, 4), delta=0., window=1e-12)
