"""Compact periodic selection, actual phase/Strang action and saved-data association."""
from math import hypot,inf,nextafter,pi
from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest
from nwqlib import LinearDynamics,NormSquared,QuadraticForm,plan,solve
from nwqlib.algorithms.lchs import LCHS,ProviderConfig
from nwqlib.algorithms.lchs.periodic import select_periodic_parameters
from nwqlib.operators.inputs import PeriodicStencil,ingest_pauli,operator_input
from nwqlib.problems.inputs import ingest_occupation,ingest_product


def periodic_plan(*,observable='Z',product=False,num_qubits=2):
    """Plan compact periodic dynamics with four Cauchy nodes and two Strang steps.

    The output is the physical ||u||^2 when observable is None, otherwise u-dagger (I...I P) u.
    """
    generator=PeriodicStencil(num_qubits=num_qubits,mass=1.,diffusion=.25,potential=.3)
    if product:
        pairs=np.array([[1.,0.]]*num_qubits,dtype=complex)
        pairs[0]=[2**-.5,2**-.5]
        initial=ingest_product(pairs)
    else:
        initial=ingest_occupation((0,)*num_qubits,num_qubits=num_qubits)
    output=NormSquared() if observable is None else QuadraticForm(
        observable=ingest_pauli([('I'*(num_qubits-1)+observable,1.)],num_qubits=num_qubits))
    method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=2,lchs_kernel='cauchy_density',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':0}))
    return plan(LinearDynamics(A=generator,initial_state=initial,time=.5),method=method,output=output)


def select(*,q=2,steps=2,options=None):
    operator=operator_input(PeriodicStencil(num_qubits=q,mass=1.,diffusion=.25,potential=.3))
    method=options or LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=steps,
        lchs_kernel='cauchy_density',k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':2,'lsb_position':0}))
    return select_periodic_parameters(operator,elapsed=.5,method=method)

def test_periodic_scalar_selection_has_no_system_expansion(monkeypatch):
    from nwqlib.operators.inputs import OperatorInput
    from nwqlib.problems.inputs import StateInput
    def forbidden(*args, **kwargs):
        pytest.fail("compact periodic selection attempted a system expansion or eigensolve")
    monkeypatch.setattr(OperatorInput, "dense_array", forbidden)
    monkeypatch.setattr(OperatorInput, "pauli_terms", forbidden)
    monkeypatch.setattr(StateInput, "physical_vector", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    narrow, _ = select()
    wide, wide_counts = select(q=80)  # Explicitly bounded metadata/table-only witness.
    assert narrow.coefficient_plan.nodes == wide.coefficient_plan.nodes == (0., 1., -2., -1.)
    assert narrow.coefficient_plan.coefficients == wide.coefficient_plan.coefficients
    assert narrow.amplitudes.shape == wide.amplitudes.shape == (4,)
    assert narrow.parameter_id != wide.parameter_id
    # Cauchy weights are [1,1/2,1/5,1/2]/pi, normalized by 11/(5*pi).
    np.testing.assert_allclose(np.abs(narrow.amplitudes)**2, [5/11, 5/22, 1/11, 5/22], rtol=5e-16, atol=0.)
    assert narrow.half_angles == (0., -1/16, 1/8, 1/16)
    assert narrow.phases == (0., -.75, 1.5, .75)
    # Direct finite sum: T^3/(3r^2) * [(.3)^3+(.8)^3+(1.3)^3/5]/pi.
    assert narrow.synthesis_bound == pytest.approx(1223 / (120000 * pi), rel=8e-16, abs=0.)
    assert wide_counts["construction_work"] < 100_000_000
    with pytest.raises(ValueError):
        narrow.amplitudes.setflags(write=True)
    # A different actual kernel must keep its complex coefficient phases
    # and finite weighted bound; no generic Cauchy/asymptotic replacement.
    from cmath import exp as cexp
    options = LCHS( hamiltonian_evolution_backend="trotter", trotter_steps=2,
        lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta":.5}),
        k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
            parameters={"num_qubits":2,"lsb_position":0}))
    alternate, _ = select(options=options)
    nodes = (0., 1., -2., -1.)
    normalization = 2*pi*cexp(-2**.5)
    coefficients = tuple(cexp(-(1+1j*k)**.5)/normalization/(1-1j*k) for k in nodes)
    # A few complex power/exponential/phase operations: 32 binary64 ulps.
    tolerance = 32 * np.finfo(float).eps
    np.testing.assert_allclose(alternate.coefficient_plan.coefficients, coefficients, rtol=tolerance, atol=0.)
    np.testing.assert_allclose([cexp(1j*value) for value in alternate.phases],
        [c/abs(c)*cexp(-.75j*k) for c,k in zip(coefficients,nodes,strict=True)], rtol=tolerance, atol=0.)
    expected = sum(abs(c)*(.5*abs(k)+.3)**3 for c,k in zip(coefficients,nodes,strict=True))/96
    assert alternate.synthesis_bound == pytest.approx(expected, rel=tolerance, abs=0.)


def test_periodic_strang_step_cx_laws_count_the_built_step():
    from qiskit import QuantumCircuit,transpile
    chosen=periodic_plan(observable=None,num_qubits=2)
    leaf=next(block for block in chosen.blocks if block.record.signature.name=='periodic_step')
    laws={(law.controlled,law.adjoint):law.value for law in leaf.record.resource_laws}
    step=leaf._constructor(leaf,None,None)
    def cx(circuit):
        return transpile(circuit,basis_gates=['cx','u'],optimization_level=0).count_ops()['cx']
    for adjoint,circuit in ((False,step),(True,step.inverse())):
        # Unoptimized unrolling keeps every slot. For two address and two system
        # qubits: three UCRZ tables of 4 CX, and two cyclic shifts, each a QFT
        # pair with one CP (2 CX) and one SWAP (3 CX) per QFT: 12+2*2*5=32.
        assert cx(circuit)==laws[False,adjoint]
        controlled=QuantumCircuit(circuit.num_qubits+1)
        controlled.append(circuit.to_gate().control(1),controlled.qubits)
        # A multiplexor emits no rotation for a zero Walsh angle, so the
        # one-control slot law is an upper bound on the controlled step.
        assert cx(controlled)<=laws[True,adjoint]


@pytest.mark.parametrize('automatic,binding', [(True, 'work'), (False, 'bytes')])
def test_periodic_construction_limits_refuse_the_selected_workload(automatic, binding):
    method = LCHS(hamiltonian_evolution_backend='trotter',
        trotter_steps=None if automatic else 12,
        trotter_synthesis_tolerance=1e-4 if automatic else None,
        lchs_kernel='cauchy_density', k_quadrature=ProviderConfig(
            implementation='signed_binary_uniform', parameters={'num_qubits': 2, 'lsb_position': 0}))
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=1., diffusion=.25, potential=.3),
        initial_state=ingest_occupation((0, 0), num_qubits=2), time=.5)
    chosen = plan(problem, method=method, output=NormSquared(), seed=7)
    limits = {}
    if binding == 'work':
        # _construction_law(2, 2, r) charges 969*r + 486 work units, so this limit admits at most 5 steps.
        limits['max_select_work'] = 5331
    if binding == 'bytes':
        limits['max_bytes'] = 4096  # Admits the four coefficient nodes, but not one step's circuit slots.
    assert chosen.reconstruction.step_counts
    with pytest.raises(ValueError, match="periodic Strang construction.*max_select_work=.*max_bytes="):
        plan(problem, method=method.revise(**limits), output=NormSquared(), seed=7)


def test_periodic_step_cap_refuses_the_required_synthesis():
    method = LCHS(hamiltonian_evolution_backend='trotter', trotter_steps=None,
        trotter_synthesis_tolerance=1e-4, max_trotter_steps=8,
        lchs_kernel='cauchy_density', k_quadrature=ProviderConfig(
            implementation='signed_binary_uniform', parameters={'num_qubits': 2, 'lsb_position': 0}))
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=1., diffusion=.25, potential=.3),
        initial_state=ingest_occupation((0, 0), num_qubits=2), time=.5)
    # The independent B/r**2 calculation in the next test selects 12 steps.
    with pytest.raises(ValueError) as refused:
        plan(problem, method=method, output=NormSquared())
    message = str(refused.value)
    assert '12 steps' in message and 'max_trotter_steps=8' in message
    admitted = plan(problem, method=method.revise(max_trotter_steps=12), output=NormSquared())
    assert admitted.reconstruction.step_counts == (12,) * 4



def test_periodic_synthesis_tolerance_selects_the_smallest_strang_step_count():
    from nwqlib.algorithms.protocol import ApplicabilityError
    # periodic_plan's grid: the one-step coefficient of select_periodic_parameters is
    # B = T^3/3 * [(.3)^3 + 2*(.8)^3/2 + (1.3)^3/5]/pi = 1223/(30000*pi) at T = .5.
    # B/r^2 <= 1e-4 first holds at r = 12: B/144 = 9.0e-5 and B/121 = 1.07e-4.
    method=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=None,trotter_synthesis_tolerance=1e-4,
        lchs_kernel='cauchy_density',k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':2,'lsb_position':0}))
    problem=LinearDynamics(A=PeriodicStencil(num_qubits=2,mass=1.,diffusion=.25,potential=.3),
        initial_state=ingest_occupation((0,0),num_qubits=2),time=.5)
    chosen=plan(problem,method=method,output=NormSquared())
    assert chosen.reconstruction.step_counts==(12,)*4
    bound=next(f.fact.value.value for f in chosen.facts if f.fact.quantity=='trotter_synthesis')
    assert bound<=1e-4<1223/(30000*pi)/11**2
    # With diffusion 1e100, T = 1e-110 and the default kernel, B is about 7.9e-28, while the binary64 T**3
    # is 0, so B must be formed exactly. The recorded bound is positive and at most the tolerance, and in
    # exact arithmetic, with each |c_j| enclosed between rationals, B/r**2 <= 1e-34 < B/(r - 1)**2.
    extreme=LCHS(hamiltonian_evolution_backend='trotter',trotter_steps=None,trotter_synthesis_tolerance=1e-34)
    stencil=PeriodicStencil(num_qubits=2,mass=1.,diffusion=1e100)
    chosen=plan(LinearDynamics(A=stencil,initial_state=ingest_occupation((0,0),num_qubits=2),time=1e-110),
        method=extreme,output=NormSquared())
    bound=next(f.fact.value.value for f in chosen.facts if f.fact.quantity=='trotter_synthesis')
    (steps,)=set(chosen.reconstruction.step_counts)
    payload,_=select_periodic_parameters(operator_input(stencil),elapsed=1e-110,method=extreme)
    low=high=Fraction(0)
    for k,c in zip(payload.coefficient_plan.nodes,payload.coefficient_plan.coefficients,strict=True):
        c=complex(c)
        square=Fraction(c.real)**2+Fraction(c.imag)**2
        below=above=hypot(c.real,c.imag)
        for _ in range(2):
            below,above=nextafter(below,0.),nextafter(above,inf)
        assert Fraction(below)**2<=square<=Fraction(above)**2
        cube=(2*abs(Fraction(k))*Fraction(stencil.diffusion))**3
        low,high=low+Fraction(below)*cube,high+Fraction(above)*cube
    scale,tolerance=Fraction(1e-110)**3/3,Fraction(1e-34)
    assert steps==payload.steps>1 and 0<bound<=1e-34
    assert scale*high/steps**2<=tolerance<scale*low/(steps-1)**2
    # A fixed count and a tolerance cannot both be given, and a dense A has no
    # planning bound for the tolerance to invert.
    with pytest.raises(ValueError,match='exactly one of trotter_steps'):
        method.revise(trotter_steps=12)
    with pytest.raises(ApplicabilityError,match='PeriodicStencil'):
        plan(LinearDynamics(A=[[.4,.15],[.05,.25]],initial_state=[1,0],time=.1),method=method)


@pytest.mark.parametrize('backend,source',[('dense_exact',False),('dense_exact',True),('trotter',False),('trotter',True),('qsp_block_encoding',False),('qsp_block_encoding',True)])
def test_selected_archive_keeps_actual_products_without_reselection(backend,source,tmp_path,monkeypatch):
    from nwqlib._choice_archive import ArchiveFiles,save_plan,load_plan
    from nwqlib.algorithms.lchs import selection,parameters,native
    from nwqlib.subroutines.qsp import evolution
    from nwqlib.subroutines.state_preparation import mps
    problem=LinearDynamics(A=[[.3,.05j],[.05j,.5]],initial_state=[1,2j],source=[.2j,.1] if source else None,time=.1)
    chosen=plan(problem,method=LCHS(hamiltonian_evolution_backend=backend,duhamel_nodes=1,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})))
    assert sum(r.width for r in chosen.construction.program.registers)<=10
    files=ArchiveFiles(tmp_path,8_000_000)
    saved=save_plan(chosen,files)
    def forbidden(*a,**kw):
        pytest.fail('saved selection repeated numerical or native work')
    monkeypatch.setattr(selection,'select_dense',forbidden)
    monkeypatch.setattr(parameters,'select_parameters',forbidden)
    monkeypatch.setattr(native,'construct_select',forbidden)
    monkeypatch.setattr(evolution,'prepare_qsp_evolution',forbidden)
    monkeypatch.setattr(mps,'_decompose_normalized_state',forbidden)
    for name in ('svd','eigvalsh','eigh','norm'):
        monkeypatch.setattr(np.linalg,name,forbidden)
    loaded=load_plan(saved,files)
    assert loaded==chosen
    original,restored=chosen._native['native_data'],loaded._native['native_data']
    assert restored.method is loaded.method
    np.testing.assert_array_equal(restored.quadrature.coefficients,original.quadrature.coefficients)
    assert not restored.quadrature.coefficients.flags.writeable
    if source:
        np.testing.assert_array_equal(restored.source_layout.elapsed_times,original.source_layout.elapsed_times)
        assert restored.source_layout.branch_kinds==original.source_layout.branch_kinds
    if backend=='qsp_block_encoding':
        assert restored.qsp_plan['prepared_evolution']==original.qsp_plan['prepared_evolution']
    elif backend=='trotter':
        assert restored.select_data.plan==original.select_data.plan
    # Planning binds the coefficient plan of the native data only without a
    # source, and loading binds the same object.
    assert loaded._native['coefficient_plan'] is (None if source else restored.coefficient_plan)
    # The rebound SELECT and PREP leaves build the same circuits, so the
    # exact statevector readout of the loaded Plan repeats the original. The
    # Plan loaded above bound the forbidden constructors, so load it again.
    monkeypatch.undo()
    again=load_plan(saved,ArchiveFiles(tmp_path,8_000_000))
    np.testing.assert_array_equal(solve(again).solution,solve(chosen).solution)


def test_periodic_archive_preserves_plan_and_solution(tmp_path):
    """The periodic Plan and its computed value survive a normal save and load."""
    import json
    from nwqlib._choice_archive import ArchiveFiles,save_plan,load_plan
    chosen=periodic_plan()
    files=ArchiveFiles(tmp_path,10**8)
    saved=save_plan(chosen,files)
    loaded=load_plan(json.loads(json.dumps(saved)),ArchiveFiles(tmp_path,10**8))
    assert loaded==chosen
    assert solve(loaded).value==solve(chosen).value


def test_periodic_public_compact_plan_preserves_actual_payload_and_rejects_foreign_association(monkeypatch):
    from nwqlib.algorithms.lchs.selected_grid import selected_payload
    from nwqlib.operators.inputs import OperatorInput
    from nwqlib.problems.inputs import StateInput
    def forbidden(*a,**kw):
        pytest.fail('compact planning expanded the system')
    monkeypatch.setattr(OperatorInput,'dense_array',forbidden)
    monkeypatch.setattr(StateInput,'physical_vector',forbidden)
    monkeypatch.setattr(np.linalg,'eigvalsh',forbidden)
    for observable,product,q in (('Y',False,2),('X',True,2),(None,False,80)):
        chosen=periodic_plan(observable=observable,product=product,num_qubits=q)
        payload=selected_payload(chosen)
        rec=chosen.reconstruction
        assert rec.success_bits==(0,1) and rec.system_bits==tuple(range(2,q+2))
        assert rec.selected_select=='periodic_strang' and rec.psd_premise=='analytic_periodic'
        assert payload.operator is chosen.problem.A and payload.amplitudes.shape==(4,)
        assert rec.quadrature_bound is rec.kernel_approximation_bound is None
        chosen._native['native_data']=replace(payload)
        with pytest.raises(ValueError,match='exact selected phase'):
            selected_payload(chosen)


def test_periodic_one_qubit_double_wrap_matches_independent_two_edge_evolution():
    chosen=periodic_plan(observable=None,num_qubits=1)
    payload=chosen._native['native_data']
    assert sum(r.width for r in chosen.construction.program.registers)==3
    identity=np.eye(2,dtype=complex)
    x=np.array([[0,1],[1,0]],dtype=complex)
    rz=np.diag(np.exp(-.5j*.5/2*.3*np.array([1,-1])))
    expected=np.zeros(2,dtype=complex)
    for k,c in zip(payload.coefficient_plan.nodes,payload.coefficient_plan.coefficients,strict=True):
        # At q=1 both wrap edges are X. The three matching factors add to
        # exp(+i*2*k*diffusion*dt*X); dropping one wrap changes this action.
        theta=.5/2*k*.25
        half=np.cos(theta/2)*identity+1j*np.sin(theta/2)*x
        full=np.cos(theta)*identity+1j*np.sin(theta)*x
        step=rz@half@full@half@rz
        expected+=c*np.exp(-.5j*k*1.5)*np.linalg.matrix_power(step,2)[:,0]
    result=solve(chosen)
    assert result.value==pytest.approx(np.vdot(expected,expected).real,abs=2e-13)
