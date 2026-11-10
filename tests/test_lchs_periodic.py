"""Compact periodic selection, actual phase/Strang action and saved-data association."""
from math import hypot,inf,nextafter,pi
from fractions import Fraction
import numpy as np
import pytest
from nwqlib import LinearDynamics,NormSquared,QuadraticForm,plan,solve
from nwqlib.algorithms.lchs import LCHS,ProviderConfig,Trotter
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
    method=LCHS(ham_evo=Trotter(steps=2),lchs_kernel='cauchy_density',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':0}))
    return plan(LinearDynamics(A=generator,initial_state=initial,time=.5),method=method,output=output)


def select(*,q=2,steps=2,options=None):
    operator=operator_input(PeriodicStencil(num_qubits=q,mass=1.,diffusion=.25,potential=.3))
    method=options or LCHS(ham_evo=Trotter(steps=steps),
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
    np.testing.assert_array_equal(narrow.coefficient_plan.nodes, (0., 1., -2., -1.))
    np.testing.assert_array_equal(wide.coefficient_plan.nodes, (0., 1., -2., -1.))
    np.testing.assert_array_equal(narrow.coefficient_plan.coefficients, wide.coefficient_plan.coefficients)
    assert narrow.amplitudes.shape == wide.amplitudes.shape == (4,)
    assert narrow.parameter_id != wide.parameter_id
    # Cauchy weights are [1,1/2,1/5,1/2]/pi, normalized by 11/(5*pi).
    np.testing.assert_allclose(np.abs(narrow.amplitudes)**2, [5/11, 5/22, 1/11, 5/22], rtol=5e-16, atol=0.)
    assert narrow.half_angles == (0., -1/16, 1/8, 1/16)
    assert narrow.phases == (0., -.75, 1.5, .75)
    # select_periodic_parameters: q2 has only the cross terms, giving 111/(64000*pi).
    # q80 adds T^3/r^2 * [(1/4)^3+(1/2)^3/5]/(2*pi) = 13/(20480*pi).
    assert (narrow.synthesis_bound, wide.synthesis_bound) == pytest.approx(
        (111 / (64000 * pi), 1213 / (512000 * pi)), rel=8e-16, abs=0.)
    assert wide_counts["construction_work"] < 100_000_000
    with pytest.raises(ValueError):
        narrow.amplitudes.setflags(write=True)
    # A different actual kernel must keep its complex coefficient phases
    # and finite weighted bound; no generic Cauchy/asymptotic replacement.
    from cmath import exp as cexp
    options = LCHS( ham_evo=Trotter(steps=2),
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
    expected = sum(abs(c)*(4*(abs(k)/4)**2*.3/3+(abs(k)/4)*.3**2/3)
        for c,k in zip(coefficients,nodes,strict=True))/32
    assert alternate.synthesis_bound == pytest.approx(expected, rel=tolerance, abs=0.)
    # mass = 2**57 makes the identity phase -T*k*mass as large as 2**57
    # radians, whose binary64 spacing is 32. Each address phase must still be
    # arg(c_j) - T*k_j*mass modulo 2*pi, the huge angle reduced by cexp
    # against pi itself; adding arg(c_j) to the unreduced angle loses it.
    heavy, _ = select_periodic_parameters(operator_input(PeriodicStencil(num_qubits=2, mass=2.**57, diffusion=0.,
        potential=.3)), elapsed=.5, method=options)
    np.testing.assert_allclose([cexp(1j*value) for value in heavy.phases],
        [c/abs(c)*cexp(-.5j*k*2.**57) for c, k in zip(heavy.coefficient_plan.coefficients,
                                                       heavy.coefficient_plan.nodes, strict=True)],
        rtol=tolerance, atol=0.)


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


def test_periodic_resource_breakdown_counts_the_repeated_strang_step():
    """The resource breakdown reads the repeated Strang step as Trotter.steps calls, as estimate does.

    The exact-reduction body calls the initial PREP, the coefficient PREP pair and the phase leaf once and the
    Strang step under a repetition of 2. The occupation PREP carries its exact gates, which hold no CX, and
    every other block one CX law, so calls times laws summed over the body equal estimate's CX count, which
    folds the same Program.
    """
    from nwqlib import estimate
    from nwqlib.algorithms.lchs.quantum_resources import resource_breakdown
    from nwqlib.resources import ResourceContext
    chosen=periodic_plan(observable=None,num_qubits=2)
    (entry,)=resource_breakdown(chosen)
    rows={row['block']:row for row in entry['blocks']}
    assert entry['repetitions'] is None and rows['periodic_step']['calls']==2
    assert all(rows[name]['calls']==1 for name in ('initial_prep','coefficient_prep','coefficient_inverse','periodic_phase'))
    assert rows['initial_prep']['laws']==() and all(gate.gate!='cx' for gate in rows['initial_prep']['decomposition'])
    total=sum(row['calls']*row['laws'][0].value for row in entry['blocks'] if row['laws'])
    cx=estimate(chosen,context=ResourceContext(basis='cx')).quantity('cx').fact.value
    assert (cx.numerator,cx.denominator)==(total,1) and total==2+2+2*32+2


def test_periodic_construction_limits_refuse_the_selected_workload():
    method = LCHS(ham_evo=Trotter(steps=None, synthesis_tol=1e-4),
        lchs_kernel='cauchy_density', k_quadrature=ProviderConfig(
            implementation='signed_binary_uniform', parameters={'num_qubits': 2, 'lsb_position': 0}))
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=1., diffusion=.25, potential=.3),
        initial_state=ingest_occupation((0, 0), num_qubits=2), time=.5)
    chosen = plan(problem, method=method, output=NormSquared(), seed=7)
    # _construction_law(2, 2, r) charges 969*r + 486 work units, so this limit admits at most 4 steps.
    limits = {'max_select_work': 4362}
    assert chosen.reconstruction.step_counts
    with pytest.raises(ValueError, match="periodic Strang construction.*max_select_work="):
        plan(problem, method=method.revise(**limits), output=NormSquared(), seed=7)
    # The tolerance selects r = 5 here (B/25 <= 1e-4 < B/16, see the planning test below), so a step limit of 4
    # refuses the selected count and 5 admits it.
    from nwqlib._limits import ResourceLimitError
    with pytest.raises(ResourceLimitError, match="needs 5 steps .*Trotter.max_steps=4"):
        plan(problem, method=method.revise(ham_evo=Trotter(
            steps=None, synthesis_tol=1e-4, max_steps=4)), output=NormSquared(), seed=7)
    plan(problem, method=method.revise(ham_evo=Trotter(
        steps=None, synthesis_tol=1e-4, max_steps=5)), output=NormSquared(), seed=7)


def test_periodic_plan_and_estimate_stay_available_and_preparation_checks_the_circuit_first(monkeypatch, tmp_path):
    """Planning a periodic stencil forms no circuit, so only prepare compares the circuit with memory.

    Relations: one Strang step holds 2q(q-1) cp, 4*floor(q/2) swap, 6+4q h, 3p
    cx and 3p+2+2q rz at top level on Aer, with p = 2**a (periodic._leaf_memory).
    A small built circuit is compared with that count, without pinning the count.
    """
    import re
    import nwqlib._limits as limits
    from nwqlib import estimate, prepare
    from nwqlib.blocks import lowering
    from nwqlib.resources import ResourceContext

    def stencil(q, time, **options):
        method = LCHS(ham_evo=Trotter(steps=None, synthesis_tol=.005),
                      approx_tol=.005, **options)
        problem = LinearDynamics(A=PeriodicStencil(num_qubits=q, mass=0., diffusion=.25),
                                 initial_state=ingest_occupation('0'*q, num_qubits=q), time=time)
        return plan(problem, method=method, output=NormSquared(), seed=7)

    # Formula workloads whose circuits would not fit: the plan and its estimate cost no such memory.
    for q, time, options in ((100, 1., {}), (40, 20., dict(max_bytes=10**9))):
        estimate(stencil(q, time, **options), context=ResourceContext(basis='cx'))

    def per_step(q, a):
        p = 1 << a
        return 2*q*(q-1) + 4*(q//2) + 6+4*q + 3*p + 3*p+2+2*q

    monkeypatch.setattr(lowering, '_ConstructorContext', lambda *a, **k: pytest.fail('circuit was constructed'))
    # About 80 GB of circuit on a 36 GiB computer, and a declared limit below 2.4 GB.
    for q, time, options, physical, clause in ((10, 60., {}, 36*2**30, 'the physical memory'),
                                               (10, 10., dict(max_bytes=10**8), None, 'max_bytes')):
        chosen = stencil(q, time, **options)
        if physical:
            monkeypatch.setattr(limits, 'physical_memory_bytes', lambda: physical)
        with pytest.raises(ValueError, match=clause) as refused:
            prepare(chosen)
        needed = int(re.search(r'needs about (\d+) bytes', str(refused.value)).group(1))
        steps = chosen.reconstruction.step_counts[0]
        a = (chosen.reconstruction.padded_branches-1).bit_length()
        assert 1024*steps*per_step(q, a) <= needed <= 1.01*1024*steps*per_step(q, a)
    monkeypatch.undo()

    # Actual Aer positions of a small circuit, within the factor two the estimate aims at.
    chosen = periodic_plan(observable=None, num_qubits=3)
    a = (chosen.reconstruction.padded_branches-1).bit_length()
    steps = chosen.reconstruction.step_counts[0]
    phase = (1 << a)-1 + max(0, (1 << a)-2)
    leaves = {block.record.signature.name: block for block in chosen.blocks}
    assert leaves['periodic_step']._native_bytes == 1024*per_step(3, a)
    assert leaves['periodic_phase']._native_bytes == 1024*phase
    positions = len(prepare(chosen, progress=False).circuits[0].data)
    assert positions/2 <= steps*per_step(3, a)+phase <= 2*positions

    # A saved and reloaded Plan keeps the same comparison for its next preparation.
    from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
    files = ArchiveFiles(tmp_path, 10**8)
    loaded = load_plan(save_plan(chosen, files), ArchiveFiles(tmp_path, 10**8))
    assert ({block.record.signature.name: (block._native_bytes, block._definition_bytes) for block in loaded.blocks}
            == {block.record.signature.name: (block._native_bytes, block._definition_bytes) for block in chosen.blocks})


def test_periodic_preparation_checks_the_trees_of_its_coefficient_prep_and_inverse_against_memory(monkeypatch):
    """A Run emits the three-step periodic circuit only if the trees of its two-qubit coefficient PREP and inverse fit in memory.

    The circuit opens with the initial state's preparation, so even noiseless
    Aer sets neither as its state, and each tree counts 4*2**2 + 2*2 = 20
    instructions of 256 bytes. Those trees alone are compared first, and the
    whole circuit's estimate follows them, including its Strang steps.
    """
    import nwqlib._limits as limits
    from nwqlib import prepare
    from nwqlib.blocks import lowering

    method = LCHS(ham_evo=Trotter(steps=3), lchs_kernel='cauchy_density',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
                                    parameters={'num_qubits': 2, 'lsb_position': 0}))
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=1., diffusion=.25, potential=.3),
        initial_state=ingest_occupation((0, 0), num_qubits=2), time=.5)
    chosen = plan(problem, method=method, output=NormSquared(), execution='quantum')

    class Emitted(Exception):
        pass

    def emission(*args, **kwargs):
        raise Emitted

    monkeypatch.setattr(lowering, '_ConstructorContext', emission)
    monkeypatch.setattr(limits, 'physical_memory_bytes', lambda: 5120)
    with pytest.raises(ValueError, match=r'direct state preparations of this circuit needs about 10240 bytes, '
                                         r'exceeding the physical memory \(5120 bytes\)'):
        prepare(chosen)
    monkeypatch.setattr(limits, 'physical_memory_bytes', lambda: 10240)
    with pytest.raises(ValueError, match=r'preparing this circuit needs about \d+ bytes'):
        prepare(chosen)
    monkeypatch.setattr(limits, 'physical_memory_bytes', lambda: 10**6)
    with pytest.raises(Emitted):
        prepare(chosen)


def test_periodic_synthesis_tolerance_selects_the_smallest_strang_step_count():
    from nwqlib.algorithms.protocol import ApplicabilityError
    # periodic_plan's grid: the one-step coefficient of select_periodic_parameters is
    # B = T^3*sum |c_j|*(4*a_j^2*p/3+a_j*p^2/3) = 111/(16000*pi) at T = .5.
    # B/r^2 <= 1e-4 first holds at r = 5: B/25 = 8.83e-5 and B/16 = 1.38e-4.
    method=LCHS(ham_evo=Trotter(steps=None,synthesis_tol=1e-4),
        lchs_kernel='cauchy_density',k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':2,'lsb_position':0}))
    problem=LinearDynamics(A=PeriodicStencil(num_qubits=2,mass=1.,diffusion=.25,potential=.3),
        initial_state=ingest_occupation((0,0),num_qubits=2),time=.5)
    chosen=plan(problem,method=method,output=NormSquared())
    assert chosen.reconstruction.step_counts==(5,)*4
    bound=next(f.fact.value.value for f in chosen.facts if f.fact.quantity=='trotter_synthesis')
    assert bound<=1e-4<111/(16000*pi)/4**2
    # With q3, diffusion 1e100, T = 1e-110 and the default kernel, B is about 1.5e-28, while binary64 T**3
    # is 0, so B must be formed exactly. The recorded bound is positive and at most the tolerance, and in
    # exact arithmetic, with each |c_j| enclosed between rationals, B/r**2 <= 1e-34 < B/(r - 1)**2.
    extreme=LCHS(ham_evo=Trotter(steps=None,synthesis_tol=1e-34))
    stencil=PeriodicStencil(num_qubits=3,mass=1.,diffusion=1e100)
    chosen=plan(LinearDynamics(A=stencil,initial_state=ingest_occupation((0,0,0),num_qubits=3),time=1e-110),
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
        cube=(abs(Fraction(k))*Fraction(stencil.diffusion))**3
        low,high=low+Fraction(below)*cube,high+Fraction(above)*cube
    scale,tolerance=Fraction(1e-110)**3/2,Fraction(1e-34)
    assert steps==payload.steps>1 and 0<bound<=1e-34
    assert scale*high/steps**2<=tolerance<scale*low/(steps-1)**2
    # A declared input norm 2**1100 lies outside binary64, but the physical
    # one-step bound norm*T**3*sum|c_j|*(|k_j|*D)**3/2 for pure diffusion
    # D = 2**-200, T = 1 and the four-node Cauchy grid c_j = .5/(pi*(1+k_j**2)),
    # k_j in {0, .5, -1, -.5}, is (.175/pi)*2**500. The stored coefficients
    # and the upward rounding of B change it by a few units in the last place.
    from qiskit import QuantumCircuit
    from nwqlib import StateVector
    from nwqlib.core.records import InputRef,Source
    from nwqlib.problems.inputs import PhysicalScale,bind_preparation_circuit
    operator=operator_input(PeriodicStencil(num_qubits=3,mass=0.,diffusion=2.**-200,potential=0.))
    reference=InputRef(identity='test:periodic-scaled',representation='circuit',
        source=Source(name='test',version='1',domain='test',reference='declared'))
    state=bind_preparation_circuit(QuantumCircuit(3),reference=reference,basis=operator.basis,
        norm=PhysicalScale(mantissa=.5,exponent=1101))
    grid=LCHS(ham_evo=Trotter(steps=1),lchs_kernel='cauchy_density',
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1}))
    chosen=plan(LinearDynamics(A=operator,initial_state=state,time=1.),method=grid,
        output=StateVector(normalization='unit'))
    strang=next(f.fact for f in chosen.facts if f.fact.quantity=='trotter_synthesis')
    assert strang.availability=='concrete' and strang.value.value==pytest.approx(.175/pi*2.**500,rel=1e-14,abs=0)
    # A tolerance alone means steps=None and gives the same record. A fixed
    # count and a tolerance cannot both be given, and a dense A has no
    # planning bound for the tolerance to invert.
    alone=Trotter(synthesis_tol=1e-4)
    assert alone==method.ham_evo and alone.content_id==method.ham_evo.content_id
    with pytest.raises(ValueError,match='exactly one of steps'):
        method.ham_evo.revise(steps=12)
    with pytest.raises(ValueError,match='exactly one of steps'):
        Trotter(steps=12,synthesis_tol=1e-4)
    with pytest.raises(ApplicabilityError,match='PeriodicStencil'):
        plan(LinearDynamics(A=[[.4,.15],[.05,.25]],initial_state=[1,0],time=.1),method=method)


@pytest.mark.parametrize('q',(2,3))
def test_periodic_solution_matches_the_dense_exponential_within_planned_bounds(q):
    from scipy.linalg import expm
    # The PeriodicStencil operator m I + nu (2I - S - S^T) + i p Z_0, built densely with qubit 0 as the
    # least significant index bit. The product initial state has no coordinate symmetry, so a misplaced
    # qubit, a reversed potential sign or a lost mass factor moves the solution by more than the bounds.
    mass,diffusion,potential,elapsed=.1,.25,.3,1.
    pairs=[[np.cos(.3+.4*j),np.sin(.3+.4*j)*np.exp(.2j*j)] for j in range(q)]
    initial=np.ones(1)
    for pair in pairs:
        initial=np.kron(pair,initial)
    n=2**q
    identity=np.eye(n)
    shift=np.roll(identity,1,axis=0)
    z0=np.diag(1.-2*(np.arange(n)&1))
    dense=mass*identity+diffusion*(2*identity-shift-shift.T)+1j*potential*z0
    problem=LinearDynamics(A=PeriodicStencil(num_qubits=q,mass=mass,diffusion=diffusion,potential=potential),
        initial_state=ingest_product(pairs),time=elapsed)
    chosen=plan(problem,method=LCHS(ham_evo=Trotter(steps=None,synthesis_tol=.01),
        approx_tol=.02),seed=7)
    assert sum(register.width for register in chosen.construction.program.registers)<=11
    result=solve(chosen)
    # Method-error-bounded class: for this unit initial state the planned kernel, k-quadrature, Strang and
    # preparation bounds together bound the L2 error of the physical vector before circuit rounding
    # (docs/algorithms/lchs.md, Error budget for heat flow on a ring). The added 1e-8 is an allowance for
    # statevector rounding, not a derived bound. The budget must stay below the smallest defect
    # displacement, about .08 for a lost mass factor, for the check to discriminate.
    bound=sum(item.fact.value.value for item in chosen.facts)
    assert bound<.03
    assert np.linalg.norm(result.solution-expm(-elapsed*dense)@initial)<=bound+1e-8


def test_selected_archive_keeps_actual_products_without_reselection(tmp_path,monkeypatch):
    from nwqlib._choice_archive import ArchiveFiles,save_plan,load_plan
    from nwqlib.algorithms.lchs import coherent_selection,selection,parameters,native
    from nwqlib.subroutines.qsp import evolution,phases
    from nwqlib.subroutines.state_preparation import mps
    problem=LinearDynamics(A=[[.3,.05j],[.05j,.5]],initial_state=[1,2j],time=.1)
    chosen=plan(problem,method=LCHS(ham_evo='qsp_block_encoding',duhamel_nodes=1,
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})))
    assert sum(r.width for r in chosen.construction.program.registers)<=10
    files=ArchiveFiles(tmp_path,8_000_000)
    saved=save_plan(chosen,files)
    def forbidden(*a,**kw):
        pytest.fail('saved selection repeated numerical or native work')
    monkeypatch.setattr(selection,'select_dense',forbidden)
    monkeypatch.setattr(parameters,'select_parameters',forbidden)
    monkeypatch.setattr(native,'construct_coherent_lcu',forbidden)
    for name in ('select_coherent_route','prepare_selected_coherent_route'):
        monkeypatch.setattr(coherent_selection,name,forbidden)
    for name in ('prepare_qsp_evolution','prepare_jacobi_anger_laurent','prepare_jacobi_anger_gqet'):
        monkeypatch.setattr(evolution,name,forbidden)
    monkeypatch.setattr(phases,'solve_laurent_qsp_phases',forbidden)
    monkeypatch.setattr(mps,'_decompose_normalized_state',forbidden)
    for name in ('svd','eigvalsh','eigh','norm'):
        monkeypatch.setattr(np.linalg,name,forbidden)
    loaded=load_plan(saved,files)
    assert loaded==chosen
    original,restored=chosen._native['native_data'],loaded._native['native_data']
    assert restored.method is loaded.method
    np.testing.assert_array_equal(restored.quadrature.coefficients,original.quadrature.coefficients)
    assert not restored.quadrature.coefficients.flags.writeable
    # The selected route, its phases and every row with its reasons come back as saved.
    assert restored.qsp_plan.phases==original.qsp_plan.phases
    assert restored.qsp_plan.selection.candidates==original.qsp_plan.selection.candidates
    assert restored.qsp_plan.preparation==original.qsp_plan.preparation
    # Without a source, planning binds the coefficient plan of the native
    # data, and loading binds the same object.
    assert loaded._native['coefficient_plan'] is restored.coefficient_plan
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
