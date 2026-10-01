"""T=0 removes evolution while requested acquisition and output work remain real."""

import numpy as np
import pytest
from nwqlib import LinearDynamics,NormalizedExpectation,QuadraticForm,NormSquared,StateVector,Solution,Samples,plan,solve,load_result
from nwqlib.algorithms.lchs import LCHS
from nwqlib.operators.inputs import OperatorInput,PeriodicStencil
from nwqlib.problems.inputs import StateInput,ingest_product


def initial_problem():
    vector=np.array([1,1j,-.25])
    observable=np.array([[.2,.1j,.3],[-.1j,-.5,.2j],[.3,-.2j,.7]])
    problem=LinearDynamics(A=np.diag([.1,.2,.3]),initial_state=vector,time=2.,initial_time=2.)
    norm=float(np.vdot(vector,vector).real)
    moment=float(np.vdot(vector,observable@vector).real)
    return problem,observable,norm,moment


@pytest.mark.parametrize('output_type',[NormalizedExpectation,QuadraticForm])
def test_quantum_initial_observable_acquires_real_readout_without_classical_action(output_type,monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms
    problem,observable,norm,moment=initial_problem()
    expected=moment/norm if output_type is NormalizedExpectation else moment
    monkeypatch.setattr(time_independent_terms,'generate_lchs_quadrature',lambda **kw: pytest.fail('T=0 selected evolution'))
    monkeypatch.setattr(OperatorInput,'matvec',lambda *a,**kw: pytest.fail('quantum request silently used classical observable action'))
    chosen=plan(problem,method=LCHS(),output=output_type(observable=observable))
    assert chosen.execution=='quantum' and chosen.experiments and not chosen.construction.kernels
    assert sum(register.width for register in chosen.construction.program.registers)==2
    result=solve(chosen)
    assert result.value==pytest.approx(expected,abs=4e-13)
    assert all(event.execution=='quantum_circuit' for event in result.data.trace.events)
    assert result.analyze().value==result.value


def test_quantum_initial_counts_keep_requested_shots():
    problem,_,_,_=initial_problem()
    chosen=plan(problem,method=LCHS(),output=NormalizedExpectation(observable=np.diag([1.,-1.,.5])),shots=128,seed=3)
    assert chosen.execution=='quantum' and chosen.shots==128
    assert sum(register.width for register in chosen.construction.program.registers)==2
    result=solve(chosen)
    assert result.value is not None
    assert all(chunk.returned_shots==128 for chunk in result.data.observations.chunks)
    assert all(event.execution=='quantum_circuit' and event.shots==128 for event in result.data.trace.events)


@pytest.mark.parametrize('output_type',[NormalizedExpectation,QuadraticForm])
def test_classical_initial_observable_is_one_saved_host_acquisition(output_type,tmp_path,monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms
    from nwqlib.operators import inputs
    problem,observable,norm,moment=initial_problem()
    expected=moment/norm if output_type is NormalizedExpectation else moment
    monkeypatch.setattr(time_independent_terms,'generate_lchs_quadrature',lambda **kw: pytest.fail('T=0 selected evolution'))
    calls=[]
    original=inputs._scaled_observable
    class CountedMatrix(np.ndarray):
        def __matmul__(self, vector):
            calls.append(1)
            return np.asarray(self) @ vector
    def scaled(operator, exponent):
        return original(operator, exponent).view(CountedMatrix)
    monkeypatch.setattr(inputs,'_scaled_observable',scaled)
    chosen=plan(problem,method=LCHS(),output=output_type(observable=observable),execution='classical')
    assert not calls
    assert chosen.execution=='classical' and len(chosen.construction.kernels)==1
    kernel=chosen.construction.kernels[0]
    assert kernel.invocation_work>=3**2+3
    assert kernel.inputs==(chosen.problem.initial_state.reference,chosen.output.observable.reference)
    result=solve(chosen)
    assert result.value==pytest.approx(expected,abs=2e-15)
    assert calls==[1]
    assert len(result.data.trace.events)==1 and result.data.trace.events[0].execution=='host_kernel'
    assert result.applications[0].name=='initial_observable'
    monkeypatch.setattr(inputs,'_scaled_observable_moment',lambda *a,**kw: pytest.fail('saved scalar recomputed its observable'))
    repeated=result.analyze()
    result.report()
    loaded=load_result(result.save(tmp_path/'initial'))
    loaded.report()
    assert loaded.analyze().value==repeated.value==result.value
    assert loaded.data.observations==result.data.observations


@pytest.mark.parametrize('limits',[{'max_select_work':40},{'max_bytes':256}])
def test_initial_host_work_is_admitted_before_observable_action(limits,monkeypatch):
    from nwqlib.operators import inputs
    problem,observable,_,_=initial_problem()
    monkeypatch.setattr(OperatorInput,'matvec',lambda *a,**kw: pytest.fail('unadmitted initial action'))
    monkeypatch.setattr(inputs,'_scaled_observable',lambda *a,**kw: pytest.fail('unadmitted observable copy'))
    # 40 work / 256 bytes fit the old unscaled action, but not the actual copy/scans.
    with pytest.raises(ValueError,match='max_select_work|initial output arrays'):
        plan(problem,method=LCHS(**limits),output=QuadraticForm(observable=observable),execution='classical')


@pytest.mark.parametrize('execution',['quantum','classical'])
def test_compact_zero_never_prepares_and_physical_array_is_materialized_once(execution,tmp_path,monkeypatch):
    from nwqlib.algorithms.lchs import host
    state=ingest_product([[0,0],[1,1]])
    problem=LinearDynamics(A=PeriodicStencil(num_qubits=2,mass=.1,diffusion=.2),initial_state=state,time=0.)
    monkeypatch.setattr(host,'_initial_acquisition',lambda *a,**kw: pytest.fail('zero vector reached PREP'))
    monkeypatch.setattr(OperatorInput,'matvec',lambda *a,**kw: pytest.fail('zero vector acquired observable action'))
    unit=solve(problem,method=LCHS(),output=StateVector(normalization='unit'),execution=execution)
    assert unit.unavailable and unit.data.observations.chunks==()
    assert solve(problem,method=LCHS(),output=NormSquared(),execution=execution).value==0
    for kind in (NormalizedExpectation,QuadraticForm):
        value=solve(problem,method=LCHS(),output=kind(observable=np.eye(4)),execution=execution)
        if kind is NormalizedExpectation:
            assert value.value is None and value.unavailable is not None
        else:
            assert value.value==0.
        assert value.data.observations.chunks==()
    chosen=plan(problem,method=LCHS(),output=Solution(),execution=execution)
    assert chosen.execution==execution and len(chosen.construction.kernels)==1
    assert chosen.construction.kernels[0].invocation_work==4
    result=solve(chosen)
    np.testing.assert_array_equal(result.solution,np.zeros(4))
    assert len(result.data.trace.events)==1 and result.data.trace.events[0].execution=='host_kernel'
    monkeypatch.setattr(host,'_execute_initial_output',lambda *a: pytest.fail('zero output materialized again'))
    np.testing.assert_array_equal(result.analyze().solution,result.solution)
    loaded=load_result(result.save(tmp_path/'zero'))
    np.testing.assert_array_equal(loaded.analyze().solution,result.solution)
    with pytest.raises(ValueError,match='initial output arrays'):
        plan(problem,method=LCHS(max_bytes=8),output=Solution(),execution=execution)
    if execution=='quantum':
        with pytest.raises(ValueError,match='undefined'):
            plan(problem,method=LCHS(),output=Samples(),shots=32)


def test_initial_identity_does_not_need_unused_source_data(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms
    monkeypatch.setattr(time_independent_terms,'generate_lchs_quadrature',
        lambda **kw: pytest.fail('algebraic identity selected an evolution'))
    problem,_,_,_=initial_problem()
    source=StateInput.from_record(problem.initial_state.to_record())
    chosen=LinearDynamics(A=OperatorInput.from_record(problem.A.to_record()),initial_state=problem.initial_state,
        source=source,time=problem.time,initial_time=problem.initial_time)
    result=solve(chosen,method=LCHS())
    assert result.data.observations.chunks==()
    np.testing.assert_array_equal(result.solution,problem.initial_state.physical_vector())
    # A nonzero source contributes nothing over zero elapsed time.
    explicit=LinearDynamics(A=np.eye(3),initial_state=[1j,0,-.5],source=[1,2,3],time=2,initial_time=2)
    result=solve(explicit,method=LCHS())
    assert result.data.observations.chunks==()
    np.testing.assert_array_equal(result.solution,np.array([1j,0,-.5]))


@pytest.mark.parametrize('storage', ['dense', 'csr', 'pauli'])
@pytest.mark.parametrize('factor', [1., -1., 2j])
def test_initial_quadratic_recovers_observable_before_physical_scale(storage, factor):
    from scipy import sparse
    from nwqlib.operators import ingest_pauli

    a = -1e308 if factor == -1 else 1e308
    vector = np.full(2, 1e-160 * factor, dtype=complex)
    matrix = np.full((2, 2), a)
    observable = (ingest_pauli((('I', a), ('X', a)), num_qubits=1) if storage == 'pauli'
                  else sparse.csr_matrix(matrix) if storage == 'csr'
                  else sparse.csc_matrix(matrix) if storage == 'csc' else matrix)
    problem = LinearDynamics(A=np.zeros((2, 2)), initial_state=vector, time=0.)
    result = solve(problem, method=LCHS(), output=QuadraticForm(observable=observable),
                   execution='classical')
    # Four terms a*|psi_j|²; neither |psi|² nor 2*a is formed in the oracle.
    expected = 4 * (a * 1e-160) * 1e-160 * abs(factor)**2
    assert result.value == pytest.approx(expected, rel=2e-14, abs=0.)


@pytest.mark.parametrize('time', [0., .1])
@pytest.mark.parametrize('output_type', [QuadraticForm, NormalizedExpectation])
def test_observable_recovery_preserves_cancellation_and_heterogeneous_storage(time, output_type):
    from scipy.sparse import csr_matrix

    for vector, matrix, expected in (
        ([1., 1.], np.diag([1e308, -1e308]), 0.),
        ([0., 1.], np.diag([1e308, 1e-308]), 1e-308),
        ([1., 0.], [[0., 1e308], [1e308, 0.]], 0.),
        ([1., 1.], np.full((2, 2), 1e308), None),
    ):
        for observable in (matrix, csr_matrix(matrix)):
            result = solve(LinearDynamics(A=np.zeros((2, 2)), initial_state=vector, time=time),
                           method=LCHS(), output=output_type(observable=observable), execution='classical')
            if expected is None:
                assert result.value is None and result.unavailable
            else:
                # Near cancellation the contraction error is absolute, bounded
                # by roundoff times ||O||*||psi||², not relative to the zero result.
                # Zero-generator evolution/normalization contributes another few
                # ulps. Nonzero tiny entries retain a relative check, not this floor.
                cancellation = 64*np.finfo(float).eps*1e308*2 if expected == 0 else 0.
                assert result.value == pytest.approx(expected, rel=64*np.finfo(float).eps, abs=cancellation)


def test_subnormal_component_outside_observable_support_does_not_hide_scalar():
    from math import ldexp

    # The second component is the smallest positive binary64 number. Its
    # underflow in the original vector frame cannot affect diag(1,0)'s QF.
    problem = LinearDynamics(A=np.zeros((2, 2)), initial_state=[1., ldexp(1., -1074)], time=0.)
    result = solve(problem, method=LCHS(), output=QuadraticForm(observable=np.diag([1.,0.])),
                   execution='classical')
    assert result.value == 1.  # Exact first-coordinate projection.
