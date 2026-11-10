"""T=0 removes evolution while requested acquisition and output work remain real."""

import numpy as np
import pytest
from nwqlib import LinearDynamics,QuadraticForm,solve
from nwqlib.algorithms.lchs import LCHS,Coherent,SampledPairs
from nwqlib.operators.inputs import OperatorInput
from nwqlib.problems.inputs import StateInput


def initial_problem():
    vector=np.array([1,1j,-.25])
    observable=np.array([[.2,.1j,.3],[-.1j,-.5,.2j],[.3,-.2j,.7]])
    problem=LinearDynamics(A=np.diag([.1,.2,.3]),initial_state=vector,time=2.,initial_time=2.)
    norm=float(np.vdot(vector,vector).real)
    moment=float(np.vdot(vector,observable@vector).real)
    return problem,observable,norm,moment


def test_initial_identity_does_not_need_unused_source_data(monkeypatch):
    from nwqlib.algorithms.lchs import selection
    monkeypatch.setattr(selection,'dense_common',
        lambda *a,**kw: pytest.fail('algebraic identity selected an evolution'))
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


def test_initial_counts_radius_reads_the_estimators_failure_probability():
    """The no-evolution counts do not depend on the estimator, so their radius takes the estimator's own probability.

    At zero elapsed time both estimators measure the same initial-state PREP,
    so equal shots and seed give equal counts. Each must report the shot
    radius at its own `fail_prob`, here 0.2 rather than the default 0.05.
    """
    problem,observable,_,_=initial_problem()
    sampling=[]
    for estimator in (Coherent(fail_prob=.2),SampledPairs(fail_prob=.2)):
        result=solve(problem,method=LCHS(ham_evo='trotter',estimator=estimator),
            output=QuadraticForm(observable=observable),execution='quantum',shots=400,seed=5)
        sampling.append(next(item for item in result.facts if item.fact.quantity=='sampling'))
    assert [item.failure_probability for item in sampling]==[.2,.2]
    # Same counts, same radius function and probability: the same binary64 value.
    assert sampling[0].fact.value==sampling[1].fact.value
