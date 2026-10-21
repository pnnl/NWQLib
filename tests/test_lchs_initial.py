"""T=0 removes evolution while requested acquisition and output work remain real."""

import numpy as np
import pytest
from nwqlib import LinearDynamics,solve
from nwqlib.algorithms.lchs import LCHS
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
