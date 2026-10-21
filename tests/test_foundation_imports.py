"""Default LCHS planning and classical action without SDK imports."""

import subprocess
import sys




def test_default_lchs_planning_and_classical_action_do_not_attempt_sdk_imports():
    program = """
import builtins, importlib.abc, importlib.util, sys
attempts = []
missing_dependency = None
def check(name):
    if name.split('.')[0] in {'qiskit', 'qiskit_aer'}:
        attempts.append(name)
        raise ModuleNotFoundError('forbidden SDK import: '+name, name=missing_dependency or name.split('.')[0])
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    absolute = importlib.util.resolve_name('.'*level+name, globals['__package__']) if level else name
    check(absolute)
    return original(name, globals, locals, fromlist, level)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        check(fullname)
sys.meta_path.insert(0, Guard())
builtins.__import__ = guarded
from nwqlib import LinearDynamics, plan, solve, estimate
from nwqlib.algorithms import LCHS
problem = LinearDynamics(A=[[.4,.15],[.05,.25]],initial_state=[1.,0.],time=.1)
selected = plan(problem, method=LCHS())
assert selected.execution == 'quantum'
assert estimate(selected).quantity('logical_width', location='logical_device').fact.value.numerator == 9
result = solve(problem,method=LCHS(),execution='classical')
assert result.plan.execution == 'classical' and result.solution.shape == (2,)
# Independent two-by-two exponential from B=A-tr(A)I/2 and B²=delta² I.
# The finite kernel and quadrature each spend half of the .01 tolerance.
from math import cosh, exp, sinh, sqrt
delta=sqrt(.075**2+.15*.05)
reference=(exp(-.0325)*(cosh(.1*delta)-.075*sinh(.1*delta)/delta),
           -.05*exp(-.0325)*sinh(.1*delta)/delta)
assert sum(abs(a-b)**2 for a,b in zip(result.solution,reference))**.5 < .01+64*sys.float_info.epsilon
assert not attempts, attempts
assert not any(name.split('.')[0] in {'qiskit','qiskit_aer'} for name in sys.modules)
# Only actual quantum preparation needs the SDK. Preserve genuine root-extra
# guidance and dependency-internal failures at that existing selected owner.
for missing_dependency in (None, 'dependency_inside_qiskit'):
    attempts.clear()
    try:
        solve(selected)
    except ModuleNotFoundError as error:
        assert attempts
        assert (any('pip install' in note for note in getattr(error, '__notes__', ()))) == (missing_dependency is None)
        if missing_dependency is not None:
            assert error.name == missing_dependency
    else:
        raise AssertionError('native execution bypassed its missing SDK')
"""
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
