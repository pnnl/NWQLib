"""Current advertised owners resolve; configuration modules remain lazy."""

import ast
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest

import nwqlib


def _lazy_export_modules():
    """Modules whose exports resolve through a module-level __getattr__.

    The root package is covered by test_shared_contracts, which resolves every
    root export in a fresh process and rejects unknown root names.
    """
    package = Path(nwqlib.__file__).parent
    names = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body):
            parts = ("nwqlib", *path.relative_to(package).with_suffix("").parts)
            names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return tuple(name for name in names if name != "nwqlib")


@pytest.mark.parametrize("name", _lazy_export_modules())
def test_declared_owner_exports_resolve(name):
    module = import_module(name)
    for public in module.__all__:
        assert getattr(module, public) is not None
    with pytest.raises(AttributeError):
        getattr(module, "not_a_nwqlib_public_entry")


def test_method_configuration_access_preserves_lazy_owner_boundaries():
    program = """
import sys
import nwqlib
assert 'nwqlib.algorithms' not in sys.modules
import nwqlib.algorithms as methods
owners = tuple('nwqlib.algorithms.' + name for name in methods._METHOD_OWNERS.values())
assert not any(name in sys.modules for name in owners)
Lanczos = methods.Lanczos
assert Lanczos is methods.Lanczos
assert 'nwqlib.algorithms.lanczos.method' in sys.modules
assert 'nwqlib.algorithms.qhd.method' not in sys.modules
assert 'nwqlib.algorithms.gcim.chemistry' not in sys.modules
assert not any(name.startswith(('qiskit', 'pyscf', 'openfermion')) for name in sys.modules)
try:
    methods.not_a_nwqlib_public_entry
except AttributeError:
    pass
else:
    raise AssertionError('unknown Method name accepted')
assert 'nwqlib.algorithms.qhd.method' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


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
