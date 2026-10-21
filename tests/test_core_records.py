"""Scientific inputs, record and unit identity, constrained-optimization residuals and
fresh-process SDK import audits."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy
from pydantic import ValidationError

from nwqlib.core import (
    Scope,
    Unit,
)
from nwqlib.evidence import Fact
from nwqlib.problems import (
    ConstrainedOptimization,
    Eigenproblem,
    LinearDynamics,
    LinearSystem,
    NormSquared,
    OptimizationCandidate,
    Solution,
    StateVector,
)
from nwqlib.problems.records import ONE, UNSPECIFIED_UNIT

ROOT = Path(__file__).resolve().parents[1]
SCOPE = Scope(domain="test algebraic finite-dimensional object")
ENERGY = Unit(symbol="hartree", dimension="energy")


def unknown_fact():
    return Fact(
        quantity="spectral gap",
        unit=ENERGY,
        scope=SCOPE,
        availability="unknown",
        reason="not supplied",
    )


def test_output_units_follow_the_problem_that_defines_them():
    # docs/problems.md, Units: the unit of a LinearSystem or LinearDynamics labels the solution
    # amplitude, so Solution and a physical StateVector carry it and NormSquared its square. An
    # Eigenproblem defines no amplitude unit, so these outputs keep an unspecified unit unless
    # they declare one.
    amplitude = Unit(symbol="m", dimension="custom")
    vector, identity = [3.0, 4j], [[1.0, 0.0], [0.0, 1.0]]
    for problem in (LinearDynamics(A=identity, initial_state=vector, time=0.0, unit=amplitude),
                    LinearSystem(A=identity, b=vector, unit=amplitude)):
        assert NormSquared().frame(problem).unit.symbol == "(m)^2"
        assert Solution().frame(problem).unit == amplitude
        assert StateVector(normalization="physical").frame(problem).unit == amplitude
        assert StateVector().frame(problem).unit.same_unit(ONE)
    eigenproblem = Eigenproblem(A=[[2.0, 0.0], [0.0, -1.0]], unit=ENERGY, scope=SCOPE)
    for output in (NormSquared(), Solution(), StateVector(normalization="physical")):
        assert output.frame(eigenproblem).unit.same_unit(UNSPECIFIED_UNIT)
        assert output.revise(unit=amplitude).frame(eigenproblem).unit == amplitude


def test_scientific_inputs_keep_original_arrays_and_detached_descriptions():
    """Mutate caller arrays and serialized descriptions to distinguish owned physical data from
    detached metadata.
    """
    matrix = np.diag([1.0, 2.0]).astype(complex)
    initial = np.array([3.0, 4j])
    assumptions = ["supplied model"]
    problem = LinearDynamics(
        A=matrix,
        initial_state=initial,
        time=2.0,
        unit="amplitude",
        scope=SCOPE,
        assumptions=assumptions,
        facts=(unknown_fact(),),
    )
    identity = problem.content_id
    ordering = problem.basis.ordering
    matrix[0, 0], initial[0] = 99, 99
    assumptions.append("later")
    assert problem.A.dense_array()[0, 0] == 1 and problem.initial_state.entry(0) == 3
    assert (
        problem.initial_state.entry(1) == 4j
        and problem.initial_state.preparation.physical_scale.as_float() == 5
    )
    assert not problem.A.dense_array().flags.writeable
    payload = problem.model_dump(mode="json")
    restored = LinearDynamics.model_validate(payload)
    payload["A"]["manifest"]["basis"]["ordering"] = "reversed"
    payload["facts"].append({"not": "a fact"})
    assert problem.content_id == restored.content_id == identity
    assert restored.basis.ordering == ordering and problem.assumptions == ("supplied model",)
    with pytest.raises(ValueError, match="unavailable"):
        restored.initial_state.entry(0)
    with pytest.raises(ValueError, match="unavailable"):
        restored.A.dense_array()
    changed = problem.revise(time=3.0)
    assert changed.content_id != identity and changed.initial_state is problem.initial_state
    assert (
        problem.time == 2.0 and changed.parent_id is None
    )  # Scientific inputs have no revision bookkeeping field.


def test_qualified_record_identity_and_unit_meaning_are_distinct():
    class First:
        class Label(Unit):
            pass

    class Second:
        class Label(Unit):
            pass

    first = First.Label(symbol="hartree", dimension="energy")
    second = Second.Label(symbol="hartree", dimension="energy")
    assert first.same_unit(second) and first.content_id != second.content_id
    revised = ENERGY.revise()
    assert revised.same_unit(ENERGY) and revised.parent_id == ENERGY.content_id
    assert revised.content_id != ENERGY.content_id


_X, _Y = sympy.symbols("x y", real=True)
_BOX = dict(objective=(_X - 1) ** 2 + _Y**2, variables=(_Y, _X), bounds=((-1.0, 1.0), (-1.0, 2.0)))


# Each accepted case lists the constructor fields and the expected residuals h
# (for h = 0) and g (for g <= 0), written out by hand. Each rejected case lists the
# constructor fields, the exception type and the message that names the failure.
@pytest.mark.parametrize("fields, expected", [
    pytest.param(dict(_BOX, equalities=(sympy.Eq(_X + _Y, 1),), inequalities=(sympy.Le(_X, _Y**2), sympy.Ge(_X, _Y))),
                 ((_X + _Y - 1,), (_X - _Y**2, _Y - _X)), id="relations"),
    pytest.param(dict(_BOX, equalities=(sympy.Eq(_X, _X, evaluate=False),),
                      inequalities=(sympy.Le(1, 2, evaluate=False),)),
                 ((0,), (-1,)), id="unevaluated-relations"),
    pytest.param(dict(_BOX, inequalities=(sympy.Lt(_X, 1),)),
                 (ValidationError, r"inequalities\[0\].*strict.*margin"), id="strict-lt"),
])
def test_constrained_optimization_stores_residuals_and_is_not_an_optimization(fields, expected):
    """Admit h = 0 and g <= 0 residuals in the original coordinate order.

    QHD's box-only planning must refuse the constrained record, so a constraint
    is never silently dropped.
    """
    import nwqlib
    from nwqlib.algorithms import QHD, ApplicabilityError
    from nwqlib.problems.records import SymbolicDescription

    if isinstance(expected[0], type):
        error, message = expected
        with pytest.raises(error, match=message):
            ConstrainedOptimization(**fields)
        return
    record = ConstrainedOptimization(**fields)
    for stored, residuals in zip((record.equalities, record.inequalities), expected):
        assert len(stored) == len(residuals)
        assert all(sympy.simplify(s - r) == 0 for s, r in zip(stored, residuals))
    restored = ConstrainedOptimization.model_validate_json(record.model_dump_json())
    assert restored.content_id == record.content_id
    assert restored.variable_names == ("y", "x") and restored.bounds == ((-1.0, 1.0), (-1.0, 2.0))
    for saved, live in zip(restored.equalities + restored.inequalities, record.equalities + record.inequalities):
        assert saved == SymbolicDescription(expression=sympy.srepr(live), display=str(live))
    assert record.default_output() == OptimizationCandidate()
    with pytest.raises(ApplicabilityError, match="QHD requires an Optimization"):
        nwqlib.plan(record, method=QHD(num_grid_points=3), execution="classical")


@pytest.mark.parametrize(
    "script", ["check_core_records.py", "check_prepared_records.py", "check_qasm_records.py"]
)
def test_fresh_subprocess_sdk_import_attempt_audit(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "docs/scripts" / script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Caught blocked import control rejected" in result.stdout
