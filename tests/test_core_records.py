"""Actual scientific inputs, immutable evidence and exact scalar metadata.

Request/output-scope duplication and its limit list are retired. Relative-zero
assessment belongs to test_error_model's supported-reference/fallback witness;
method applicability and achieved results belong to the actual family tests.
"""

import hashlib
import json
import math
from collections import UserDict
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import numpy as np
import pytest
import sympy
from pydantic import ValidationError

from nwqlib.core import (
    Basis,
    Complex128,
    Float64,
    InputRef,
    Limit,
    Rational,
    Scope,
    Source,
    Symbol,
    Unit,
)
from nwqlib.evidence import Evidence, Fact, FramedFact
from nwqlib.problems import (
    Accuracy,
    ConstrainedOptimization,
    Eigenphase,
    Eigenproblem,
    Eigenvalue,
    Expectation,
    LinearDynamics,
    LinearSystem,
    NormSquared,
    Optimization,
    OptimizationCandidate,
    QuadraticForm,
    Samples,
    Solution,
    SpectralEstimation,
    StateVector,
)
from nwqlib.problems.records import UNSPECIFIED_UNIT

ROOT = Path(__file__).resolve().parents[1]
SCOPE = Scope(domain="test algebraic finite-dimensional object")
ONE = Unit(symbol="1", dimension="dimensionless")
ENERGY = Unit(symbol="hartree", dimension="energy")


SOURCE = Source(
    name="independent fixture declaration",
    version="1",
    domain=SCOPE.domain,
    reference="test:assertions",
)


def input_ref(name="H"):
    return InputRef(identity=name, representation="declared immutable object", source=SOURCE)


def unknown_fact():
    return Fact(
        quantity="spectral gap",
        unit=ENERGY,
        scope=SCOPE,
        availability="unknown",
        reason="not supplied",
    )


def test_nested_detachment_identity_and_validated_record_revisions():
    evidence = Evidence(kind="user_assertion", source=SOURCE)
    admitted = Fact(
        quantity="projection error",
        unit=ENERGY,
        scope=SCOPE,
        availability="concrete",
        value=Float64(value=0.0),
        evidence=evidence,
    )
    original_id = admitted.content_id
    payload = admitted.model_dump(mode="json")
    restored = Fact.model_validate(payload)
    payload["evidence"]["source"]["reference"] = "changed"
    payload["assumptions"].append("fabricated")
    assert admitted.content_id == restored.content_id == original_id
    assert restored.evidence.source.reference == "test:assertions"
    with pytest.raises(ValidationError):
        admitted.evidence.source.reference = "changed"
    assumptions = ["Hermitian by user declaration"]
    revised = admitted.revise(assumptions=assumptions)
    assumptions.append("injected")
    assert revised.assumptions == ("Hermitian by user declaration",)
    assert revised.parent_id == original_id and revised.content_id != original_id
    assert admitted.assumptions == () and admitted.model_copy().content_id == original_id
    with pytest.raises(ValidationError):
        admitted.model_copy(update={"value": float("inf")})
    with pytest.raises(TypeError):
        Fact.model_construct(quantity="unchecked")
    with pytest.raises(TypeError, match="use revise or model_copy"):
        admitted.copy(exclude={"unit"})
    with pytest.raises(ValueError, match="derived"):
        admitted.revise(parent_id="sha256:" + "a" * 64)


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


def test_current_record_formats_and_nested_identities():
    """Check current formats and nested identities through normal mapping round trips."""
    fact = unknown_fact()
    data = fact.model_dump(mode="json")
    assert Fact.model_validate_json(json.dumps(data, sort_keys=True)).content_id == fact.content_id
    data["unit"]["symbol"] = "eV"
    with pytest.raises(ValidationError, match="content_id"):
        Fact.model_validate(data)
    del data["content_id"]
    with pytest.raises(ValidationError, match="content_id"):
        Fact.model_validate(data)  # The unchanged nested ID is also a commitment.
    data = fact.model_dump(mode="json", exclude_computed_fields=True)
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        Fact.model_validate(data)
    problem = Eigenproblem(A=[[1.0, 0.0], [0.0, 2.0]])
    wire = problem.model_dump(mode="json", exclude_computed_fields=True)
    wire["A"]["format"] = "nwqlib.operator_input/99"
    with pytest.raises(ValidationError, match="unsupported"):
        Eigenproblem.model_validate(wire)
    for mapping in (dict, UserDict, MappingProxyType):
        assert Unit.model_validate(mapping(ONE.model_dump())) == ONE


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
    with pytest.raises(ValidationError, match="content_id"):
        Second.Label.model_validate(first.model_dump(mode="json"))
    revised = ENERGY.revise()
    assert revised.same_unit(ENERGY) and revised.parent_id == ENERGY.content_id
    assert revised.content_id != ENERGY.content_id


def test_eigen_target_and_spectral_population_keep_their_actual_distinct_intents():
    eigen = Eigenproblem(A=[[1.0, 1j], [-1j, 2.0]], unit=ENERGY, scope=SCOPE)
    projected = eigen.revise(subspace=input_ref("span-of-reference"))
    assert (
        eigen.target == projected.target == "smallest" and eigen.content_id != projected.content_id
    )
    assert "subspace" in Eigenvalue().frame(projected).conditioning
    assert "smallest" in Eigenvalue().frame(eigen).conditioning
    spectral = SpectralEstimation(hamiltonian=eigen.A, initial_state=[1.0, 0.0], unit=ENERGY)
    unitary = SpectralEstimation(unitary=[[0.0, 1.0], [1.0, 0.0]], initial_state=[1.0, 0.0])
    assert spectral.default_output().kind == unitary.default_output().kind == "eigenphase"
    assert Eigenphase().frame(spectral).unit.symbol == "turn"
    assert Eigenphase().metric == "circular_distance_turns"
    with pytest.raises(ValidationError, match="exactly one"):
        spectral.revise(unitary=unitary.unitary)
    with pytest.raises(ValidationError, match="exactly one"):
        spectral.revise(hamiltonian=None)
    with pytest.raises(ValidationError, match="Hermitian"):
        Eigenproblem(A=[[1.0, 1.0], [0.0, 2.0]])


def test_physical_outputs_do_not_inherit_observable_or_eigenvalue_units():
    vector = [3.0, 4j]
    expectation = Expectation(
        state=vector, observable=[[2.0, 0.0], [0.0, -1.0]], unit=ENERGY, scope=SCOPE
    )
    families = (
        expectation,
        Eigenproblem(A=expectation.observable, unit=ENERGY, scope=SCOPE),
        SpectralEstimation(
            hamiltonian=expectation.observable, initial_state=vector, unit=ENERGY, scope=SCOPE
        ),
    )
    amplitude = Unit(symbol="m", dimension="custom")
    for problem in families:
        for output in (NormSquared(), Solution(), StateVector(normalization="physical")):
            assert output.frame(problem).unit.same_unit(UNSPECIFIED_UNIT)
            assert output.revise(unit=amplitude).frame(problem).unit == amplitude
        assert StateVector().frame(problem).unit.same_unit(ONE)
    normalized = expectation.default_output()
    quadratic = QuadraticForm(observable=expectation.observable)
    assert normalized.frame(expectation).unit == ENERGY
    assert quadratic.frame(expectation).unit.same_unit(UNSPECIFIED_UNIT)
    explicit = quadratic.revise(unit="hartree*m^2")
    assert explicit.frame(expectation).unit.symbol == "hartree*m^2"
    assert (
        normalized.content_id != quadratic.content_id
        and normalized.frame(expectation).scope == SCOPE
    )
    # These inputs have norm 5 and preserve physical phase; no algorithm has
    # acquired either the physical quadratic form 2 or normalized value 2/25.
    assert expectation.state.preparation.physical_scale.as_float() == 5.0
    assert expectation.state.entry(0) == 3.0 and expectation.state.entry(1) == 4j
    for problem in (
        LinearDynamics(A=[[1.0, 0.0], [0.0, 1.0]], initial_state=vector, time=0.0, unit=amplitude),
        LinearSystem(A=[[1.0, 0.0], [0.0, 1.0]], b=vector, unit=amplitude),
    ):
        assert NormSquared().frame(problem).unit.symbol == "(m)^2"
        assert Solution().frame(problem).unit == amplitude
        assert StateVector(normalization="physical").frame(problem).unit == amplitude
        assert StateVector().frame(problem).unit.same_unit(ONE)
    physical = StateVector(normalization="physical")
    quotient = physical.revise(global_phase="modulo_global_phase")
    assert physical.metric == "l2" and quotient.metric == "phase_aligned_l2"
    assert physical.content_id != quotient.content_id and Solution().metric == "l2"


def test_accuracy_has_one_positive_tolerance_and_no_achieved_claim():
    """Separate a requested tolerance from an achieved claim and exercise open probability-domain
    endpoints.
    """
    absolute = Accuracy(absolute_tolerance=0.001, component="sampling")
    relative = Accuracy(relative_tolerance=0.1)
    assert absolute.content_id != relative.content_id
    assert relative.component == "total" and absolute.confidence == 0.95
    revised = absolute.model_copy(update={"absolute_tolerance": 0.002})
    assert absolute.absolute_tolerance == 0.001 and revised.absolute_tolerance == 0.002
    for fields in (
        {},
        {"absolute_tolerance": 0.1, "relative_tolerance": 0.1},
        {"absolute_tolerance": 0.0},
        {"relative_tolerance": -1.0},
        {"relative_tolerance": math.inf},
        {"absolute_tolerance": True},
    ):
        with pytest.raises(ValidationError):
            Accuracy(**fields)
    for confidence in (0.0, 1.0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValidationError):
            absolute.revise(confidence=confidence)
    from nwqlib.evidence import ErrorFrame

    frame = ErrorFrame(
        quantity="error",
        metric="absolute_error",
        unit=ENERGY,
        scope=SCOPE,
        conditioning="specified target",
    )
    zero_probability = FramedFact(
        frame=frame, bindings=(), fact=unknown_fact(), failure_probability=0.0
    )
    assert zero_probability.failure_probability == 0.0
    for probability in (1.0, -1.0, math.nan, math.inf, True):
        with pytest.raises(ValidationError):
            zero_probability.revise(failure_probability=probability)


def test_fact_availability_evidence_and_revision_identity():
    """Contrast unknown, concrete zero and symbolic facts without upgrading an assertion when it
    is witnessed.
    """
    unknown = unknown_fact()
    asserted = Evidence(kind="user_assertion", source=SOURCE)
    zero = Fact(
        quantity="projection error",
        unit=ENERGY,
        scope=SCOPE,
        availability="concrete",
        value=Float64(value=0.0),
        evidence=asserted,
    )
    assert unknown.value is None and unknown.evidence is None and zero.value.value == 0.0
    assert zero.evidence.kind == "user_assertion" and zero.evidence.status == "declared"
    symbolic = Fact(
        quantity="gap",
        unit=ENERGY,
        scope=SCOPE,
        availability="symbolic",
        symbol=Symbol(name="Delta", reference=SOURCE),
    )
    assert Fact.model_validate_json(symbolic.model_dump_json()) == symbolic
    for change in ({"value": Float64(value=0.0)}, {"evidence": asserted}, {"reason": None}):
        with pytest.raises(ValidationError):
            unknown.revise(**change)
    assert unknown.revise(availability="not_applicable", reason="different target").value is None
    with pytest.raises(ValidationError):
        asserted.revise(status="witnessed")
    witnessed = asserted.revise(
        status="witnessed",
        artifact="test:received-user-assertion",
        witnessed_scope=SCOPE,
        subject_id=zero.content_id,
    )
    new_zero = zero.revise(evidence=witnessed)
    assert new_zero.content_id != zero.content_id and new_zero.evidence.kind == "user_assertion"
    assert zero.evidence == asserted
    data = unknown.model_dump(exclude_computed_fields=True)
    data["parent_id"] = "sha256:" + "a" * 64
    assert Fact.model_validate(data).content_id != unknown.content_id


def test_numeric_encodings_are_exact_and_round_trip():
    third = Rational(numerator=1, denominator=3)
    assert third == Rational(numerator=2, denominator=6)
    assert Rational.model_validate_json(third.model_dump_json()).denominator == 3
    half = Rational(numerator=1, denominator=2)
    payload = half.model_dump(mode="json")
    payload.update(numerator=2, denominator=4)
    assert Rational.model_validate(payload).content_id == half.content_id
    assert Rational.model_validate_json(json.dumps(payload)) == half
    # The independently encoded unreduced fields cannot carry the normalized ID.
    fields = {key: value for key, value in payload.items() if key != "content_id"}
    encoding = json.dumps(
        {
            "identity_encoding": "nwqlib.record/1",
            "type": "nwqlib.core.records.Rational",
            "fields": fields,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload["content_id"] = "sha256:" + hashlib.sha256(encoding.encode()).hexdigest()
    with pytest.raises(ValueError, match="content_id does not match"):
        Rational.model_validate(payload)
    binary = Float64(value=1 / 3)
    assert binary.content_id != third.content_id and isinstance(binary.model_dump()["value"], float)
    assert Float64.model_validate(binary.model_dump(mode="json")) == binary
    with pytest.raises(ValidationError, match="content_id"):
        Float64.model_validate({"value": binary.value, "content_id": "sha256:" + "0" * 64})
    assert Float64(value=-0.0).content_id == Float64(value=0.0).content_id
    z = Complex128(real=-0.0, imag=1 / 3)
    reloaded = Complex128.model_validate_json(z.model_dump_json())
    assert math.copysign(1, reloaded.real) == 1 and reloaded.imag == 1 / 3
    for number in (math.inf, -math.inf, math.nan):
        for cls, kwargs in (
            (Float64, {"value": number}),
            (Complex128, {"real": 0.0, "imag": number}),
        ):
            with pytest.raises(ValidationError):
                cls(**kwargs)
    for numerator, denominator in ((1, 0), (1.0, 3)):
        with pytest.raises(ValidationError):
            Rational(numerator=numerator, denominator=denominator)


def test_limits_preserve_stock_flow_stage_and_exact_counts():
    """Use 2**53 + 1 to expose float coercion while checking distinct stock, consumption and
    deadline units.
    """
    byte = Unit(symbol="byte", dimension="bytes")
    memory = Limit(
        stage="planning",
        metric="memory",
        unit=byte,
        kind="capacity_stock",
        value=2**53 + 1,
        scope="process peak",
    )
    stored = memory.revise(stage="execution", metric="stored")
    transfer = Limit(
        stage="preparation",
        metric="transfer",
        unit=byte,
        kind="consumption",
        value=2**53 + 1,
        scope="all transfers",
    )
    cpu = Limit(
        stage="verification",
        metric="cpu_time",
        unit=Unit(symbol="s", dimension="time"),
        kind="consumption",
        value=0.5,
        scope="total",
    )
    deadline = cpu.revise(metric="wall_time", kind="deadline")
    shots = Limit(
        stage="execution",
        metric="shots",
        unit=Unit(symbol="count", dimension="count"),
        kind="consumption",
        value=2**53 + 1,
        scope="all acquisitions",
    )
    for limit in (memory, stored, transfer, cpu, deadline, shots):
        assert Limit.model_validate_json(limit.model_dump_json()) == limit
    assert memory.value == transfer.value == shots.value == 2**53 + 1 and type(memory.value) is int
    with pytest.raises(ValidationError):
        memory.revise(kind="consumption")
    with pytest.raises(ValidationError):
        memory.revise(unit=cpu.unit)
    for value in (-1, math.inf):
        with pytest.raises(ValidationError):
            memory.revise(value=value)
        with pytest.raises(ValidationError):
            shots.revise(value=value)
    with pytest.raises(ValidationError):
        cpu.revise(unit=Unit(symbol="ms", dimension="time"))  # No automatic time conversion.


def test_current_problem_domains_and_output_metrics_admit_original_coordinates():
    """Keep original dimensions, elapsed time and variable order while rejecting mismatched
    scientific domains.
    """
    import sympy as sp

    matrix = [[1.0, 1.0], [0.0, 2.0]]
    dynamics = LinearDynamics(
        A=matrix, initial_state=[3.0, 4j], source=[0.0, 1.0], initial_time=1.0, time=2.0
    )
    system = LinearSystem(A=dynamics.A, b=dynamics.initial_state)
    assert (
        dynamics.elapsed_time == 1.0
        and dynamics.default_output().kind == system.default_output().kind == "solution"
    )
    assert (
        Samples().metric == "total_variation" and OptimizationCandidate().metric == "objective_gap"
    )
    for changes in (
        {"time": 0.0},
        {"source": [1.0, 0.0, 0.0]},
        {"initial_state": [1.0, 0.0, 0.0]},
        {"time_unit": ENERGY},
    ):
        with pytest.raises(ValidationError):
            dynamics.revise(**changes)
    with pytest.raises(ValidationError, match="dimension"):
        system.revise(b=[1.0, 0.0, 0.0])
    x, y = sp.symbols("x y", real=True)
    optimization = Optimization(
        objective=x * x + y, variables=(y, x), bounds=((-1.0, 1.0), (-2.0, 2.0))
    )
    restored = Optimization.model_validate_json(optimization.model_dump_json())
    assert restored.variable_names == ("y", "x") and restored.content_id == optimization.content_id
    for changes in (
        {"variables": (x, x)},
        {"bounds": ((1.0, -1.0), (-2.0, 2.0))},
        {"objective": x + y + sp.Symbol("z")},
    ):
        with pytest.raises(ValidationError):
            optimization.revise(**changes)
    with pytest.raises(TypeError, match="SymPy"):
        optimization.revise(objective="x*x+y")
    with pytest.raises(ValidationError):
        Basis(identity="bad", dimension=0, ordering="empty")


_X, _Y, _Z = sympy.symbols("x y z", real=True)
_BOX = dict(objective=(_X - 1) ** 2 + _Y**2, variables=(_Y, _X), bounds=((-1.0, 1.0), (-1.0, 2.0)))


def _saved_box(**constraints):
    """Descriptive JSON fields of a record, as a saved record rebuilds them."""
    return json.loads(ConstrainedOptimization(**_BOX, **constraints).model_dump_json())


# Each accepted case lists the constructor fields and the expected residuals h
# (for h = 0) and g (for g <= 0), written out by hand. Each rejected case lists the
# constructor fields, the exception type and the message that names the failure.
@pytest.mark.parametrize("fields, expected", [
    pytest.param(dict(_BOX, equalities=(sympy.Eq(_X + _Y, 1),), inequalities=(sympy.Le(_X, _Y**2), sympy.Ge(_X, _Y))),
                 ((_X + _Y - 1,), (_X - _Y**2, _Y - _X)), id="relations"),
    pytest.param(dict(_BOX, equalities=(_X - 2 * _Y,), inequalities=(_X * _Y - 1,)),
                 ((_X - 2 * _Y,), (_X * _Y - 1,)), id="bare-expressions"),
    pytest.param(dict(_BOX, inequalities=(sympy.Integer(-1),)), ((), (-1,)), id="constant-residual"),
    pytest.param(dict(_BOX, equalities=(sympy.Eq(_X, _X, evaluate=False),),
                      inequalities=(sympy.Le(1, 2, evaluate=False),)),
                 ((0,), (-1,)), id="unevaluated-relations"),
    pytest.param(dict(_BOX, inequalities=(sympy.Lt(_X, 1),)),
                 (ValidationError, r"inequalities\[0\].*strict.*margin"), id="strict-lt"),
    pytest.param(dict(_BOX, inequalities=(sympy.Le(_X, 1), sympy.Gt(_X, _Y))),
                 (ValidationError, r"inequalities\[1\].*strict.*margin"), id="strict-gt"),
    pytest.param(dict(_BOX, equalities=(sympy.Ne(_X, _Y),)),
                 (ValidationError, r"equalities\[0\].*Ne"), id="unequal"),
    pytest.param(dict(_BOX, equalities=(sympy.Eq(_X, _X),)),
                 (ValidationError, r"equalities\[0\] is True, which SymPy evaluated"), id="evaluated-true"),
    pytest.param(dict(_BOX, inequalities=(sympy.Le(2, 1),)),
                 (ValidationError, r"inequalities\[0\] is False, which SymPy evaluated"), id="evaluated-false"),
    pytest.param(dict(_BOX, inequalities=(sympy.And(_X <= 1, _Y <= 1),)),
                 (ValidationError, r"inequalities\[0\].*And"), id="boolean"),
    pytest.param(dict(_BOX, inequalities={sympy.Ge(_X, 1)}),
                 (TypeError, "inequalities must be a tuple"), id="unordered-container"),
    pytest.param(dict(_BOX, equalities=(sympy.Eq(sympy.Matrix([_X, _Y]), sympy.Matrix([1, 2]), evaluate=False),)),
                 (ValidationError, r"equalities\[0\] has a side of type \w*Matrix"), id="matrix-relation"),
    pytest.param(dict(_BOX, equalities=(sympy.Le(_X, 1),)),
                 (ValidationError, r"equalities\[0\].*inequalities"), id="inequality-as-equality"),
    pytest.param(dict(_BOX, inequalities=(sympy.Eq(_X, 1),)),
                 (ValidationError, r"inequalities\[0\].*equalities"), id="equality-as-inequality"),
    pytest.param(dict(_BOX, inequalities=(_X - 1, _Z - 1)),
                 (ValidationError, r"inequalities\[1\].*missing from variables.*z"), id="unknown-symbol"),
    pytest.param(dict(_BOX), (ValidationError, "at least one"), id="no-constraint"),
    pytest.param(lambda: dict(_saved_box(equalities=(_X - _Y,)), equalities=(_X - _Y,)),
                 (ValidationError, "consistently live or descriptive"), id="descriptive-box-live-constraint"),
    pytest.param(lambda: dict(_BOX, inequalities=_saved_box(inequalities=(_X - _Y,))["inequalities"]),
                 (ValidationError, "consistently live or descriptive"), id="live-box-descriptive-constraint"),
])
def test_constrained_optimization_stores_residuals_and_is_not_an_optimization(fields, expected):
    """Admit h = 0 and g <= 0 residuals in the original coordinate order.

    QHD's box-only planning must refuse the constrained record, so a constraint
    is never silently dropped.
    """
    import nwqlib
    from nwqlib.algorithms import QHD, ApplicabilityError
    from nwqlib.problems.records import SymbolicDescription

    fields = fields() if callable(fields) else fields
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
