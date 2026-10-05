"""Independent output-frame, exact-scalar and covariance witnesses; no native work."""

import re
from fractions import Fraction
from unittest.mock import ANY

import pytest

from nwqlib.core import Float64, Rational, Scope, Source, Unit
from nwqlib.evidence import (
    AssessmentContext, CheckDomain, CheckSpec, CrossCovariance, ErrorFrame, ErrorModel,
    ErrorTerm, Evidence, EstimatorContribution, Fact, FramedFact, IndependenceLaw,
    TargetReference, assemble_check, linear_variance,
)
from nwqlib.evidence._work import ExactArithmetic
from nwqlib.problems.records import Accuracy

SCOPE = Scope(domain="supplied projected-energy evidence")
SOURCE = Source(name="independent scalar fixture", version="1", domain=SCOPE.domain,
                reference="explicit exact scalar relations and finite probability spaces")
FRAME = ErrorFrame(quantity="eigenvalue", metric="absolute_error", unit=Unit(symbol="hartree", dimension="energy"),
                   scope=SCOPE, conditioning="specified input and target")
CONTEXT = AssessmentContext(problem_id="sha256:" + "1"*64, base_construction_id="sha256:" + "2"*64,
    selected_construction_id="sha256:" + "2"*64, plan_id="sha256:" + "3"*64,
    observation_id="sha256:" + "4"*64, result_id="sha256:" + "5"*64, admitted=True)


def value(item):
    raw = item.fact.value if isinstance(item, FramedFact) else item
    return Fraction(raw.numerator, raw.denominator)


def fact(name, number=None, *, frame=FRAME, kind="proved_relation", bindings=(), assumptions=(), failure_probability=0.):
    fields = dict(quantity=name, unit=frame.unit, scope=frame.scope)
    if number is None:
        fields.update(availability="unknown", reason="not supplied")
    else:
        ratio = Fraction(number)
        fields.update(availability="concrete", value=Rational(numerator=ratio.numerator, denominator=ratio.denominator),
            evidence=Evidence(kind=kind, source=SOURCE, status="witnessed", artifact="independently supplied scalar relation",
                              witnessed_scope=frame.scope, subject_id=CONTEXT.result_id))
    return FramedFact(frame=frame, bindings=bindings, fact=Fact(**fields, assumptions=assumptions),
        failure_probability=None if number is None else failure_probability)


def model(*, physical=Fraction(1, 8), sampling=Fraction(1, 4), extra=()):
    terms = tuple(ErrorTerm(name=name, stage="analysis", source=SOURCE, formula="supplied absolute bound",
        fact=fact(name, number), coverage=name, failure_probability=0.) for name, number in (("physical", physical), ("sampling", sampling)))
    return ErrorModel(output_id="sha256:" + "6"*64, subject_id=CONTEXT.problem_id,
        construction_id=CONTEXT.base_construction_id, frame=FRAME, required_sources=("physical", "sampling", *extra),
        terms=terms, source=SOURCE)


def test_new_criterion_keeps_original_context_and_component_scope():
    original = model(physical=None)
    sampling = original.assess(Accuracy(absolute_tolerance=.25, component="sampling"), context=CONTEXT)
    assert sampling.status == "PASS" and value(sampling.covered_subtotal) == Fraction(1, 4)
    total = original.assess(Accuracy(absolute_tolerance=.25), context=CONTEXT)
    assert total.status == "INCONCLUSIVE" and total.remaining == ("physical",)
    tighter = original.assess(Accuracy(absolute_tolerance=.125, component="sampling"), context=CONTEXT)
    assert tighter.status == "INCONCLUSIVE" and tighter.context == sampling.context == CONTEXT
    # A filled subject cache keeps equality, and an operand of another type gets its own comparison.
    assert CONTEXT.subjects and CONTEXT.model_copy() == CONTEXT == ANY
    assert CONTEXT.__eq__(object()) is NotImplemented
    assert tighter.accuracy != sampling.accuracy
    assert original.terms[0].fact.fact.value is None
    # Zero residual of the excited eigenpair diag(0,1), v=(0,1), E=1 does not
    # establish its ground-energy error, which is exactly one.
    projected = model(physical=0, sampling=0, extra=("ground_identification",))
    result = projected.assess(Accuracy(absolute_tolerance=.01), context=CONTEXT)
    assert result.status == "INCONCLUSIVE" and result.remaining == ("ground_identification",)
    assert value(result.covered_subtotal) == 0


def test_invalid_bounds_frames_and_associations_reject_before_missing_evidence():
    original = model(physical=None)
    negative = fact("physical", Fraction(-1, 10**13))
    with pytest.raises(ValueError, match="negative"):
        original.assess(Accuracy(absolute_tolerance=.1, component="sampling"), context=CONTEXT, facts=(negative,))
    wrong = fact("sampling", 0, frame=FRAME.revise(domain="zero_input"))
    with pytest.raises(ValueError, match="frame"):
        original.assess(Accuracy(absolute_tolerance=.1), context=CONTEXT, facts=(wrong,))
    with pytest.raises(ValueError, match="different scientific"):
        original.assess(Accuracy(absolute_tolerance=.1), context=CONTEXT.revise(problem_id="sha256:" + "a"*64))
    empirical = fact("sampling", 0, kind="observed")
    result = model(physical=0).assess(Accuracy(absolute_tolerance=.1), context=CONTEXT, facts=(empirical,))
    assert result.status == "INCONCLUSIVE" and result.unverified == ("sampling",)
    loaded = type(result).model_validate_json(result.model_dump_json())
    assert loaded == result and loaded.facts[-1].fact.evidence.kind == "observed"
    for field in ("threshold", "covered_subtotal", "failure_probability"):
        with pytest.raises(ValueError):
            result.revise(**{field: Rational(numerator=-1, denominator=1)})


def test_relative_scale_requires_supported_nonzero_target_or_explicit_fallback():
    original = model(physical=Fraction(1, 2), sampling=0)
    accuracy = Accuracy(relative_tolerance=Fraction(1, 8))
    def assess(reference=None, **kwargs):
        return original.assess(accuracy, context=CONTEXT, reference=reference, **kwargs)
    reference = TargetReference(relation="exact_target", fact=fact("eigenvalue", -2), failure_probability=0.)
    assert value(assess(reference).threshold) == Fraction(1, 4)
    assert assess(reference).status == "INCONCLUSIVE"
    lower = reference.revise(relation="magnitude_lower_bound", fact=fact("eigenvalue", 1))
    assert value(assess(lower).threshold) == Fraction(1, 8)
    upper = reference.revise(relation="magnitude_upper_bound", fact=fact("eigenvalue", 10))
    assert assess(upper).threshold is None
    for raw in (0, -1):
        assert assess(lower.revise(fact=fact("eigenvalue", raw))).threshold is None
    zero = reference.revise(fact=fact("eigenvalue", 0))
    assert assess(zero).threshold is None
    assert assess(zero, absolute_fallback=.5).status == "PASS"
    assert assess().threshold is None
    assert assess(reference.revise(fact=fact("eigenvalue", 2, kind="observed"))).threshold is None
    with pytest.raises(ValueError, match="relative"):
        original.assess(Accuracy(absolute_tolerance=.5), context=CONTEXT, absolute_fallback=.5)


def test_point_restrictions_and_unresolved_premises_survive_export():
    from nwqlib.ir import Binding
    binding = Binding(parameter="n", value=4)
    original = model(physical=0)
    fixed = fact("sampling", 0, bindings=(binding,))
    point = CONTEXT.revise(bindings=(binding,))
    assert original.assess(Accuracy(absolute_tolerance=.1), context=point, facts=(fixed,)).status == "PASS"
    with pytest.raises(ValueError, match="conflicting"):
        original.assess(Accuracy(absolute_tolerance=.1), context=CONTEXT.revise(bindings=(binding.revise(value=1),)), facts=(fixed,))
    unresolved = fixed.revise(fact=fixed.fact.revise(assumptions=("unverified model premise",)))
    result = original.assess(Accuracy(absolute_tolerance=.1), context=point, facts=(unresolved,))
    copied = FramedFact.model_validate_json(result.facts[-1].model_dump_json())
    assert copied.bindings == (binding,) and copied.fact.assumptions == ("unverified model premise",)
    assert original.assess(Accuracy(absolute_tolerance=.1), context=point, facts=(copied,)).status == "INCONCLUSIVE"


def test_check_domain_and_signed_metric_are_distinct_from_evidence_status():
    check = CheckSpec(name="residual", claim_id=CONTEXT.result_id, frame=FRAME, domain=CheckDomain(lower=0),
        source=SOURCE, threshold=.1, prerequisites=(), access=("supplied scalar",), experiments=0,
        classical_work="one scalar comparison", reference_work="none", data_description="one framed scalar")
    assert assemble_check(check, artifact_id=CONTEXT.result_id).status == "NOT_RUN"
    for number, status in ((0, "PASS"), (Fraction(1, 10**13), "PASS"), (1, "FAIL")):
        result = assemble_check(check, artifact_id=CONTEXT.result_id, fact=fact("residual", number, kind="observed"))
        assert result.status == status and result.fact.fact.evidence.kind == "observed"
    for number in (-1, Fraction(-1, 10**13)):
        with pytest.raises(ValueError, match="nonnegative"):
            assemble_check(check, artifact_id=CONTEXT.result_id, fact=fact("residual", number))
    signed = check.revise(frame=FRAME.revise(metric="signed_value"), domain=CheckDomain())
    assert assemble_check(signed, artifact_id=CONTEXT.result_id,
        fact=fact("residual", -1, frame=signed.frame)).status == "PASS"
    bounded = check.revise(domain=CheckDomain(lower=0, upper=1, integer=True))
    with pytest.raises(ValueError, match="integer"):
        assemble_check(bounded, artifact_id=CONTEXT.result_id, fact=fact("residual", Fraction(1, 2)))


def covariance_case():
    frame = FRAME.revise(metric="variance", unit=Unit(symbol="hartree^2", dimension="custom"))
    x = EstimatorContribution(data_id="sha256:" + "a"*64, coefficient=1., variance=fact("variance", 3, frame=frame))
    y = x.revise(data_id="sha256:" + "b"*64)
    return frame, x, y


def test_covariance_counts_actual_variables_and_keeps_conditioning():
    frame, x, y = covariance_case()
    kwargs = dict(joint_id=CONTEXT.result_id, frame=frame)
    assert value(linear_variance(contributions=(x, x), **kwargs).value) == 12  # Var(2X)=4*3.
    assert linear_variance(contributions=(x, y), **kwargs).value.fact.availability == "unknown"
    law = IndependenceLaw(joint_id=CONTEXT.result_id, frame=frame, bindings=(), evidence=Evidence(kind="user_assertion", source=SOURCE))
    result = linear_variance(contributions=(x, y), independence=law, **kwargs)
    assert value(result.value) == 6 and result.value.fact.evidence.kind == "user_assertion"
    cross = CrossCovariance(left=x.data_id, right=y.data_id, fact=fact("covariance", 1, frame=frame))
    assert value(linear_variance(contributions=(x, y), covariance=(cross,), **kwargs).value) == 8
    unknown = x.revise(variance=fact("variance", frame=frame))
    canceled = linear_variance(contributions=(unknown, unknown.revise(coefficient=-1.)), **kwargs).value
    assert value(canceled) == 0 and canceled.fact.assumptions == ()
    # Independent signs have sum-square values (4,0,0,4); conditioning on X=Y
    # leaves (4,4), so variance changes from 2 to 4.
    conditional = frame.revise(conditioning="X=Y")
    a, b = (item.revise(variance=fact("variance", 1, frame=conditional)) for item in (x, y))
    cross = cross.revise(fact=fact("covariance", 1, frame=conditional))
    result = linear_variance(joint_id=CONTEXT.result_id, frame=conditional, contributions=(a, b), covariance=(cross,))
    assert value(result.value) == Fraction(4 + 4, 2) == 4
    with pytest.raises(ValueError, match="conditioning"):
        linear_variance(joint_id=CONTEXT.result_id, frame=conditional, contributions=(a, b), independence=law)


def test_impossible_covariance_and_negative_variance_reject_without_roundoff_fiction():
    frame, x, y = covariance_case()
    kwargs = dict(joint_id=CONTEXT.result_id, frame=frame)
    for cross in (4, 3 + Fraction(1, 10**13)):
        with pytest.raises(ValueError, match="variance product"):
            linear_variance(contributions=(x, y), covariance=(CrossCovariance(left=x.data_id, right=y.data_id,
                fact=fact("covariance", cross, frame=frame)),), **kwargs)
    for raw in (Rational(numerator=-1, denominator=10**13), Float64(value=-1e-13)):
        invalid = x.revise(variance=x.variance.revise(fact=x.variance.fact.revise(value=raw)))
        with pytest.raises(ValueError, match="nonnegative"):
            linear_variance(contributions=(invalid, invalid.revise(coefficient=-1.)), **kwargs)
    a, b = .3, .7
    assert Fraction(a*b)**2 > Fraction(a*a)*Fraction(b*b)
    with pytest.raises(ValueError, match="variance product"):
        linear_variance(contributions=(x.revise(variance=fact("variance", a*a, frame=frame)),
            y.revise(variance=fact("variance", b*b, frame=frame))), covariance=(CrossCovariance(left=x.data_id,
            right=y.data_id, fact=fact("covariance", a*b, frame=frame)),), **kwargs)
    for number in (0, Fraction(1, 10**13)):
        assert value(linear_variance(contributions=(x.revise(variance=fact("variance", number, frame=frame)),), **kwargs).value) == number
    x, y, z = (x.revise(data_id="sha256:" + letter*64, variance=fact("variance", 1, frame=frame)) for letter in "abc")
    pairs = tuple(CrossCovariance(left=a.data_id, right=b.data_id, fact=fact("covariance", Fraction(-3, 4), frame=frame))
                  for a, b in ((x, y), (x, z), (y, z)))
    # Every pair satisfies Cauchy-Schwarz, but Var(X+Y+Z)=3-6*(3/4)<0.
    with pytest.raises(ValueError, match="negative variance"):
        linear_variance(contributions=(x, y, z), covariance=pairs, **kwargs)
    assert linear_variance(contributions=(x, y, z), covariance=pairs[:2], **kwargs).value.fact.availability == "unknown"


def test_exact_integer_guard_precedes_arithmetic_and_preserves_small_cases(monkeypatch):
    arithmetic = ExactArithmetic(max_integer_bits=64)
    conversions = []
    original = Fraction.__new__
    def converted(cls, *args, **kwargs):
        conversions.append(args)
        return original(cls, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(Fraction, "__new__", converted)
        with pytest.raises(ValueError, match="before operation"):
            arithmetic.fraction(2**64)
    assert conversions == []
    left, right = Fraction(2**40), Fraction(2**40)
    multiplications = []
    multiply = Fraction.__mul__
    def multiplied(a, b):
        multiplications.append((a, b))
        return multiply(a, b)
    with monkeypatch.context() as scoped:
        scoped.setattr(Fraction, "__mul__", multiplied)
        with pytest.raises(ValueError, match="max_integer_bits"):
            arithmetic.multiply(left, right)
    assert multiplications == []
    assert arithmetic.add(Fraction(1, 8), Fraction(1, 4)) == Fraction(3, 8)
    assert arithmetic.divide(Fraction(-3, 8), Fraction(3, 4)) == Fraction(-1, 2)
    smallest = float.fromhex("0x0.0000000000001p-1022")
    assert ExactArithmetic().fraction(smallest) == Fraction(1, 2**1074)


def test_independent_scalar_bound_integer_boundary_and_conditions():
    from nwqlib.evidence import resolve_scalar_bound

    for decay, needed in (("inverse", 4), ("inverse_sqrt", 16)):
        result = resolve_scalar_bound(1, Fraction(3, 8), fixed_error=Fraction(1, 8),
                                      decay=decay, maximum=100, work_per_unit=3)
        assert result.integer == needed
        assert result.covered_bound <= Fraction(3, 8)
        assert result.declared_work == 3 * needed
        assert resolve_scalar_bound(1, Fraction(3, 8), fixed_error=Fraction(1, 8),
                                    decay=decay, maximum=needed - 1).integer is None
        assert resolve_scalar_bound(1, Fraction(3, 8), fixed_error=Fraction(1, 8),
                                    decay=decay, maximum=100, fixed=needed - 1).integer is None
        assert resolve_scalar_bound(1, Fraction(3, 8), fixed_error=Fraction(1, 8),
                                    decay=decay, maximum=100, fixed=needed).integer == needed
    conditional = resolve_scalar_bound(1, 1, maximum=4, conditions=(None,), assumptions=("supplied law",))
    assert conditional.integer == 1 and conditional.status == "conditional"
    assert conditional.to_dict()["conditions"] == [None]
    assert resolve_scalar_bound(1, 1, maximum=4, conditions=(False,)).status == "inapplicable"
    assert resolve_scalar_bound(None, 1, maximum=4).status == "unresolved"
    assert resolve_scalar_bound(0, 0, maximum=4).integer == 1
    assert resolve_scalar_bound(1, 0, maximum=4).integer is None
    assert resolve_scalar_bound(0, 0, fixed_error=1, maximum=4).integer is None


def test_scalar_bound_rejects_wrong_domain_frame_and_prospective_arithmetic():
    from nwqlib.evidence import resolve_scalar_bound

    for kwargs in ({"coefficient": -1}, {"tolerance": float("nan")}, {"fixed_error": -1},
                   {"maximum": True}, {"minimum": 0}, {"fixed": 5}, {"work_per_unit": -1}):
        values = dict(coefficient=1, tolerance=1, maximum=4)
        values.update(kwargs)
        with pytest.raises(ValueError):
            resolve_scalar_bound(**values)
    other = FRAME.revise(unit=Unit(symbol="m", dimension="custom"))
    with pytest.raises(ValueError, match="metric, unit"):
        resolve_scalar_bound(fact("c", 1), fact("epsilon", 1, frame=other), maximum=4)
    assert resolve_scalar_bound(fact("c", 1), fact("epsilon", Fraction(1, 4)), maximum=4).integer == 4
    with pytest.raises(ValueError, match="before operation"):
        resolve_scalar_bound(1, Fraction(1, 2**32), maximum=2**40, decay="inverse_sqrt", max_integer_bits=48)


def test_replacing_bound_never_inherits_displaced_confidence():
    original = model(physical=0, sampling=Fraction(1, 100))
    accuracy = Accuracy(absolute_tolerance=.1, component='sampling', confidence=.95)
    unknown = fact('sampling', Fraction(1,100), failure_probability=None)
    assert original.assess(accuracy,context=CONTEXT,facts=(unknown,)).status == 'INCONCLUSIVE'
    weaker = fact('sampling', Fraction(1,100), failure_probability=.1)
    assert original.assess(accuracy,context=CONTEXT,facts=(weaker,)).status == 'INCONCLUSIVE'
    known = fact('sampling', Fraction(1,100), failure_probability=.01)
    checked = original.assess(accuracy,context=CONTEXT,facts=(known,))
    assert checked.status == 'PASS'
    assert checked.facts[0].failure_probability == .01
    assert original.assess(Accuracy(absolute_tolerance=.1,component='sampling',confidence=.999),
        context=CONTEXT,facts=checked.facts).status == 'INCONCLUSIVE'


def test_criterion_free_result_keeps_its_actual_error_fact():
    from nwqlib import Expectation, solve
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators.inputs import ingest_pauli
    result = solve(Expectation(state=[1,0],observable=ingest_pauli([('I',1.)],num_qubits=1)),method=ExpectationMethod())
    frame = result.plan.output.frame(result.plan.problem)
    statement = FramedFact(frame=frame,bindings=(),failure_probability=0.,fact=Fact(
        quantity='sampling',unit=frame.unit,scope=frame.scope,availability='concrete',value=Rational(numerator=0,denominator=1),
        evidence=Evidence(kind='proved_relation',source=SOURCE,status='witnessed',
            artifact='identity observable is the constant one; zero acquired random settings',
            witnessed_scope=frame.scope,subject_id=result.observation_id)))
    checked = result.revise(facts=(statement,))._attach(result.plan,result.data)
    assert not checked.data.observations.chunks
    assessment = checked.assess(absolute_tolerance=.01,component='sampling')
    assert assessment.status == 'PASS'
    assert checked.plan is result.plan and checked.data is result.data


def _unit_cases():
    import numpy as np
    from nwqlib import Eigenproblem, LinearDynamics, LinearSystem
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Eigenphase, Eigenvalue, NormalizedExpectation, NormSquared, Samples, StateVector

    energy = np.diag([-1.137, 0.5])
    system = dict(A=[[1.1, 0.1], [0.1, 0.9]], b=[1.0, 0.25])
    conflicts = (
        (Eigenproblem(A=energy, unit="Hartree"), Eigenvalue(unit="eV"), "Hartree"),
        (Eigenproblem(A=energy), Eigenphase(unit="rad"), "turn"),
        (LinearSystem(**system), Samples(unit="count"), "1"),
        (LinearSystem(**system, unit="m"), StateVector(unit="m"), "1"),
        (LinearSystem(**system, unit="m"), NormSquared(unit="m"), "(m)^2"),
    )
    observable = ingest_pauli((("Z", 1.0),), num_qubits=1)
    legal = (
        (Eigenproblem(A=energy, unit="Hartree"), Eigenvalue(unit="Hartree"), "Hartree"),
        (Eigenproblem(A=energy), Eigenphase(unit="turn"), "turn"),
        (Eigenproblem(A=energy), Eigenvalue(unit="eV"), "eV"),
        (LinearDynamics(A=np.eye(2), initial_state=[1, 0], time=0.1, unit="m"),
         NormalizedExpectation(observable=observable, unit="K"), "K"),
    )
    return conflicts, legal


def test_output_unit_conflicting_with_a_defined_unit_raises_before_planning():
    """An output unit relabels an unconverted value, so a conflict with the Problem's unit is an input error."""
    from nwqlib import plan
    from nwqlib.algorithms.lanczos import Lanczos

    conflicts, legal = _unit_cases()
    for problem, output, defined in conflicts:
        with pytest.raises(ValueError, match=re.escape(f"conflicts with the unit '{defined}'")):
            output.frame(problem)
    problem, output, _ = conflicts[0]
    with pytest.raises(ValueError, match="conflicts with the unit 'Hartree'"):
        plan(problem, method=Lanczos(initial_state=(0.8, 0.6), krylov_dimension=2), output=output,
             execution="classical", seed=1)
    # The same symbol, or a label for a quantity without a defined unit, stays legal.
    for problem, output, expected in legal:
        assert output.frame(problem).unit.symbol == expected
    problem, output, _ = legal[0]
    selected = plan(problem, method=Lanczos(initial_state=(0.8, 0.6), krylov_dimension=2), output=output,
                    execution="classical", seed=1)
    assert selected.error_model.frame.unit.symbol == "Hartree"


def _sampling_runs(shots=None):
    """Return one small run per estimator, with exact readout unless shots is given.

    QCELS measures a fixed schedule of powers, and RWPE without an update
    reports its prior. Classical RWPE emulates one random bit from each
    exact signal, and SPE and RFE draw their frequencies or powers at random,
    so their estimates depend on random draws even with exact readout.
    """
    import numpy as np
    import sympy as sp
    from nwqlib import Eigenproblem, Expectation, LinearDynamics, LinearSystem, Optimization, solve
    from nwqlib.algorithms import LCHS, ExpectationMethod
    from nwqlib.algorithms.qpe import QCELS, RFE, RWPE, SPE
    from nwqlib.algorithms.gcim import FixedGCIM
    from nwqlib.algorithms.lanczos import Lanczos
    from nwqlib.algorithms.qhd import QHD
    from nwqlib.algorithms.qls import QLS
    from nwqlib._prepared_execution import Run
    from test_adapt_primary import plan_for as adapt_plan

    x = sp.Symbol("x", real=True)
    energy = np.array([[1.5, -1.0], [-1.0, 0.5]])
    return dict(
        expectation=lambda: solve(Expectation(state=[2.0, 2.0], observable=[[2.0, 0.0], [0.0, 0.0]]),
                                  method=ExpectationMethod(), shots=shots, seed=7),
        lanczos=lambda: solve(Eigenproblem(A=energy), method=Lanczos(initial_state=[1, 0], krylov_dimension=2),
                              shots=shots, seed=7),
        fixed_gcim=lambda: solve(Eigenproblem(A=energy), method=FixedGCIM(basis=([1.0, 0.0], [0.0, 1.0])),
                                 shots=shots, seed=7),
        qls=lambda: solve(LinearSystem(A=[[1.1, 0.1], [0.1, 0.9]], b=[1.0, 0.25]), method=QLS(), seed=7),
        qhd=lambda: solve(Optimization(objective=(x - 0.2) ** 2, variables=(x,), bounds=((-1.0, 1.0),)),
                          method=QHD(), seed=7),
        adapt=lambda: Run(adapt_plan(execution="classical", initial_state=(1, 0))).wait(timeout=5, poll_interval=0),
        lchs=lambda: solve(LinearDynamics(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1, 0], time=0.1),
                           method=LCHS(), seed=7),
        qcels=lambda: solve(Eigenproblem(A=np.diag([0.2, 0.7])), method=QCELS(initial_state=[1, 0]),
                            shots=shots, seed=7),
        rwpe_prior=lambda: solve(Eigenproblem(A=np.diag([0.2, 0.7])),
                                 method=RWPE(initial_state=[1, 0], tau=1.0, max_steps=0),
                                 execution="classical", seed=7),
        rwpe=lambda: solve(Eigenproblem(A=np.diag([0.2, 0.7])),
                           method=RWPE(initial_state=[1, 0], tau=1.0, max_steps=1, prior_std=1.0),
                           execution="classical", seed=7),
        spe=lambda: solve(Eigenproblem(A=np.diag([0.2, 0.7])),
                          method=SPE(initial_state=[1, 0], overlap_lower_bound=1.0),
                          execution="classical", seed=7),
        rfe=lambda: solve(Eigenproblem(A=np.diag([0.2, 0.7])),
                          method=RFE(initial_state=[1, 0], tau=1.0, num_samples=3,
                                     num_frequencies=16),
                          execution="classical", seed=7),
    )


@pytest.mark.parametrize("family,status", [
    ("expectation", "PASS"), ("lanczos", "PASS"), ("fixed_gcim", "PASS"), ("adapt", "PASS"),
    ("lchs", "PASS"), ("qcels", "PASS"), ("rwpe_prior", "PASS"),
    ("qls", "NOT_APPLICABLE"), ("qhd", "NOT_APPLICABLE"),
    ("rwpe", "INCONCLUSIVE"), ("spe", "INCONCLUSIVE"), ("rfe", "INCONCLUSIVE"),
])
def test_exact_readout_sampling_is_zero_only_without_random_draws(family, status):
    """Exact readout proves zero sampling error only for an estimator that makes no random draw.

    A random draw of the estimator itself, such as an emulated measurement
    outcome or a random schedule, keeps sampling unknown. A family that
    declares no sampling source is not applicable.
    """
    result = _sampling_runs()[family]()
    assert result.plan.shots is None
    sampling = result.assess(component="sampling", absolute_tolerance=1e-12)
    assert sampling.status == status
    if status == "PASS":
        assert sampling.remaining == () and sampling.covered == ("sampling",)
        assert value(sampling.covered_subtotal) == 0
        assert sampling.failure_probability is not None and value(sampling.failure_probability) == 0
    elif status == "NOT_APPLICABLE":
        assert sampling.remaining == () and sampling.covered == ()
        assert "sampling" not in result.plan.error_model.required_sources
    else:
        assert sampling.remaining == ("sampling",) and sampling.covered == ()
        assert all(fact.fact.quantity != "sampling" for fact in result.facts)
    # A zero covers only sampling. Total error keeps the other unknown sources.
    total = result.assess(absolute_tolerance=1.0)
    assert total.status == "INCONCLUSIVE"
    assert ("sampling" in total.remaining) == (status == "INCONCLUSIVE")


@pytest.mark.parametrize("family", ["expectation", "lanczos", "fixed_gcim", "qcels"])
def test_finite_shots_keep_sampling_required(family):
    """Drawn outcomes never receive the exact-readout zero."""
    result = _sampling_runs(shots=64)[family]()
    assert all(fact.fact.quantity != "sampling" or fact.fact.evidence.kind != "proved_relation"
               for fact in result.facts)
    sampling = result.assess(component="sampling", absolute_tolerance=1e-12)
    assert sampling.status == "INCONCLUSIVE"
