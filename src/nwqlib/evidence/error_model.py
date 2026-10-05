"""Framed scalar evidence and sufficient accuracy criteria, without acquisition.

The triangle inequality combines already propagated contributions in one target
frame. A receipt cannot change that frame, admit a parameter point, or turn a
PREP-on-zero error or residual into a full-operator or ground-state bound.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, PrivateAttr, StrictBool, model_validator

from nwqlib.core.records import (
    ContentID, Float64, Nonnegative, Rational, Real, Record, Scope, Source, Stage, Text, Unit,
    _declared_fields_equal,
)
from nwqlib.ir.expressions import Binding, number
from nwqlib.operators.access import Count
from nwqlib.problems.records import Accuracy
from .records import Evidence, Fact
from ._work import DEFAULT_MAX_INTEGER_BITS, ExactArithmetic

ClaimStatus = Literal["PASS", "INCONCLUSIVE", "NOT_APPLICABLE"]
CheckStatus = Literal["PASS", "FAIL", "NOT_RUN", "INCONCLUSIVE", "NOT_APPLICABLE"]

def _same_scope(left, right):
    return left is not None and right is not None and (left.kind, left.domain) == (right.kind, right.domain)


class ErrorFrame(Record):
    """What an error value refers to: quantity, metric, unit, scope, conditions and domain.

    Every error bound and check carries one, and the frame of a Result's
    output is `result.plan.error_model.frame`. A value answers a claim only
    when the two frames are compatible, which requires all six fields to
    agree in meaning, with units compared by symbol and dimension and scopes
    by kind and domain. A value for the projected problem or for a zero input therefore
    cannot answer a claim about the target quantity, and missing context is
    never read as an absence of conditions. Build one with keyword arguments
    to state a value you supply. Every field except `domain` is required.

    Attributes:
        quantity: Required. Name of the quantity the error describes.
        metric: Required. Error metric, for example an absolute difference
            or `variance`.
        unit: Required. Unit of the metric's value.
        scope: Required. Scope of the Problem the quantity belongs to.
        conditioning: Required. Conditions under which the value is
            meaningful.
        domain: Default `"target"`, the requested quantity.
            `"full_operator"` is a property of the whole operator,
            `"projected"` one of the projected problem of a subspace method,
            and `"zero_input"` the error of a preparation applied to the
            all-zero input.
    """

    quantity: Text
    metric: Text
    unit: Unit
    scope: Scope
    conditioning: Text
    domain: Literal["target", "full_operator", "projected", "zero_input"] = "target"

    def compatible(self, other):
        """Compare scientific dimensions, independently of record ancestry."""
        return ((self.quantity, self.metric, self.conditioning, self.domain)
                == (other.quantity, other.metric, other.conditioning, other.domain)
                and self.unit.same_unit(other.unit) and _same_scope(self.scope, other.scope))

    @classmethod
    def from_output(cls, problem, output):
        """Use the scientific output owner without constructing an accuracy request."""
        frame = output.frame(problem)
        if type(frame) is not cls:
            raise TypeError("output.frame(problem) must return its actual ErrorFrame")
        return frame


class FramedFact(Record):
    """A [`Fact`][nwqlib.evidence.records.Fact] tied to its [`ErrorFrame`][nwqlib.evidence.error_model.ErrorFrame] and parameter point.

    `Result.facts`, `ClaimAssessment.facts` and the `facts` that
    `Result.verify` returns hold them. A value is used only in its own frame
    and at its own parameter point, and the open assumptions of its fact
    stay attached when an assessed value is reused. Build one with keyword
    arguments to supply a bound to `Result.assess(facts=...)`. `frame`,
    `bindings` and `fact` are required.

    Attributes:
        frame: Required. The frame of the value.
        bindings: Required. Parameter values the statement is restricted
            to. `()` declares no restriction.
        fact: Required. The value. Its `quantity` names the role the value
            plays for the code that reads it, for example the `sampling`
            source of an expectation value.
        failure_probability: Default `None`, which means unknown. Probability
            in `[0, 1)` that this bound fails. A bound supplied in place of
            another does not inherit the other's probability, so a
            replacement without one has unknown confidence.

    Raises:
        ValueError: If the fact's unit or scope differs from the frame's, or
            a parameter is restricted twice.
    """

    frame: ErrorFrame
    bindings: tuple[Binding, ...]
    fact: Fact
    failure_probability: Annotated[Real, Field(ge=0, lt=1)] | None = None

    @model_validator(mode="after")
    def _context(self):
        if (not self.frame.unit.same_unit(self.fact.unit)
                or not _same_scope(self.frame.scope, self.fact.scope)):
            raise ValueError("framed fact unit/scope must match its explicit frame")
        if len({b.parameter for b in self.bindings}) != len(self.bindings):
            raise ValueError("framed fact parameter restrictions must be unique")
        return self


class AssessmentContext(Record):
    """Role-bearing lineage and point, admitted only by actual Plan readiness.

    observation_id names the aggregate ObservationView; contribution_ids name
    its chunks. Certificate preserves these roles and appends result_id.
    subjects is derived; selected/result association still needs direct checks.
    """

    problem_id: ContentID
    base_construction_id: ContentID
    selected_construction_id: ContentID | None = None
    plan_id: ContentID | None = None
    observation_id: ContentID | None = None
    contribution_ids: tuple[ContentID, ...] = ()
    result_id: ContentID | None = None
    bindings: tuple[Binding, ...] = ()
    admitted: StrictBool = False
    _subject_index: frozenset | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _roles(self):
        self._subject_index = None
        if len({b.parameter for b in self.bindings}) != len(self.bindings):
            raise ValueError("assessment parameter names must be unique")
        if len(set(self.contribution_ids)) != len(self.contribution_ids):
            raise ValueError("contributing observation identities must be unique")
        if self.admitted and (self.plan_id is None or self.selected_construction_id is None):
            raise ValueError("admitted context requires the actual Plan and selected construction")
        if self.contribution_ids and self.observation_id is None:
            raise ValueError("contributing chunks require their aggregate observation role")
        if self.result_id is not None and self.plan_id is None:
            raise ValueError("result context requires its Plan role")
        return self

    @property
    def subjects(self):
        if self._subject_index is None:
            self._subject_index = frozenset(x for x in (self.problem_id, self.base_construction_id,
                self.selected_construction_id, self.plan_id, self.observation_id,
                *self.contribution_ids, self.result_id) if x is not None)
        return self._subject_index

    # Compare declared fields only, so the lazily cached subject index
    # never makes two equal contexts compare unequal.
    __eq__ = _declared_fields_equal


def compatible_point(bindings, context):
    """Return whether evidence restricted to ``bindings`` applies at the context's point.

    Unrestricted evidence always applies. Restricted evidence applies only
    at a point that the Plan has admitted and that binds every restricted
    parameter to the same value. A differing value raises, even at a point
    that is only proposed, because the evidence then describes another point.
    """
    if context is None:
        return not bindings
    point = {b.parameter: number(b.value) for b in context.bindings}
    for binding in bindings:
        if binding.parameter in point and number(binding.value) != point[binding.parameter]:
            raise ValueError("conflicting evidence parameter restriction and assessment point")
    return not bindings or (context.admitted and all(b.parameter in point for b in bindings))


def applicable(statement, *, frame, context):
    """Central structural frame/point rule, separate from proposition support."""
    if not isinstance(statement, FramedFact):
        raise TypeError("scientific evidence requires a FramedFact")
    if not statement.frame.compatible(frame):
        raise ValueError("evidence frame quantity/metric/unit/scope/conditioning/domain mismatch")
    return compatible_point(statement.bindings, context)


def receipt_supported(evidence, *, scope, subjects, kinds):
    """Check only the receipt and the consumer's explicit proposition kinds."""
    return bool(evidence and evidence.kind in kinds and evidence.status == "witnessed"
                and evidence.subject_id in subjects and _same_scope(evidence.witnessed_scope, scope))


def supported(statement, *, frame, context):
    """Whether a statement can support an accuracy PASS in this frame and context.

    It must apply at the admitted point, carry no open assumption and have
    witnessed proved_relation or certified_bound evidence for a subject of
    the context. Numerical estimates, observations and assertions never
    qualify, however small their value.
    """
    matches = applicable(statement, frame=frame, context=context)
    return (matches and not statement.fact.assumptions and receipt_supported(
        statement.fact.evidence, scope=frame.scope, subjects=context.subjects if context else (),
        kinds={"proved_relation", "certified_bound"}))


def predicate_frame(frame):
    """Dimensionless truth in the same target's conditioning/scope/domain."""
    return ErrorFrame(quantity=frame.quantity, metric="predicate",
        unit=Unit(symbol="1", dimension="dimensionless"), scope=frame.scope,
        conditioning=frame.conditioning, domain=frame.domain)


def predicate_value(statement, *, frame, context):
    """Return supported True/False, or None; never promote receipt kind."""
    expected = predicate_frame(frame)
    valid = supported(statement, frame=expected, context=context)
    return statement.fact.value if valid and type(statement.fact.value) is bool else None


def union_bindings(*groups):
    """Union actually used restrictions; distinct declared values conflict."""
    values = {}
    for group in groups:
        for binding in group:
            old = values.get(binding.parameter)
            if old is not None and number(old.value) != number(binding.value):
                raise ValueError("conflicting parameter restrictions in derived evidence")
            values.setdefault(binding.parameter, binding)
    return tuple(values[key] for key in sorted(values))


def _keep_premises(statement, conditions, inputs, context):
    """Keep used restrictions/unresolved premises when an assessed value leaves its term."""
    if not conditions and not inputs:
        return statement
    premises = (statement, *conditions, *inputs)
    restrictions = union_bindings(*(p.bindings for p in premises))
    assumptions = [a for p in premises for a in p.fact.assumptions]
    for premise in (*conditions, *inputs):
        mathematical = receipt_supported(premise.fact.evidence, scope=premise.frame.scope,
            subjects=context.subjects, kinds={"proved_relation", "certified_bound"})
        if not mathematical or (premise in conditions and premise.fact.value is not True):
            assumptions.append(f"unresolved prerequisite {premise.fact.quantity} ({premise.content_id})")
    fields = {name: getattr(statement.fact, name) for name in Fact.model_fields}
    fields.update(assumptions=tuple(dict.fromkeys(assumptions)), parent_id=statement.fact.content_id)
    fact = Fact(**fields)
    fields = dict(frame=statement.frame, bindings=restrictions, fact=fact, parent_id=statement.content_id)
    return FramedFact(**fields)


def scalar(statement: FramedFact, arithmetic: ExactArithmetic) -> Fraction | None:
    """Read finite real scalar evidence; booleans and complex values are not errors."""
    fact = statement.fact
    if fact.availability != "concrete":
        return None
    if isinstance(fact.value, (Rational, Float64)):
        return arithmetic.fraction(fact.value)
    raise ValueError("error quantities require real scalar facts")


def _rational(value):
    if value is None:
        return None
    fields = dict(numerator=value.numerator, denominator=value.denominator)
    return Rational(**fields)


class TargetReference(Record):
    """A reference value of the target that sets the scale of a relative criterion.

    Pass it as `reference=` to `Result.assess(relative_tolerance=...)`. For
    the criterion `|error| <= r*|target|`, a proved lower bound
    `L <= |target|` gives the sufficient threshold `r*L`, and an exact
    nonzero target gives `r*|target|`. An upper bound on the magnitude or an
    observed
    estimate gives none, and neither does a reference whose failure
    probability is unknown or whose evidence is not witnessed proved or
    certified support. The reference's failure probability joins the union
    bound of the assessment. Build it with keyword arguments. All three
    fields are required.

    Attributes:
        relation: Required. `"exact_target"`, `"magnitude_lower_bound"` or
            `"magnitude_upper_bound"`.
        fact: Required. A [`FramedFact`][nwqlib.evidence.error_model.FramedFact]
            whose quantity is the frame's target quantity. It holds the
            reference's whole frame.
        failure_probability: Required. Probability in `[0, 1)` that the
            reference fails, or `None` when unknown.

    Raises:
        ValueError: If the fact names another quantity or is not a real
            scalar, or a magnitude upper bound, or an exact target of a
            `norm_squared` quantity, is negative.
    """

    relation: Literal["exact_target", "magnitude_lower_bound", "magnitude_upper_bound"]
    fact: FramedFact
    failure_probability: Annotated[Real, Field(ge=0, lt=1)] | None

    @model_validator(mode="after")
    def _target(self):
        if self.fact.fact.quantity != self.fact.frame.quantity:
            raise ValueError("reference fact must name its scientific target quantity")
        value = self.fact.fact.value
        if value is not None and not isinstance(value, (Rational, Float64)):
            raise ValueError("reference targets require real scalar facts")
        if value is not None:
            negative = (value.numerator if isinstance(value, Rational) else value.value) < 0
            if negative and (self.relation == "magnitude_upper_bound" or (
                    self.relation == "exact_target" and self.fact.frame.quantity == "norm_squared")):
                raise ValueError("norm-squared targets and magnitude upper bounds must be nonnegative")
        return self


class ErrorTerm(Record):
    """One error source of an [`ErrorModel`][nwqlib.evidence.error_model.ErrorModel], already propagated into the output's frame.

    Methods build these. `role` is independent of the value's availability
    and evidence, and `coverage` cannot remove another required source of
    the model. The fields below are read-only.

    Attributes:
        name: Name of the source, equal to `fact.fact.quantity`.
        stage: Workflow stage where the error arises.
        source: Where the bound comes from.
        formula: Text form of the bound.
        fact: The bound as a [`FramedFact`][nwqlib.evidence.error_model.FramedFact].
        coverage: Text saying what the bound covers.
        role: Default `"additive_bound"`, a nonnegative bound added in the
            triangle inequality. The other roles, `"amplification"`,
            `"covariance_contribution"`, `"conditional_term"` and
            `"listed_only"`, are listed but not added.
        conditions: Boolean predicates that must hold, in the dimensionless
            predicate frame of this term.
        inputs: Values the bound depends on.
        failure_probability: Probability in `[0, 1)` that this bound fails,
            not a variance. `None` when unknown.

    Raises:
        ValueError: If `name` differs from the fact's quantity, the value is
            not a real scalar, or an additive bound is negative.
    """

    name: Text
    stage: Stage
    source: Source
    formula: Text
    fact: FramedFact
    coverage: Text
    role: Literal["additive_bound", "amplification", "covariance_contribution", "conditional_term", "listed_only"] = "additive_bound"
    conditions: tuple[FramedFact, ...] = ()
    inputs: tuple[FramedFact, ...] = ()
    failure_probability: Annotated[Real, Field(ge=0, lt=1)] | None = None

    @model_validator(mode="after")
    def _frame(self):
        if self.fact.fact.quantity != self.name:
            raise ValueError("error fact must match source name")
        value = self.fact.fact.value
        if value is not None and not isinstance(value, (Rational, Float64)):
            raise ValueError("error quantities require real scalar facts")
        if value is not None and (value.numerator if isinstance(value, Rational) else value.value) < 0 and self.role == "additive_bound":
            raise ValueError("additive bounds must be nonnegative")
        for condition in self.conditions:
            applicable(condition, frame=predicate_frame(self.fact.frame), context=None)
        return self


class ClaimAssessment(Record):
    """The outcome of checking a Result against an accuracy criterion.

    [`Result.assess`][nwqlib.core.analysis.Result.assess] returns it (see
    [Check a result against a
    tolerance](evidence.md#check-a-result-against-a-tolerance)).
    The answer is `status`:

    - `PASS`: every error source the Method lists has a bound that applies
      at the assessed point and, with its conditions and inputs, has
      witnessed proved or certified support and no open assumption. The
      sum of the bounds is at most `threshold`, and their combined failure
      probability meets the requested confidence.
    - `INCONCLUSIVE`: otherwise. `remaining` lists the sources without a
      bound, `unverified` the bounded sources without that support, and
      `prerequisites` the reasons. A sum of bounds above the threshold is
      INCONCLUSIVE, not evidence that the error exceeds the tolerance.
    - `NOT_APPLICABLE`: a `component="sampling"` criterion on an error model
      that lists no sampling source.

    The bounds are added by the triangle inequality and their failure
    probabilities by the union bound, with the assumptions stated under
    [`ErrorModel.assess`][nwqlib.evidence.error_model.ErrorModel.assess]. A
    `component="sampling"` criterion assesses the sampling contribution
    only and never covers total error. The assessment is a separate record, so
    assessing with another criterion changes neither the Result nor its
    Plan. The fields below are read-only.

    Attributes:
        model_id: Content hash of the error model assessed against.
        accuracy: The criterion as an [`Accuracy`][nwqlib.problems.records.Accuracy]:
            tolerance, confidence and component.
        absolute_fallback: Absolute tolerance used only when the relative
            scale cannot be established, or `None`.
        output_id: Content hash of the output whose error is assessed.
        context: Identities of the Problem, circuit description, Plan,
            observations and Result of the assessed data, with the parameter
            point.
        frame: The [`ErrorFrame`][nwqlib.evidence.error_model.ErrorFrame] of
            the output.
        method: How the bounds were combined.
        reference: The [`TargetReference`][nwqlib.evidence.error_model.TargetReference]
            of a relative criterion, or `None`.
        status: `"PASS"`, `"INCONCLUSIVE"` or `"NOT_APPLICABLE"`, as above.
        threshold: Absolute threshold in the output unit, or `None` when a
            relative scale was not established.
        covered_subtotal: Sum of the bounds that entered the triangle
            inequality, an exact rational.
        covered: Sources whose bounds entered the sum.
        remaining: Required sources without an applicable additive bound.
        unverified: Covered sources whose bound, conditions or inputs lack
            witnessed proved or certified support.
        prerequisites: Reasons that prevent PASS.
        failure_probability: Union-bound failure probability of the covered
            bounds and the reference, capped at one, or `None` when any
            probability is unknown.
        facts: The [`FramedFact`][nwqlib.evidence.error_model.FramedFact]
            records used, with their assumptions and failure probabilities.
    """

    model_id: ContentID
    accuracy: Accuracy
    absolute_fallback: Annotated[Real, Field(gt=0)] | None = None
    output_id: ContentID
    context: AssessmentContext
    frame: ErrorFrame
    method: Text
    reference: TargetReference | None = None
    status: ClaimStatus
    threshold: Rational | None
    covered_subtotal: Rational
    covered: tuple[Text, ...]
    remaining: tuple[Text, ...]
    unverified: tuple[Text, ...]
    prerequisites: tuple[Text, ...]
    failure_probability: Rational | None
    facts: tuple[FramedFact, ...]

    @model_validator(mode="after")
    def _domains(self):
        if self.covered_subtotal.numerator < 0 or (self.threshold is not None and self.threshold.numerator < 0):
            raise ValueError("error subtotal and threshold must be nonnegative")
        if self.absolute_fallback is not None and self.accuracy.relative_tolerance is None:
            raise ValueError("absolute_fallback belongs only to an explicit relative assessment")
        failure = self.failure_probability
        if failure is not None and not 0 <= failure.numerator <= failure.denominator:
            raise ValueError("capped union failure probability must lie in [0, 1]")
        return self

    def validate_plan(self, plan):
        """Check original acquisition identities without reassessing a criterion."""
        model = plan.error_model
        if (model is None or self.model_id != model.content_id
                or not self.frame.compatible(ErrorFrame.from_output(plan.problem, plan.output))
                or self.frame != model.frame or self.output_id != plan.output.content_id
                or self.context.problem_id != plan.problem.content_id
                or self.context.plan_id != plan.content_id
                or self.context.base_construction_id != plan._construction_id):
            raise ValueError("claim belongs to another Plan, ErrorModel or scientific output")
        return self


class ErrorModel(Record):
    """The error sources a Method lists for its output, each already propagated into the output's frame.

    `result.plan.error_model` holds it, and `Result.assess` calls `assess`
    with `result.facts`. The Method lists every error source its output
    needs, including any subspace or ground-state identification. The
    library does not check that list for completeness. The fields below are
    read-only.

    Attributes:
        output_id: Content hash of the output the model describes.
        subject_id: Content hash of the Problem.
        construction_id: Content hash of the planned circuit description.
        frame: The [`ErrorFrame`][nwqlib.evidence.error_model.ErrorFrame]
            of the output, shared by every additive term.
        required_sources: Names of the sources a total-error claim needs.
        terms: The [`ErrorTerm`][nwqlib.evidence.error_model.ErrorTerm]
            records.
        source: Where the model comes from.

    Raises:
        ValueError: If source names repeat, no source is required, or an
            additive term has another frame.
    """

    output_id: ContentID
    subject_id: ContentID
    construction_id: ContentID
    frame: ErrorFrame
    required_sources: tuple[Text, ...]
    terms: tuple[ErrorTerm, ...]
    source: Source

    @model_validator(mode="after")
    def _inventory(self):
        if len({t.name for t in self.terms}) != len(self.terms) or len(set(self.required_sources)) != len(self.required_sources):
            raise ValueError("error source names must be unique")
        if not self.required_sources:
            raise ValueError("a concrete claim needs its necessary source inventory")
        for term in self.terms:
            if term.role == "additive_bound" and not term.fact.frame.compatible(self.frame):
                raise ValueError("additive terms require a common target frame; supply justified propagation first")
        return self

    def assess(self, accuracy, *, context, facts=(), reference=None, absolute_fallback=None, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
        """Assess an accuracy criterion against the model's bounds at an existing parameter point, measuring nothing.

        Most callers use `Result.assess`, which supplies the context and the
        error bounds in `result.facts`. Two rules combine the bounds, each
        with its assumption:

        - Triangle inequality. Its assumption, which the code does not check,
          is that the Method has propagated every source into this output's
          frame, so the output error is the sum `e = sum_k e_k` of the
          propagated source errors. The code checks only that the frames
          agree. Where each bound `|e_k| <= b_k` holds, `|e| <= sum_k b_k`.
          The sum covers the error only when every required source has a
          bound, so a missing source keeps the result INCONCLUSIVE.
        - Union bound. If bound k fails with probability at most `delta_k`,
          all bounds, and a relative reference, hold together with
          probability at least `1 - sum_k delta_k`, whatever their
          dependence, so no independence is assumed. Confidence c needs
          `sum_k delta_k <= 1 - c`, and an unknown `delta_k` makes the total
          unknown.

        PASS also requires the sum to be at most the threshold and every
        covered bound to apply at the assessed point, carry no open
        assumption and have witnessed `proved_relation` or `certified_bound`
        evidence for a subject of the context. Numerical estimates,
        observations and assertions never qualify, however small their
        value. `component="sampling"` assesses that contribution only, so
        missing hardware, preparation or model terms cannot give it
        total-error coverage. A model that lists no sampling source, as QLS
        and QHD do with exact readout, has no sampling contribution to bound,
        so that criterion is NOT_APPLICABLE rather than missing.

        Args:
            accuracy (Accuracy): The criterion.
            context (AssessmentContext): Identities and parameter point of
                the assessed data. `Result.assess` builds it from the Result.
            facts (tuple[FramedFact, ...]): Bounds that replace the model's
                terms of the same name.
            reference (TargetReference | None): Scale of a relative
                criterion.
            absolute_fallback (float | None): Positive absolute tolerance
                used only when the relative scale cannot be established.
                Requires a relative criterion.
            max_integer_bits (int): Bit limit of the exact arithmetic.
                Default 4096.

        Returns:
            assessment (ClaimAssessment): The status with its threshold,
                covered sum, covered, remaining and unverified sources,
                reasons and capped failure probability.

        Raises:
            TypeError: If `accuracy`, `context` or `reference` has another
                type.
            ValueError: If the context belongs to another Problem or circuit
                description, a supplied fact names no source or repeats one,
                a bound does not apply in its frame or is a negative
                additive bound, or `absolute_fallback` is not positive or
                lacks a relative criterion.
        """
        if type(accuracy) is not Accuracy or type(context) is not AssessmentContext:
            raise TypeError("assessment requires Accuracy and its actual AssessmentContext")
        if (context.problem_id != self.subject_id or context.base_construction_id != self.construction_id):
            raise ValueError("model and assessment belong to different scientific inputs or construction")
        if reference is not None and type(reference) is not TargetReference:
            raise TypeError("relative reference requires an explicit TargetReference relation")
        arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
        if absolute_fallback is not None:
            if accuracy.relative_tolerance is None or arithmetic.fraction(absolute_fallback) <= 0:
                raise ValueError("absolute_fallback must be positive and requires an explicit relative assessment")
        supplied = {f.fact.quantity: f for f in facts}
        if len(supplied) != len(facts) or not supplied.keys() <= {t.name for t in self.terms}:
            raise ValueError("supplied error facts require unique existing source names")
        # Admission applies even to a supplied contribution outside the selected
        # component. A narrow criterion cannot conceal an invalid bound or frame.
        for term in self.terms:
            statement = supplied.get(term.name, term.fact)
            applicable(statement, frame=term.fact.frame, context=context)
            value = scalar(statement, arithmetic)
            if value is not None and value < 0 and term.role == "additive_bound":
                raise ValueError("supplied additive bound is negative")
        # A component criterion reads only its own term. It can never cover
        # total error, which needs every required source.
        terms = self.terms if accuracy.component == "total" else tuple(t for t in self.terms if t.name == "sampling")
        undeclared = (accuracy.component == "sampling" and not terms and "sampling" not in self.required_sources)
        needed = set(self.required_sources if accuracy.component == "total" else () if undeclared else ("sampling",))
        needed.update(t.name for t in terms if t.role == "additive_bound")
        covered, remaining, unverified, used, reasons = [], [], [], [], []
        if undeclared:
            reasons.append("sampling: this error model declares no sampling source")
        if not context.admitted and context.bindings:
            reasons.append("parameter point is proposed; Plan admission is required")
        total, failure = arithmetic.fraction(0), arithmetic.fraction(0)
        failure_known = True
        for term in terms:
            statement = supplied.get(term.name, term.fact)
            if statement.fact.quantity != term.name:
                raise ValueError("supplied fact does not match the term source")
            matches = applicable(statement, frame=term.fact.frame, context=context)
            value = scalar(statement, arithmetic)
            conditions = tuple(predicate_value(c, frame=term.fact.frame, context=context) for c in term.conditions)
            inputs = tuple(supported(c, frame=c.frame, context=context) for c in term.inputs)
            if term.role != "additive_bound" or value is None:
                if term.name in needed:
                    remaining.append(term.name)
                    reasons.append(term.name + ": " + (statement.fact.reason or "no applicable additive bound"))
                continue
            probability = statement.failure_probability
            if term.name not in supplied and probability is None:
                probability = term.failure_probability
            # A replacement bound brings its own confidence. Inheriting the
            # displaced bound's failure probability would relabel old evidence.
            kept = _keep_premises(statement, term.conditions, term.inputs, context)
            if kept.failure_probability != probability:
                kept = kept.revise(failure_probability=probability)
            used.append(kept)
            # Triangle inequality. The ErrorModel validator already requires
            # every additive term to share this output frame.
            total = arithmetic.add(total, value)
            covered.append(term.name)
            if not (matches and supported(statement, frame=self.frame, context=context)
                    and all(v is True for v in conditions) and all(inputs)):
                unverified.append(term.name)
            for condition, value in zip(term.conditions, conditions, strict=True):
                if value is not True:
                    reasons.append(f"{term.name}: prerequisite {condition.fact.quantity} is not supported true")
            # Union bound over the failure events of the covered bounds. It
            # holds for dependent events, and one unknown probability makes
            # the total unknown.
            if probability is None:
                failure_known = False
            else:
                failure = arithmetic.add(failure, arithmetic.fraction(probability))
        remaining.extend(sorted(needed - {t.name for t in terms}))
        # A source without a bound has no stated failure probability either.
        failure_known &= not remaining
        threshold, reference_reasons, reference_failure = accuracy_threshold(accuracy,
            frame=self.frame, reference=reference, context=context, arithmetic=arithmetic,
            absolute_fallback=absolute_fallback)
        reasons.extend(reference_reasons)
        # A relative threshold holds only when the reference bound holds, so
        # its failure probability joins the same union.
        failure = arithmetic.add(failure, reference_failure)
        requested_failure = arithmetic.subtract(arithmetic.fraction(1), arithmetic.fraction(accuracy.confidence))
        if not failure_known or not arithmetic.le(failure, requested_failure):
            reasons.append("required failure probability is unsupported")
        within = threshold is not None and arithmetic.le(total, threshold)
        if threshold is not None and not within:
            reasons.append("covered upper bound exceeds the sufficient accuracy threshold")
        one = arithmetic.fraction(1)
        bounded_failure = failure if arithmetic.le(failure, one) else one
        return ClaimAssessment(model_id=self.content_id, accuracy=accuracy, absolute_fallback=absolute_fallback,
            output_id=self.output_id, context=context, frame=self.frame,
            method="triangle inequality in the selected output component; union bound on declared failures",
            reference=reference, status="NOT_APPLICABLE" if undeclared else
                "PASS" if not reasons and not remaining and not unverified and within else "INCONCLUSIVE",
            threshold=_rational(threshold), covered_subtotal=_rational(total), covered=tuple(covered),
            remaining=tuple(remaining), unverified=tuple(unverified), prerequisites=tuple(reasons),
            failure_probability=_rational(bounded_failure) if failure_known else None, facts=tuple(used))


def accuracy_threshold(accuracy, *, frame, reference, context, arithmetic, absolute_fallback=None):
    """Use the selected absolute criterion or a justified nonzero relative scale.

    For |error| <= r*|target|, a proved lower bound L <= |target| supplies the
    sufficient threshold r*L. An upper bound or an observed estimate does not.
    An explicitly selected absolute fallback is a separate post-run criterion
    used only when that relative scale cannot be established.
    """
    zero = arithmetic.fraction(0)
    if reference is not None:
        applicable(reference.fact, frame=frame, context=context)
    if accuracy.absolute_tolerance is not None:
        return arithmetic.fraction(accuracy.absolute_tolerance), (), zero
    if reference is not None:
        target = scalar(reference.fact, arithmetic)
        if (target is not None and reference.failure_probability is not None
                and supported(reference.fact, frame=frame, context=context)
                and (reference.relation == "exact_target" and target != 0
                     or reference.relation == "magnitude_lower_bound" and target > 0)):
            threshold = arithmetic.multiply(arithmetic.fraction(accuracy.relative_tolerance), arithmetic.absolute(target))
            return threshold, (), arithmetic.fraction(reference.failure_probability)
    if absolute_fallback is not None:
        return arithmetic.fraction(absolute_fallback), (), zero
    return None, ("relative reference relation and its nonzero domain are not established",), zero


EXACT_READOUT_SAMPLING = Source(name="exact_readout_sampling", version="2",
    domain="observations without drawn outcomes (exact probabilities, Pauli values, amplitudes "
           "or host scalars) and an estimate that, as its Method states, averages or emulates "
           "no random draw",
    reference="nwqlib.evidence.error_model.exact_readout_sampling")


def exact_readout_sampling(plan, observations, *, random_draws):
    """Return the zero ``sampling`` fact when the estimate samples nothing, or ``()``.

    The ``sampling`` source is the statistical error of an estimate that
    averages or emulates the outcomes of random draws. Measurement outcomes
    drawn by a backend, which counts and provider estimates hold, are such
    draws. So are the draws a Method makes itself: an emulated measurement
    outcome, such as the one-bit datum that classical RWPE draws from an
    exact signal, and a random schedule of settings, such as the RFE powers
    or the SPE frequencies, whose finite average estimates a sum over the
    draw distribution. A draw that only selects an input, such as the
    reference state that Lanczos draws when ``initial_state`` is omitted,
    fixes the state that the Plan records. The estimate averages nothing
    over it, so its effect belongs to the Method's identification or
    projection terms. A backend seed that only selects a shot stream draws
    nothing when the readout is exact.

    With exact readout no observation draws outcomes. Every observation is an
    exact probability, Pauli expectation, amplitude or host scalar, as with
    ``shots=None``, and a population without observations draws none
    either. The observations cannot show the draws a Method makes, so the
    calling Method states them in ``random_draws``. When the observations
    draw no outcomes and ``random_draws`` is false, the estimate has no
    sampling term, and its sampling error is zero by definition. The fact
    states that zero as a proved relation, witnessed by the observation
    population, with failure probability zero. Otherwise the function
    returns ``()`` and the Method's own ``sampling`` term applies, which
    stays unknown unless the Method bounds it. Floating-point and native
    simulation error are other sources and stay with their own terms.

    Args:
        plan: Plan whose output frame states the fact.
        observations: Observation population the estimate was computed from.
        random_draws: Whether the estimate averages or emulates the outcomes
            of random draws that the Method makes itself, such as emulated
            measurement outcomes or a random schedule of settings.
    """
    if random_draws or any(chunk.returned_shots is not None
                           or chunk.observation.kind in {"counts", "estimated_observable"}
                           for chunk in observations.chunks):
        return ()
    frame = plan.output.frame(plan.problem)
    evidence = Evidence(kind="proved_relation", source=EXACT_READOUT_SAMPLING, status="witnessed",
        artifact="exact readout and no random draw by the Method",
        witnessed_scope=frame.scope, subject_id=observations.content_id)
    return (FramedFact(frame=frame, bindings=(), failure_probability=0., fact=Fact(quantity="sampling",
        unit=frame.unit, scope=frame.scope, availability="concrete", value=Rational(numerator=0, denominator=1),
        evidence=evidence)),)


def result_context(result):
    """Describe the actual attached experiment and contributing data, without replay."""
    plan, data = result.plan, result.data
    result.validate_plan(plan)
    if data.trace.plan_id != plan.content_id or data.observations.content_id != result.observation_id:
        raise ValueError("result data belongs to another Plan or observation population")
    chunks = {chunk.content_id: chunk for chunk in data.observations.chunks}
    if not set(result.contribution_ids) <= chunks.keys():
        raise ValueError("result contributions are missing from its actual acquired data")
    if result.construction_id == plan._construction_id:
        bindings = plan.construction.program.bindings
    else:
        contributed = [chunks[key] for key in result.contribution_ids]
        if not contributed:
            raise ValueError("a nonbase result requires its actual selected-point observations")
        first = contributed[0]
        point = plan.resolve(first.experiment, bindings=first.bindings)
        if (point.selected_construction(plan).content_id != result.construction_id
                or any(chunk.bindings != point.bindings for chunk in contributed)):
            raise ValueError("result construction differs from its actual selected observation point")
        bindings = point.bindings
    return AssessmentContext(problem_id=plan.problem.content_id, base_construction_id=plan._construction_id,
        selected_construction_id=result.construction_id, plan_id=plan.content_id,
        observation_id=result.observation_id, contribution_ids=result.contribution_ids,
        result_id=result.content_id, bindings=bindings, admitted=True)


def assess_result(result, *, accuracy=None, absolute_tolerance=None, relative_tolerance=None,
                  confidence=.95, component="total", facts=(), reference=None,
                  absolute_fallback=None, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Assess a new explicit criterion on an existing Result; acquire nothing."""
    if accuracy is None:
        accuracy = Accuracy(absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
                            confidence=confidence, component=component)
    elif (absolute_tolerance is not None or relative_tolerance is not None or confidence != .95 or component != "total"):
        raise ValueError("supply either accuracy or its individual criterion fields")
    plan = result.plan
    if plan.error_model is None:
        raise ValueError("this method has no output-error model; inspect its existing diagnostics directly")
    expected = ErrorFrame.from_output(plan.problem, plan.output)
    if (plan.error_model.output_id != plan.output.content_id or not plan.error_model.frame.compatible(expected)):
        raise ValueError("Plan error model does not describe its selected scientific output")
    by_name = {fact.fact.quantity: fact for fact in result.facts}
    if len({fact.fact.quantity for fact in facts}) != len(facts):
        raise ValueError("supplied error facts require distinct source names")
    by_name.update((fact.fact.quantity, fact) for fact in facts)
    return plan.error_model.assess(accuracy, context=result_context(result), facts=tuple(by_name.values()),
        reference=reference, absolute_fallback=absolute_fallback, max_integer_bits=max_integer_bits)


class CheckDomain(Record):
    """Mathematical range of a check value, in the check's unit.

    `CheckSpec.domain` holds it. A value outside the range is rejected before
    any status is decided. The fields below are read-only.

    Attributes:
        lower: Inclusive lower end, or `None` for no lower end.
        upper: Inclusive upper end, or `None` for no upper end.
        integer: Whether the value must be a whole number, for example a
            zero-one flag.
        roundoff_tolerance: Absolute window at each end, in the check's
            unit. A value outside the range by at most this amount is
            compared at the end, and its raw value stays in the fact. Zero
            means exact comparison. The code that sets a nonzero window must
            justify its scale, because a number alone does not bound
            rounding error. The built-in checks of nonnegative errors use
            zero.

    Raises:
        ValueError: If `lower` exceeds `upper`, or a nonzero window is set on
            an integer domain or a domain without ends.
    """

    lower: Real | None = None
    upper: Real | None = None
    integer: StrictBool = False
    roundoff_tolerance: Nonnegative = 0.0

    @model_validator(mode="after")
    def _range(self):
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("check domain endpoints must be ordered")
        if self.roundoff_tolerance and (self.integer or self.lower is self.upper is None):
            raise ValueError("roundoff belongs to a bounded continuous check domain")
        return self


class CheckSpec(Record):
    """One scalar check: the quantity, its threshold and domain, and what running it costs.

    A verification options record produces one per selected criterion, and
    `Result.verify` returns one fact per check. Each verification fact
    records the content hash of its check, so any revision of the check,
    including only a new threshold, needs a new verification. The fields
    below are read-only.

    Attributes:
        name: Check name, which the answering fact's quantity repeats.
        claim_id: Content hash of the output the check concerns.
        frame: The [`ErrorFrame`][nwqlib.evidence.error_model.ErrorFrame] of
            the checked value.
        domain: The [`CheckDomain`][nwqlib.evidence.error_model.CheckDomain]
            of valid values. The threshold compares a valid value and does
            not define the domain, and the domain does not depend on how the
            metric is named.
        source: Implementation that computes the value.
        options_id: Content hash of the options record that selected the
            check, or `None` for a check of a supplied value.
        threshold: Nonnegative threshold. PASS means value `<=` threshold.
            `None` keeps the check INCONCLUSIVE.
        prerequisites: Open assumptions, which keep the check INCONCLUSIVE.
        access: Data the check reads.
        experiments: Number of new experiments the check runs.
        classical_work: Text form of its classical computation.
        reference_work: Text form of its reference computation.
        data_description: What the check stores.
    """

    schema_version: Literal[2] = 2
    name: Text
    claim_id: ContentID
    frame: ErrorFrame
    domain: CheckDomain
    source: Source
    options_id: ContentID | None = None
    threshold: Nonnegative | None
    prerequisites: tuple[Text, ...]
    access: tuple[Text, ...]
    experiments: Count
    classical_work: Text
    reference_work: Text
    data_description: Text


class CheckAssessment(Record):
    """Outcome of one scalar check: its status and the fact it was decided on.

    `Certificate.checks` holds them, and
    [`assemble_check`][nwqlib.evidence.error_model.assemble_check] returns
    one. PASS or FAIL concerns only this check's threshold, and the evidence
    kind of the fact is unchanged. The fields below are read-only.

    Attributes:
        check_id: Content hash of the [`CheckSpec`][nwqlib.evidence.error_model.CheckSpec].
        artifact_id: Content hash of the Result the check concerns.
        status: `"PASS"` or `"FAIL"` compares the value with the threshold.
            `"NOT_RUN"` means no fact was supplied. `"INCONCLUSIVE"` means
            the value, the threshold, the assessed point or support bound to
            this Result is missing, or an assumption is open. `"NOT_APPLICABLE"`
            means the fact states that the check does not apply.
        fact: The supplied fact, with its raw value and evidence kind, or
            `None`.
        reason: Explanation of the status, including any roundoff-window
            adjustment.
    """

    check_id: ContentID
    artifact_id: ContentID
    status: CheckStatus
    fact: FramedFact | None
    reason: Text


def assemble_check(check: CheckSpec, *, artifact_id: str, fact: FramedFact | None = None, max_integer_bits=DEFAULT_MAX_INTEGER_BITS) -> CheckAssessment:
    """Decide one check from a supplied fact, outside a Certificate.

    The status is decided as in
    [`Certificate.with_verification`][nwqlib.evidence.error_model.Certificate.with_verification],
    without a Result context. A fact restricted to a parameter point
    therefore stays INCONCLUSIVE here. A fact from a verification answers
    only the unmodified check that it records, so a check revised
    afterwards, for example with another threshold, is rejected. Run
    `Result.verify` again with new options instead. A value outside the
    check's domain raises before any status is decided.

    Args:
        check (CheckSpec): The check.
        artifact_id (str): Content hash of the Result the check concerns.
        fact (FramedFact | None): The value, or `None`, which gives
            `NOT_RUN`.
        max_integer_bits (int): Bit limit of the exact arithmetic. Default
            4096.

    Returns:
        outcome (CheckAssessment): The status and reason.

    Raises:
        ValueError: If the fact names another quantity or frame, was
            produced by other options or for another check, or lies outside
            the check's domain.
    """
    return _assemble_check(check, artifact_id=artifact_id, fact=fact, context=None, max_integer_bits=max_integer_bits)


def _admit_check_value(check, value, arithmetic):
    """Apply the selected quantity's domain before evidence/status decisions."""
    if value is None:
        return value, ""
    domain = check.domain
    if domain.integer and value.denominator != 1:
        raise ValueError("check value must be an integer in its mathematical domain")
    tolerance = arithmetic.fraction(domain.roundoff_tolerance)
    adjusted = ""
    for endpoint, below in ((domain.lower, True), (domain.upper, False)):
        if endpoint is None:
            continue
        bound = arithmetic.fraction(endpoint)
        violates = not arithmetic.le(bound, value) if below else not arithmetic.le(value, bound)
        if violates:
            distance = arithmetic.subtract(bound, value) if below else arithmetic.subtract(value, bound)
            if not arithmetic.le(distance, tolerance):
                description = "nonnegative" if below and bound == 0 else "within its mathematical domain"
                raise ValueError(f"{check.frame.metric} must be {description}; supplied value violates the selected endpoint")
            value = bound
            comparison = "compared as zero" if bound == 0 else "compared at the selected endpoint"
            adjusted = (f"; raw value {comparison} (domain endpoint {endpoint}) under explicit absolute "
                        f"roundoff tolerance {domain.roundoff_tolerance} in {check.frame.unit.symbol}; raw fact stored")
    return value, adjusted


def _assemble_check(check, *, artifact_id, fact, context=None, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Evaluate one check: frame and identity binding, then domain admission, then status.

    A fact produced for another options record or CheckSpec raises before
    its value is admitted. Domain admission then rejects a
    definition-invalid value before any status, so an impossible value
    raises instead of becoming INCONCLUSIVE or FAIL.
    """
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    status, reason = "NOT_RUN", "no check evidence supplied"
    if context is not None and context.result_id != artifact_id:
        raise ValueError("check context belongs to another actual result")
    if fact is not None:
        if fact.fact.quantity != check.name:
            raise ValueError("check fact quantity mismatch")
        matches = applicable(fact, frame=check.frame, context=context)
        evidence = fact.fact.evidence
        # A receipt-witnessed value answers only the complete options record that
        # produced it. Any other selection, including another threshold, needs a
        # new explicit verification. Attaching a fact never recomputes it.
        if (evidence is not None and evidence.artifact_kind == "verification_receipt"
                and (evidence.options_id != check.options_id or evidence.source != check.source)):
            raise ValueError("verification fact options identity does not match the selected check. Attach it "
                             "with the options that produced it, or call verify again with the new options")
        if (evidence is not None and evidence.artifact_kind == "verification_receipt"
                and evidence.check_id != check.content_id):
            raise ValueError("verification fact answers another check. A check revised after verification, for "
                             "example with another threshold, needs verify again with options that select it")
        value, adjustment = _admit_check_value(check, scalar(fact, arithmetic), arithmetic)
        if fact.fact.availability == "not_applicable":
            status, reason = "NOT_APPLICABLE", fact.fact.reason
        else:
            if not matches:
                status, reason = "INCONCLUSIVE", "point-restricted evidence needs an admitted context"
            elif (value is None or check.threshold is None or check.prerequisites or fact.fact.assumptions
                    or not receipt_supported(fact.fact.evidence, scope=check.frame.scope,
                        subjects=context.subjects if context else (artifact_id,),
                        kinds={"proved_relation", "certified_bound", "numerical_estimate", "empirical_prediction",
                               "user_assertion", "external_specification", "observed"})):
                status, reason = "INCONCLUSIVE", "check value or applicable artifact-bound premises unavailable"
            else:
                threshold = arithmetic.fraction(check.threshold)
                status = "PASS" if arithmetic.le(value, threshold) else "FAIL"
                reason = "supplied scalar check criterion only; evidence kind is unchanged"
            reason += adjustment
    fields = dict(check_id=check.content_id, artifact_id=artifact_id, status=status, fact=fact, reason=reason)
    return CheckAssessment(**fields)


class Certificate(Record):
    """An accuracy assessment of one Result together with its completed verification checks.

    Build it with keyword arguments from an assessment, for example
    `Certificate(plan_id=result.plan_id, result_id=result.content_id,
    assessment=result.assess(absolute_tolerance=1e-6), checks=())`, then
    attach the `facts` that `Result.verify` returned with `with_verification`
    (see [Keep an assessment and its checks
    together](evidence.md#keep-an-assessment-and-its-checks-together)). All four fields
    are required. Checks and the assessment stay separate, so a check that
    passes its own threshold does not change an INCONCLUSIVE assessment.

    Attributes:
        plan_id: Required. Content hash of the Result's Plan.
        result_id: Required. Content hash of the Result.
        assessment: Required. The [`ClaimAssessment`][nwqlib.evidence.error_model.ClaimAssessment]
            of that Result.
        checks: Required. Completed checks as
            [`CheckAssessment`][nwqlib.evidence.error_model.CheckAssessment]
            records, `()` to start.

    Raises:
        ValueError: If the assessment belongs to another Plan or Result, or a
            check to another Result.
    """

    plan_id: ContentID
    result_id: ContentID
    assessment: ClaimAssessment
    checks: tuple[CheckAssessment, ...]

    def with_verification(self, result, *, options, evidence, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
        """Return a new Certificate with the `facts` of a completed verification attached as checks.

        Nothing is verified or assessed again. Each fact from a verification
        must come from these same `options`, compared by their complete
        content hash, and answer the check it records. The hash covers every
        field and the revision history (`parent_id`), so keep the options
        object you passed to `Result.verify`, or its saved JSON. Changing any
        option, even only a threshold, needs a new `Result.verify` call with
        the new options, because the old `facts` were never produced for it.
        An attached check replaces an earlier check of the same `CheckSpec`.

        Args:
            result (Result): The certified Result, with its Plan attached.
            options (object): The options record passed to `Result.verify`.
            evidence (tuple[FramedFact, ...]): The `facts` that
                `Result.verify` returned.
            max_integer_bits (int): Bit limit of the exact arithmetic.
                Default 4096.

        Returns:
            certificate (Certificate): A revision with the checks attached.

        Raises:
            ValueError: If `result` is not the certified Result, a fact names
                no check of `options` or repeats one, or a fact was produced
                by other options or for another check.
        """
        plan = result.plan
        if self.plan_id != plan.content_id or self.result_id != result.content_id:
            raise ValueError("verification attachment requires the exact certified Plan/result")
        result.validate_plan(plan)
        specs = options.verification_checks(result)
        names = {check.name for check in specs}
        supplied = {fact.fact.quantity: fact for fact in evidence}
        if len(supplied) != len(evidence) or not supplied.keys() <= names:
            raise ValueError("verification facts must name distinct selected criteria")
        replacements = {check.content_id: _assemble_check(check, artifact_id=result.content_id,
            fact=supplied.get(check.name), context=self.assessment.context, max_integer_bits=max_integer_bits) for check in specs}
        checks = tuple(replacements.pop(check.check_id, check) for check in self.checks) + tuple(replacements.values())
        return self.revise(checks=checks)

    @model_validator(mode="after")
    def _identity(self):
        if self.assessment.context.plan_id != self.plan_id or self.assessment.context.result_id != self.result_id:
            raise ValueError("certificate must keep its actual Plan/result assessment")
        if any(c.artifact_id != self.result_id for c in self.checks):
            raise ValueError("certificate check belongs to another result")
        return self
