"""Bounded scalar covariance with explicit framed values and joint-data identity."""

from pydantic import model_validator

from nwqlib.core.records import ContentID, Float64, Rational, Real, Record, Source
from .error_model import (
    ErrorFrame, FramedFact, _rational, applicable, predicate_frame,
    receipt_supported, scalar, union_bindings,
)
from .records import Evidence, Fact
from ._work import DEFAULT_MAX_INTEGER_BITS, ExactArithmetic
from nwqlib.ir.expressions import Binding


class EstimatorContribution(Record):
    """coefficient multiplies the random variable data_id; variance has squared units.

    Repeated data_id is the same variable, never another independent sample.
    """

    data_id: ContentID
    coefficient: Real
    variance: FramedFact


class CrossCovariance(Record):
    """One supplied unordered pair; absence never means zero covariance."""

    left: ContentID
    right: ContentID
    fact: FramedFact

    @model_validator(mode="after")
    def _distinct(self):
        if self.left == self.right:
            raise ValueError("diagonal variance belongs to the contribution")
        return self


class IndependenceLaw(Record):
    """Joint relation with its own frame, restrictions, predicates and receipt."""

    joint_id: ContentID
    frame: ErrorFrame
    bindings: tuple[Binding, ...]
    evidence: Evidence
    conditions: tuple[FramedFact, ...] = ()

    @model_validator(mode="after")
    def _restrictions(self):
        if len({b.parameter for b in self.bindings}) != len(self.bindings):
            raise ValueError("independence-law restrictions must be unique")
        for condition in self.conditions:
            applicable(condition, frame=predicate_frame(self.frame), context=None)
        return self


class VarianceAssessment(Record):
    """Variance and stored joint-data inputs; value owns the complete output frame.

    This is not a standard deviation, confidence bound or PSD validation.
    """

    joint_id: ContentID
    value: FramedFact
    contributions: tuple[EstimatorContribution, ...]
    covariance: tuple[CrossCovariance, ...]
    independence: IndependenceLaw | None


def linear_variance(*, joint_id: str, frame: ErrorFrame,
                    contributions: tuple[EstimatorContribution, ...],
                    covariance: tuple[CrossCovariance, ...] = (),
                    independence: IndependenceLaw | None = None, max_integer_bits=DEFAULT_MAX_INTEGER_BITS) -> VarianceAssessment:
    """Evaluate Var(sum a_i X_i) in O(contributions + supplied pairs) record space.

    Var(sum a_i X_i)=sum a_i^2 Var(X_i)+2 sum_{i<j} a_i a_j Cov(X_i,X_j), after
    contributions with one data_id are merged into one variable by adding
    their coefficients. This merge accounts for the correlation between two
    estimators that share calibration data. Their derivatives with respect to
    the shared population add before squaring. Every pair of active
    variables needs a supplied covariance or the IndependenceLaw, and a
    missing one leaves the result unknown. A supplied covariance c must
    satisfy the Cauchy-Schwarz inequality c^2<=Var(X)Var(Y).
    Group repeated variables before missing-pair decisions. Exact cancellation
    needs no unused variance. Never invent independence, a dense matrix or PSD
    repair. Exact scalar operations use their finite integer representation guard.
    Its evidence is a proved relation only when every used input is witnessed
    proof. An asserted or externally specified input makes it a user
    assertion, and any other mix makes it a numerical estimate.

    Args:
        joint_id: Identity of the joint data the variables belong to.
        frame: ErrorFrame with the variance metric and the squared output unit.
        contributions: EstimatorContribution per (variable, coefficient) term.
        covariance: Explicit cross-covariances of distinct variables.
        independence: Declared IndependenceLaw in place of explicit covariances.
        max_integer_bits: Integer width limit of the exact rational arithmetic.

    Returns:
        VarianceAssessment whose value is concrete with its evidence kind, or
        unknown when a needed variance or covariance is missing.
    """
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    if frame.metric != "variance":
        raise ValueError("linear_variance requires the variance metric and its squared unit")
    if independence is not None and covariance:
        raise ValueError("choose explicit covariance or a declared independence law")
    if independence is not None:
        if not isinstance(independence, IndependenceLaw):
            raise TypeError("independence requires an explicit joint-data law")
        if independence.joint_id != joint_id or not independence.frame.compatible(frame):
            raise ValueError("independence joint identity/frame/conditioning mismatch")
    grouped, variances, variance_values = {}, {}, {}
    zero = arithmetic.fraction(0)
    for item in contributions:
        statement = item.variance
        applicable(statement, frame=frame, context=None)
        raw = statement.fact.value
        if raw is not None and not isinstance(raw, (Rational, Float64)):
            raise ValueError("variance requires a real scalar")
        value = scalar(statement, arithmetic)
        if value is not None and not arithmetic.le(zero, value):
            raise ValueError("supplied variance must be nonnegative")
        if item.data_id in variances and statement != variances[item.data_id]:
            raise ValueError("one random variable cannot carry conflicting variance facts")
        variances[item.data_id] = statement
        variance_values[item.data_id] = value
        coefficient = arithmetic.fraction(item.coefficient)
        grouped[item.data_id] = arithmetic.add(grouped.get(item.data_id, zero), coefficient)
    active = {key: coefficient for key, coefficient in grouped.items() if coefficient}
    total, missing = zero, False
    used = []
    for key, coefficient in active.items():
        statement = variances[key]
        used.append(statement)
        value = variance_values[key]
        if value is None:
            missing = True
        else:
            squared = arithmetic.multiply(coefficient, coefficient)
            total = arithmetic.add(total, arithmetic.multiply(squared, value))
    pairs, relevant_pairs = set(), 0
    for pair in covariance:
        key = tuple(sorted((pair.left, pair.right)))
        if key in pairs or pair.left not in grouped or pair.right not in grouped:
            raise ValueError("covariance pairs must be unique and name supplied random variables")
        pairs.add(key)
        applicable(pair.fact, frame=frame, context=None)
        raw = pair.fact.fact.value
        if raw is not None and not isinstance(raw, (Rational, Float64)):
            raise ValueError("covariance requires a real scalar")
        value = scalar(pair.fact, arithmetic)
        vx, vy = variance_values[pair.left], variance_values[pair.right]
        # Supplied moments must describe a possible pair, even when its
        # coefficient cancels. Missing unused moments remain unnecessary.
        if value is not None and vx is not None and vy is not None:
            covariance_squared = arithmetic.multiply(value, value)
            product_bound = arithmetic.multiply(vx, vy)
            if not arithmetic.le(covariance_squared, product_bound):
                raise ValueError("supplied covariance exceeds the variance product; no joint law")
        if pair.left in active and pair.right in active:
            relevant_pairs += 1
            used.append(pair.fact)
            if value is None:
                missing = True
            else:
                product = arithmetic.multiply(active[pair.left], active[pair.right])
                product = arithmetic.multiply(product, value)
                term = arithmetic.multiply(2, product)
                total = arithmetic.add(total, term)
    needs_independence = independence is not None and len(active) > 1
    law_supported = False
    assumptions = [a for f in used for a in f.fact.assumptions]
    if needs_independence:
        used.extend(independence.conditions)
        for condition in independence.conditions:
            applicable(condition, frame=predicate_frame(frame), context=None)
            assumptions.extend(condition.fact.assumptions)
            if condition.fact.value is not True:
                missing = True
            elif not receipt_supported(condition.fact.evidence, scope=frame.scope, subjects=(joint_id,),
                                       kinds={"proved_relation"}):
                assumptions.append(f"unresolved independence predicate {condition.fact.quantity} ({condition.content_id})")
        law_supported = (independence.evidence.status == "declared" or receipt_supported(
            independence.evidence, scope=frame.scope, subjects=(joint_id,), kinds={independence.evidence.kind}))
    if not law_supported:
        missing |= relevant_pairs != len(active) * (len(active) - 1) // 2
    restrictions = union_bindings(*(f.bindings for f in used),
                                 independence.bindings if needs_independence else ())
    # Conditions on canceled/unused inputs are not premises of the zero identity.
    if missing:
        fields = dict(quantity=frame.quantity, unit=frame.unit, scope=frame.scope,
                      availability="unknown", reason="required variance or cross-covariance is unavailable",
                      assumptions=tuple(dict.fromkeys(assumptions)))
    else:
        if total < 0:
            raise ValueError("supplied joint law gives negative variance")
        bases = [f.fact.evidence for f in used]
        if needs_independence:
            bases.append(independence.evidence)
        # Proposition-specific rule: proof of exact supplied variances plus the
        # exact covariance identity can remain proof. A bound is not an exact
        # covariance value; other numerical bases stay numerical, not certified.
        proved = all(receipt_supported(b, scope=frame.scope, subjects=(joint_id,),
                                       kinds={"proved_relation"}) for b in bases)
        kind = "proved_relation" if proved else "numerical_estimate"
        if any(b is not None and b.kind in {"user_assertion", "external_specification"} for b in bases):
            kind = "user_assertion"
        source_fields = dict(name="linear covariance identity", version="1", domain=frame.conditioning,
            reference="Var(sum a_i X_i)=sum a_i^2 Var(X_i)+2 sum_{i<j} a_i a_j Cov(X_i,X_j)")
        source = Source(**source_fields)
        evidence_fields = dict(kind=kind, source=source)
        if proved:
            evidence_fields.update(status="witnessed", subject_id=joint_id, witnessed_scope=frame.scope,
                                   artifact="exact finite scalar covariance identity on the stored premises")
        evidence = Evidence(**evidence_fields)
        fields = dict(quantity=frame.quantity, unit=frame.unit, scope=frame.scope, availability="concrete",
                      value=_rational(total), evidence=evidence,
                      assumptions=tuple(dict.fromkeys(assumptions)))
    fact = Fact(**fields)
    fields = dict(frame=frame, bindings=restrictions, fact=fact)
    statement = FramedFact(**fields)
    fields = dict(joint_id=joint_id, value=statement, contributions=contributions,
                  covariance=covariance, independence=independence)
    return VarianceAssessment(**fields)
