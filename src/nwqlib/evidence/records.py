"""Fact availability and declared evidence are independent scientific dimensions."""

from typing import Literal

from pydantic import StrictBool, model_validator

from nwqlib.core.records import ContentID, Record, Scalar, Scope, Source, Stage, Symbol, Text, Unit

EvidenceKind = Literal[
    "proved_relation", "certified_bound", "numerical_estimate", "empirical_prediction",
    "user_assertion", "external_specification", "observed",
]


class WorkProvenance(Record):
    """Work done to obtain a value: when it ran, what it computed and where its output is.

    `Evidence.work` holds these records. The fields below are read-only.

    Attributes:
        stage: Workflow stage in which the work ran: `"planning"`,
            `"preparation"`, `"execution"`, `"analysis"` or
            `"verification"`.
        description: What was computed and its size, for example entries
            scanned and scalar operations.
        artifact: Identifier of the record that holds the output of the work.
    """

    stage: Stage
    description: Text
    artifact: Text


class Evidence(Record):
    """The basis of a value (proof, bound, estimate, assertion or observation) and the record that produced it.

    `Fact.evidence` holds it. `kind` is a declaration, and validating the
    record never turns it into a verified result. A value is witnessed only
    when `status` is `"witnessed"` and the evidence names the record of the
    computation that produced the value (`artifact`), its scope and its
    subject. Without that, even a declared `proved_relation` stays an
    unverified declaration. A user assertion can be witnessed, meaning it
    was received, without becoming proof of the asserted value. Evidence
    from a verification also names the options record that produced the
    value, and the value answers only those options. When it answers a
    check, `check_id` names that exact `CheckSpec`, so a check revised
    afterwards, for example with another threshold, is not answered by it.
    A verification value that answers no check, such as an output-error
    component, has no `check_id`.

    Build it with keyword arguments to state the basis of a value you
    supply, for example `Evidence(kind="user_assertion", source=source)`.
    `kind` and `source` are required. The other fields describe a witnessed
    value and default to `None` or empty.

    Attributes:
        kind: Required. Declared basis of the value: `"proved_relation"` or
            `"certified_bound"` (the only kinds that can support an
            accuracy PASS), `"numerical_estimate"`,
            `"empirical_prediction"`, `"user_assertion"`,
            `"external_specification"` or `"observed"`.
        source: Required. Versioned `Source` that declares the basis.
        work: Default `()`. [`WorkProvenance`][nwqlib.evidence.records.WorkProvenance]
            records of the work done to obtain the value.
        status: Default `"declared"`, a declaration alone. `"witnessed"`
            means the evidence names the record of the computation that
            produced the value.
        artifact: Default `None`. Identifier of that record, set exactly
            when witnessed.
        artifact_kind: Default `None`. `"verification_receipt"` when that
            record is a verification record.
        witnessed_scope: Default `None`. Scope that the record covers, set
            exactly when witnessed.
        subject_id: Default `None`. Content hash of the witnessed subject,
            such as a Result or an operator input, set exactly when
            witnessed.
        options_id: Default `None`. Content hash of the verification options
            that produced the value, set exactly for a verification record.
        check_id: Default `None`. Content hash of the `CheckSpec` the value
            answers.
        schema_version: Format version of the record, 4.

    Raises:
        ValueError: If the witnessed fields are not all set exactly when
            `status` is `"witnessed"`, or `options_id` or `check_id` is set
            without a verification record.
    """

    schema_version: Literal[4] = 4
    kind: EvidenceKind
    source: Source
    work: tuple[WorkProvenance, ...] = ()
    status: Literal["declared", "witnessed"] = "declared"
    artifact: Text | None = None
    artifact_kind: Literal["verification_receipt"] | None = None
    witnessed_scope: Scope | None = None
    subject_id: ContentID | None = None
    options_id: ContentID | None = None
    check_id: ContentID | None = None

    @model_validator(mode="after")
    def _receipt(self):
        receipt = (self.artifact, self.witnessed_scope, self.subject_id)
        if self.status == "witnessed" and any(item is None for item in receipt):
            raise ValueError("witnessed evidence requires artifact, scope and subject identity")
        if self.status == "declared" and any(item is not None for item in receipt):
            raise ValueError("receipt fields belong to witnessed evidence")
        if self.artifact_kind is not None and self.status != "witnessed":
            raise ValueError("typed receipt reference belongs to witnessed evidence")
        if (self.artifact_kind == "verification_receipt") != (self.options_id is not None):
            raise ValueError("verification receipt evidence requires exactly the options identity that produced it")
        if self.check_id is not None and self.artifact_kind != "verification_receipt":
            raise ValueError("a check identity belongs to verification receipt evidence")
        return self


class Fact(Record):
    """One value with its availability, unit, scope and evidence.

    A concrete fact has a value and its [`Evidence`][nwqlib.evidence.records.Evidence].
    A symbolic fact has only a symbol that names an expression. An unknown
    or not-applicable fact has a reason and no value, so missing information
    never reads as zero. Build one with keyword arguments to supply a value.
    `quantity`, `unit`, `scope` and `availability` are required.

    Attributes:
        quantity: Required. Name of the quantity.
        unit: Required. Unit of the value.
        scope: Required. Scope in which the value holds.
        availability: Required. `"concrete"`, `"symbolic"`, `"unknown"` or
            `"not_applicable"`.
        value: Default `None`. An exact `Rational`, a `Float64`, a
            `Complex128` or, for a predicate, a boolean. Set exactly when
            concrete.
        symbol: Default `None`. Symbol naming an expression, set exactly
            when symbolic.
        reason: Default `None`. Why the value is unknown or not applicable,
            set exactly then.
        evidence: Default `None`. Required for a concrete value, absent when
            the value is unknown or not applicable.
        assumptions: Default `()`. Open assumptions the value depends on.

    Raises:
        ValueError: If the fields set do not match `availability` as above.
    """

    quantity: Text
    unit: Unit
    scope: Scope
    availability: Literal["concrete", "symbolic", "unknown", "not_applicable"]
    value: Scalar | StrictBool | None = None
    symbol: Symbol | None = None
    reason: Text | None = None
    evidence: Evidence | None = None
    assumptions: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def _availability(self):
        if self.availability == "concrete":
            if self.value is None or self.symbol is not None or self.reason is not None:
                raise ValueError("concrete fact requires a value, without symbol or missing reason")
            if self.evidence is None:
                raise ValueError("concrete fact requires its declared evidence basis")
        elif self.availability == "symbolic":
            if self.symbol is None or self.value is not None or self.reason is not None:
                raise ValueError("symbolic fact requires only an inert symbol")
        elif self.reason is None or any(x is not None for x in (self.value, self.symbol, self.evidence)):
            raise ValueError("unavailable facts require their own reason and no value or evidence basis")
        return self
