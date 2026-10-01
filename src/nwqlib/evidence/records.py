"""Fact availability and declared evidence are independent scientific dimensions."""

from typing import Literal

from pydantic import StrictBool, model_validator

from nwqlib.core.records import ContentID, Record, Scalar, Scope, Source, Stage, Symbol, Text, Unit

EvidenceKind = Literal[
    "proved_relation", "certified_bound", "numerical_estimate", "empirical_prediction",
    "user_assertion", "external_specification", "observed",
]


class WorkProvenance(Record):
    """Work actually used to acquire evidence. This record launches no work.

    Attributes:
        stage: Workflow stage in which the work ran.
        description: What was computed and its derived size, for example entries scanned and scalar operations.
        artifact: Identity of the receipt or record that holds the work's output.
    """

    stage: Stage
    description: Text
    artifact: Text


class Evidence(Record):
    """Declared basis, never promoted by validation into scientific certification.

    A witnessed claim requires a specific artifact and scope receipt. Without
    that receipt, even a declared proved_relation remains an unverified source
    declaration. A user assertion can be recorded as witnessed (the assertion
    was received) without becoming proof of the asserted quantity. A
    verification receipt witness also names the complete options record that
    produced the value, and the value answers only that selection. When the
    value answers a selected check, check_id names that exact CheckSpec, so a
    criterion revised afterwards, for example with another threshold, is not
    answered by it. A receipt value that answers no check, such as an
    output-error component, has no check_id.

    Attributes:
        kind: Declared basis of the value, from ``proved_relation`` and ``certified_bound`` (the only kinds that can support an accuracy PASS) to ``numerical_estimate``, ``empirical_prediction``, ``user_assertion``, ``external_specification`` and ``observed``.
        source: Versioned Source that declares the basis.
        work: Work actually used to acquire the value.
        status: ``declared`` for a source declaration alone, ``witnessed`` when a receipt artifact records it.
        artifact: Identity of the witnessing artifact, required exactly when witnessed.
        artifact_kind: ``verification_receipt`` when the artifact is a VerificationReceipt, otherwise None.
        witnessed_scope: Scope that the artifact witnesses, required exactly when witnessed.
        subject_id: Identity of the witnessed subject, such as a Result or an operator input, required exactly when witnessed.
        options_id: Identity of the verification options that produced the value, required exactly for a verification receipt.
        check_id: Identity of the CheckSpec the value answers, or None.
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
    """One quantity with explicit availability, units, scope and evidence basis.

    A concrete fact carries a value and its declared evidence. A symbolic
    fact carries only an inert symbol. An unknown or not_applicable fact
    carries its reason and no value, so missing evidence never reads as zero.

    Attributes:
        quantity: Name of the quantity the fact states.
        unit: Unit of the value.
        scope: Evidence scope in which the value holds.
        availability: ``concrete``, ``symbolic``, ``unknown`` or ``not_applicable``.
        value: Exact or binary64 scalar, or a boolean for a predicate, present exactly when concrete.
        symbol: Inert symbol naming an expression, present exactly when symbolic.
        reason: Why the value is unknown or not applicable, present exactly then.
        evidence: Declared evidence basis, required for a concrete value and absent when unavailable.
        assumptions: Open premises on which the value depends.
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
