"""Checks of saved projected quantities, and the verification-record step shared by most checks.

`_publish` and `witness_check_facts` bind check values to a
VerificationReceipt, the verification record, for the projected,
energy-shift and number-sector checks and for the QLS, QPE and QHD
verifications. Nothing here takes measurements or solves an eigenproblem.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Literal

from pydantic import model_validator

from nwqlib.core.records import Complex128, Nonnegative, Rational, Record, Source, Text, Unit
from .error_model import CheckDomain, CheckSpec, ErrorFrame, FramedFact, _assemble_check
from .records import Evidence, Fact


@dataclass(frozen=True)
class ProjectedDiagnostics:
    """The saved diagnostics of a projected eigenvalue solve: overlap matrix, its spectrum and two error values.

    `result.projected_diagnostics()` returns it for a Lanczos, FixedGCIM or
    ADAPT Result, and the checks of
    [`ProjectedVerificationOptions`][nwqlib.evidence.verification.ProjectedVerificationOptions]
    read it. It holds values the solve already produced, in the Method's
    own coordinates, so getting it runs no solve, reconstruction or
    decomposition. A value the Result did not save is `None`. The fields
    below are read-only.

    Attributes:
        overlap: The overlap (Gram) matrix S of the trial basis, as rows:
            the raw Chebyshev Gram matrix of Lanczos, with float entries, or
            the matrix that FixedGCIM and ADAPT assembled from their
            readouts, completed from its upper triangle by conjugation, with
            `Complex128` entries. `None`
            when unavailable.
        spectrum: The eigenvalues of S before the overlap cutoff, negative
            ones included. `None` when unavailable.
        normalization: `abs(c† S c - 1)` for the coefficient vector c of the
            reported, lowest Ritz pair, or `None`.
        backward_error: The dimensionless backward error
            `||A c - x S c|| / ((||A||_F + |x| ||S||_F) ||c||)` of the lowest
            Ritz pair (x, c), or `None`. Lanczos states it on the normalized
            pencil (K, S) of its solve, with `K = (H - center S) / alpha` and
            `x = (E - center) / alpha`. FixedGCIM and ADAPT state it on the
            physical pencil (H, S) in the Problem's energy unit, with `x = E`.
            Equal thresholds therefore mean different things for Lanczos and
            for FixedGCIM and ADAPT ([verification
            guide](../verification.md#projected-quantities)).
    """

    overlap: tuple | None
    spectrum: tuple | None
    normalization: float | None
    backward_error: float | None


ProjectedCriterion = Literal["overlap_normalization", "gram_psd_deficit", "gram_hermiticity", "projected_backward_error"]
_PROJECTED_SOURCE = Source(name="stored_projected_checks", version="1",
    domain="scalar criteria on stored projected coordinates; no full-state or ground claim",
    reference="nwqlib.evidence.verification.verify_projected")


class ProjectedVerificationOptions(Record):
    """Options of the projected checks of a Lanczos, FixedGCIM or ADAPT Result.

    Build it with keyword arguments, for example
    `ProjectedVerificationOptions(name="projected",
    comparisons=("gram_psd_deficit",), tolerance=1e-10)`, and pass it to
    `result.verify(checks=...)` or
    [`verify_projected`][nwqlib.evidence.verification.verify_projected].
    `name`, `comparisons` and `tolerance` are required. Each criterion is a
    dimensionless value on `[0, inf)`, compared with `tolerance` as an
    acceptance threshold. Nothing is repaired, and the values are read from
    what the Result already holds (`result.projected_diagnostics()`), so no
    solve runs again:

    - `overlap_normalization`: the saved absolute error of the coefficient
      vector's normalization in the overlap (Gram) matrix S.
    - `gram_psd_deficit`: `max(0, -min(spectrum))` of the raw overlap
      spectrum, before positive-subspace filtering.
    - `gram_hermiticity`: the largest absolute real or imaginary component of
      `S - S†` of the saved overlap matrix.
    - `projected_backward_error`: the saved backward error of the
      projected generalized eigenproblem, on the normalized pencil (K, S)
      for Lanczos and on the physical pencil (H, S) for FixedGCIM and ADAPT.
      Equal thresholds mean different things for the two, so compare values
      only within one family (formula in the [verification
      guide](../verification.md#projected-quantities)).

    The criteria concern the projected problem only. They give no
    full-state residual, ground-state identification or total physical-error
    bound.

    Attributes:
        name: Required. Prefix of the check names, which are
            `name + "." + criterion`.
        comparisons: Required. Nonempty tuple of distinct criteria from the
            list above.
        tolerance: Required. Nonnegative threshold shared by the criteria.

    Raises:
        ValueError: If `comparisons` is empty or repeats a criterion.
    """

    name: Text
    comparisons: tuple[ProjectedCriterion, ...]
    tolerance: Nonnegative
    source: ClassVar[Source] = _PROJECTED_SOURCE

    @model_validator(mode="after")
    def _selection(self):
        if not self.comparisons or len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("select distinct nonempty projected criteria")
        return self

    def verification_checks(self, result):
        """Return one CheckSpec per selected criterion, in the projected frame of ``result``.

        Each check is a dimensionless scalar on the domain [0, inf) compared
        with ``tolerance``. ``overlap_normalization`` and
        ``projected_backward_error`` read saved scalars,
        ``gram_psd_deficit`` is ``max(0, -min(spectrum))`` of the raw overlap
        spectrum before positive-subspace filtering, and ``gram_hermiticity``
        is the largest absolute real or imaginary component of ``S - S†``.
        The Result must supply ``projected_diagnostics``.
        """
        plan = result.plan
        result.validate_plan(plan)
        if not callable(getattr(result, "projected_diagnostics", None)):
            raise TypeError("result has no stored projected diagnostic consumer")
        definitions = {
            "overlap_normalization": ("overlap_normalization_error", "stored absolute S-normalization defect", "one stored scalar"),
            "gram_psd_deficit": ("gram_psd_deficit", "max(0,-minimum raw overlap eigenvalue), before positive-subspace filtering", "linear spectrum scan"),
            "gram_hermiticity": ("maximum_component_hermiticity_defect", "maximum absolute real/imaginary component of stored S-S†", "quadratic exact-scalar Gram scan"),
            "projected_backward_error": ("projected_backward_error", "stored backward error of the producer's pencil, normalized (K,S) for Lanczos and physical (H,S) for GCIM; not a full-state residual", "one stored scalar"),
        }
        return tuple(CheckSpec(name=self.name + "." + criterion, claim_id=plan.output.content_id,
            frame=ErrorFrame(quantity=criterion, metric=definitions[criterion][0],
                unit=Unit(symbol="1", dimension="dimensionless"), scope=plan.problem.evidence_scope,
                conditioning=definitions[criterion][1], domain="projected"),
            domain=CheckDomain(lower=0.), source=self.source, options_id=self.content_id,
            threshold=self.tolerance, prerequisites=(), access=("actual result's stored projected coordinates",),
            experiments=0, classical_work=definitions[criterion][2], reference_work="none; no reconstruction or decomposition",
            data_description="criterion scalar and source result identity; raw diagnostics stay in that result") for criterion in self.comparisons)


def _parts(value):
    """Return the exact real and imaginary parts of a stored Complex128 or real entry."""
    if isinstance(value, Complex128):
        return Fraction(value.real), Fraction(value.imag)
    return Fraction(value), Fraction(0)


def _hermiticity(matrix):
    """Return max over i, j of the absolute real and imaginary parts of (S - S†)_ij, exactly.

    ``(S - S†)_ij = S_ij - conj(S_ji)`` has real part ``Re S_ij - Re S_ji``
    and imaginary part ``Im S_ij + Im S_ji``. Because
    ``(S - S†)_ji = -conj((S - S†)_ij)``, the scan visits only ``j >= i``,
    including the diagonal, whose defect is ``2 |Im S_ii|``. The value is a
    derived certificate quantity compared with a threshold, and a rounded
    subtraction could fall below it, so the components are exact
    ``Fraction`` values of the stored binary64 entries. A missing matrix
    returns None.
    """
    if matrix is None:
        return None
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("stored Gram matrix must be square")
    defect = Fraction(0)
    for i in range(size):
        for j in range(i, size):
            ar, ai = _parts(matrix[i][j])
            br, bi = _parts(matrix[j][i])
            defect = max(defect, abs(ar - br), abs(ai + bi))
    return defect


def _publish(result, options, checks, metrics):
    """Bind each computed scalar check value to one new verification record, keeping the raw facts.

    Each raw fact is first evaluated against its check, so that a value
    outside the quantity's definition raises before the record exists.
    Every returned fact with a value is then witnessed by the record and
    holds the content hashes of the options and of its CheckSpec, which
    Certificate.with_verification and assemble_check compare. A fact without
    a value keeps no evidence and attaches as INCONCLUSIVE.
    """
    from nwqlib.execution import KernelApplication, VerificationReceipt, verification_invocation

    raw = []
    for check, value in zip(checks, metrics, strict=True):
        fields = dict(quantity=check.name, unit=check.frame.unit, scope=check.frame.scope)
        if value is None:
            fields.update(availability="unknown", reason="selected stored diagnostic unavailable; no computation was replayed")
        else:
            # A binary64 value or an exact Fraction, stored as its exact ratio.
            numerator, denominator = value.as_integer_ratio()
            fields.update(availability="concrete", value=Rational(numerator=numerator, denominator=denominator),
                          evidence=Evidence(kind="numerical_estimate", source=options.source))
        fact = FramedFact(frame=check.frame, bindings=(), fact=Fact(**fields))
        _assemble_check(check, artifact_id=result.content_id, fact=fact)
        raw.append(fact)
    application = KernelApplication(name=options.name, implementation=options.source, arguments=(), facts=tuple(raw))
    receipt = VerificationReceipt(plan_id=result.plan_id, result_id=result.content_id,
        invocation_id=verification_invocation(), construction_id=result.construction_id, artifact_ids=(),
        reference=options.source, options_id=options.content_id, applications=(application,))
    return receipt, witness_check_facts(receipt, result, checks)


def witness_check_facts(receipt, result, checks):
    """Return the fact that answers each CheckSpec, witnessed by the completed verification record ``receipt``.

    Each returned fact is the raw fact in the record that carries its
    check's name. A fact with a value receives evidence that cites the
    record, the Result, the options that produced it and the CheckSpec it
    answers. Certificate.with_verification and assemble_check compare these
    fields before they give PASS or FAIL. A raw fact inside the record cannot
    carry them, because it exists before the record's content hash does. A
    fact without a value keeps no evidence and attaches as INCONCLUSIVE.

    Returns:
        One FramedFact per CheckSpec, in the order of ``checks``.
    """
    raw = {fact.fact.quantity: fact for application in receipt.applications for fact in application.facts}
    witnessed = []
    for check in checks:
        fact = raw[check.name]
        if fact.fact.evidence is None:
            witnessed.append(fact)
            continue
        evidence = Evidence(kind=fact.fact.evidence.kind, source=fact.fact.evidence.source, status="witnessed",
            artifact_kind="verification_receipt", artifact=receipt.content_id,
            subject_id=result.content_id, witnessed_scope=fact.frame.scope, options_id=receipt.options_id,
            check_id=check.content_id)
        fields = {name: getattr(fact.fact, name) for name in Fact.model_fields}
        witnessed.append(FramedFact(frame=fact.frame, bindings=fact.bindings, fact=Fact(**{**fields, "evidence": evidence})))
    return tuple(witnessed)


def verify_projected(result, *, options: ProjectedVerificationOptions):
    """Run the projected checks on a Result's saved diagnostics, without solving again.

    `result.verify(checks=options)` calls this for Lanczos, FixedGCIM and
    ADAPT. The Result must supply `projected_diagnostics()`. The criteria
    concern the projected problem only, not a full-state residual or
    ground-state identification.

    Args:
        result (Result): A Lanczos, FixedGCIM or ADAPT Result with its Plan.
        options (ProjectedVerificationOptions): The selected criteria.

    Returns:
        verification (tuple): `(receipt, facts)`: the `VerificationReceipt`
            of this computation, and one
            [`FramedFact`][nwqlib.evidence.error_model.FramedFact] per
            criterion that cites it. A criterion whose saved value is
            missing gives an unknown fact.

    Raises:
        TypeError: If `options` is not a `ProjectedVerificationOptions`, or
            the Result supplies no projected diagnostics.
    """
    if type(options) is not ProjectedVerificationOptions:
        raise TypeError("projected verification requires concrete selected options")
    checks = options.verification_checks(result)
    diagnostics = result.projected_diagnostics()
    if type(diagnostics) is not ProjectedDiagnostics:
        raise TypeError("producer must return actual ProjectedDiagnostics")
    values = []
    for criterion in options.comparisons:
        if criterion == "overlap_normalization":
            value = diagnostics.normalization
        elif criterion == "projected_backward_error":
            value = diagnostics.backward_error
        elif criterion == "gram_hermiticity":
            value = _hermiticity(diagnostics.overlap)
        else:
            # The minimum and its negation are exact in binary64.
            minimum = min(diagnostics.spectrum) if diagnostics.spectrum else None
            value = None if minimum is None else -minimum if minimum < 0 else 0.0
        values.append(value)
    return _publish(result, options, checks, values)
