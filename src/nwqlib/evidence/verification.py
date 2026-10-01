"""Explicit checks of stored scientific quantities; no acquisition or eigensolve."""

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from nwqlib.core.records import Complex128, Nonnegative, Rational, Record, Source, Text, Unit
from ._work import DEFAULT_MAX_INTEGER_BITS, ExactArithmetic
from .error_model import CheckDomain, CheckSpec, ErrorFrame, FramedFact, _assemble_check
from .records import Evidence, Fact


@dataclass(frozen=True)
class ProjectedDiagnostics:
    """Borrowed immutable coordinates already produced by a projected solver.

    overlap uses the producing method's actual Gram convention; spectrum is the
    raw pre-filter spectrum. Missing diagnostics do not launch another solve.

    Attributes:
        overlap: Existing raw projected overlap matrix in its original coordinates.
        spectrum: Existing overlap eigenvalues, preserving negative modes before filtering.
        normalization: Stored generalized-vector overlap-normalization error, or None if unavailable.
        backward_error: Stored projected generalized-eigenpair backward error, or None if unavailable.
            Lanczos computes it on its normalized pencil (K, S) and GCIM on
            the physical pencil (H, S), so equal thresholds mean different
            things for the two producers.
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
    """Selected dimensionless criteria; tolerance is acceptance, never repair.

    Attributes:
        name: Prefix of the check names.
        comparisons: Distinct selected criteria among ``overlap_normalization``, ``gram_psd_deficit``, ``gram_hermiticity`` and ``projected_backward_error``.
        tolerance: Nonnegative threshold shared by the selected criteria.
        source: Implementation source of the stored projected checks.
    """

    name: Text
    comparisons: tuple[ProjectedCriterion, ...]
    tolerance: Nonnegative
    source: Source = _PROJECTED_SOURCE

    @model_validator(mode="after")
    def _selection(self):
        if not self.comparisons or len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("select distinct nonempty projected criteria")
        if self.source != _PROJECTED_SOURCE:
            raise ValueError("projected checks require their actual implementation source")
        return self

    def verification_checks(self, result):
        """Return one CheckSpec per selected criterion, in the projected frame of ``result``.

        Each check is a dimensionless scalar on the domain [0, inf) compared
        with ``tolerance``. ``overlap_normalization`` and
        ``projected_backward_error`` read stored scalars,
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


def _parts(value, arithmetic):
    """Return the exact real and imaginary parts of a stored Complex128 or real entry."""
    if isinstance(value, Complex128):
        return arithmetic.fraction(value.real), arithmetic.fraction(value.imag)
    return arithmetic.fraction(value), arithmetic.fraction(0)


def _hermiticity(matrix, arithmetic):
    """Return max over i, j of the absolute real and imaginary parts of (S - S†)_ij, exactly.

    ``(S - S†)_ij = S_ij - conj(S_ji)`` has real part ``Re S_ij - Re S_ji``
    and imaginary part ``Im S_ij + Im S_ji``. Because
    ``(S - S†)_ji = -conj((S - S†)_ij)``, the scan visits only ``j >= i``,
    including the diagonal, whose defect is ``2 |Im S_ii|``. A missing
    matrix returns None.
    """
    if matrix is None:
        return None
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("stored Gram matrix must be square")
    defect = arithmetic.fraction(0)
    for i in range(size):
        for j in range(i, size):
            ar, ai = _parts(matrix[i][j], arithmetic)
            br, bi = _parts(matrix[j][i], arithmetic)
            for component in (arithmetic.subtract(ar, br), arithmetic.add(ai, bi)):
                component = arithmetic.absolute(component)
                if arithmetic.le(defect, component):
                    defect = component
    return defect


def _publish(result, options, checks, metrics, *, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Bind each performed scalar check to one new receipt, preserving raw facts.

    Each raw fact is first assembled against its check so that a
    definition-invalid value raises before a receipt exists. Every returned
    fact with a value is then witnessed by the receipt and records the
    options identity and its CheckSpec identity, which
    Certificate.with_verification and assemble_check compare. A fact without
    a value keeps no evidence and attaches as INCONCLUSIVE.
    """
    from nwqlib.execution import KernelApplication, VerificationReceipt, verification_invocation

    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    raw = []
    for check, value in zip(checks, metrics, strict=True):
        fields = dict(quantity=check.name, unit=check.frame.unit, scope=check.frame.scope)
        if value is None:
            fields.update(availability="unknown", reason="selected stored diagnostic unavailable; no computation was replayed")
        else:
            ratio = arithmetic.fraction(value)
            fields.update(availability="concrete", value=Rational(numerator=ratio.numerator, denominator=ratio.denominator),
                          evidence=Evidence(kind="numerical_estimate", source=options.source))
        fact = FramedFact(frame=check.frame, bindings=(), fact=Fact(**fields))
        _assemble_check(check, artifact_id=result.content_id, fact=fact, max_integer_bits=max_integer_bits)
        raw.append(fact)
    application = KernelApplication(name=options.name, implementation=options.source, arguments=(), facts=tuple(raw))
    receipt = VerificationReceipt(plan_id=result.plan_id, result_id=result.content_id,
        invocation_id=verification_invocation(), construction_id=result.construction_id, artifact_ids=(),
        reference=options.source, options_id=options.content_id, applications=(application,))
    return receipt, witness_check_facts(receipt, result, checks)


def witness_check_facts(receipt, result, checks):
    """Return the fact that answers each CheckSpec, witnessed by the completed ``receipt``.

    Each returned fact is the receipt's raw application fact named like its
    check. A fact with a value receives evidence that cites the receipt, the
    Result, the options that produced it and the CheckSpec it answers.
    Certificate.with_verification and assemble_check compare these fields
    before they give PASS or FAIL. A raw fact inside the receipt cannot carry
    them, because it exists before the receipt's content identity does. A
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


def verify_projected(result, *, options: ProjectedVerificationOptions, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
    """Read selected existing scalar/spectrum/Gram diagnostics, never solve again.

    result must supply projected_diagnostics(). The return value is
    (receipt, facts), one VerificationReceipt and one fact per selected
    criterion. The criteria concern the projected problem only, not a
    full-state residual or ground-state identity.
    """
    if type(options) is not ProjectedVerificationOptions:
        raise TypeError("projected verification requires concrete selected options")
    checks = options.verification_checks(result)
    diagnostics = result.projected_diagnostics()
    if type(diagnostics) is not ProjectedDiagnostics:
        raise TypeError("producer must return actual ProjectedDiagnostics")
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    values = []
    for criterion in options.comparisons:
        if criterion == "overlap_normalization":
            value = diagnostics.normalization
        elif criterion == "projected_backward_error":
            value = diagnostics.backward_error
        elif criterion == "gram_hermiticity":
            value = _hermiticity(diagnostics.overlap, arithmetic)
        else:
            spectrum = diagnostics.spectrum
            value = None
            if spectrum:
                minimum = arithmetic.fraction(spectrum[0])
                for item in spectrum[1:]:
                    item = arithmetic.fraction(item)
                    if arithmetic.le(item, minimum):
                        minimum = item
                value = arithmetic.subtract(arithmetic.fraction(0), minimum) if minimum < 0 else arithmetic.fraction(0)
        values.append(value)
    return _publish(result, options, checks, values, max_integer_bits=max_integer_bits)
