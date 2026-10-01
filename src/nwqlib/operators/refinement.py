"""Explicit Pauli and sparse structural bounds from the supplied binary64 data."""

from fractions import Fraction
from math import fsum
from typing import Literal, NamedTuple
from uuid import uuid4

from pydantic import model_validator

from nwqlib.core.records import Float64, Rational, Record, Scope, Source, Text, Unit
from nwqlib.evidence.error_model import ErrorFrame, FramedFact
from nwqlib.evidence.records import Evidence, Fact, WorkProvenance
from .access import Count, DEFAULT_INPUT_BYTES, InputManifest, _check_bytes

Refinement = Literal["certify_pauli_l1", "certify_sparse_rows"]
SOURCE = Source(name="structural_operator_facts", version="1",
    domain="exact binary64 input coefficients; structural enclosures of the admitted operator",
    reference="triangle inequality for unitary Pauli words; Hermitian Gershgorin discs")


class RefinementOption(Record):
    """A concrete available scan and its data/arithmetic growth.

    Attributes:
        operation: ``certify_pauli_l1`` for Hermitian Pauli terms or ``certify_sparse_rows`` for Hermitian CSR/CSC storage.
        access: The admitted input access the scan reads.
        work_law: Text form of its scalar-operation law.
        payload_law: Text form of its byte law.
    """

    operation: Refinement
    access: Text
    work_law: Text
    payload_law: Text


class RefinementReceipt(Record):
    """One actual scan, with input and output populations in an exact frame.

    scalar_operations is a derived arithmetic count, not measured CPU time.
    input_bytes are the borrowed labels or compressed arrays; output_bytes
    count exact numerator/denominator payloads. Neither quantity is process RSS.

    Attributes:
        acquisition_id: Fresh identity of this scan, so a repeated scan is a new acquisition.
        manifest: Manifest of the scanned operator input.
        unit: Unit of the operator's values.
        scope: Evidence scope of the resulting facts.
        operation: The scan that ran.
        input_entries: Pauli terms or stored nonzero entries read.
        input_bytes: Bytes of the labels or compressed arrays read.
        scalar_operations: Derived count of exact scalar operations.
        output_bytes: Bytes of the exact numerators and denominators produced.
    """

    acquisition_id: Text
    manifest: InputManifest
    unit: Unit
    scope: Scope
    operation: Refinement
    input_entries: Count
    input_bytes: Count
    scalar_operations: Count
    output_bytes: Count


_QUANTITIES = {
    "operator_norm_upper_bound": ("operator", "spectral_norm_upper_bound", "full_operator"),
    "identity_coefficient": ("operator", "identity_coefficient", "full_operator"),
    "centered_norm_upper_bound": ("operator_minus_exact_identity_coefficient", "spectral_norm_upper_bound", "full_operator"),
    "eigenvalue_lower_bound": ("operator", "eigenvalue_lower_bound", "full_operator"),
    "eigenvalue_upper_bound": ("operator", "eigenvalue_upper_bound", "full_operator"),
}


def _frame(manifest, unit, scope, name):
    """Return the full-operator ErrorFrame of one structural quantity, conditioned on this exact input."""
    quantity, metric, domain = _QUANTITIES[name]
    return ErrorFrame(quantity=quantity, metric=metric, unit=unit, scope=scope,
        conditioning="admitted native operator " + manifest.reference.content_id, domain=domain)


def _fraction(value):
    return (Fraction(value.numerator, value.denominator) if isinstance(value, Rational)
            else Fraction(value.value))


class OperatorFactReport(Record):
    """Stored facts and scan receipts. Construction never acquires native data.

    Attributes:
        manifest: Manifest of the operator input the facts describe.
        unit: Unit of the operator's values.
        scope: Evidence scope of the facts.
        facts: One FramedFact per structural quantity, unknown until a scan acquires it.
        options: Scans available for this input, empty when none applies.
        receipts: Receipts of every scan in this report's chain, in run order. A later scan's facts replace an earlier scan's facts for the same quantities.
        reason: Why no scan is available, required exactly when options is empty.
    """

    manifest: InputManifest
    unit: Unit
    scope: Scope
    facts: tuple[FramedFact, ...]
    options: tuple[RefinementOption, ...]
    receipts: tuple[RefinementReceipt, ...] = ()
    reason: Text | None = None

    @model_validator(mode="after")
    def _association(self):
        """Associate scalar operator facts and refinement receipts with one input, frame and
        valid numerical domain.
        """
        if bool(self.options) == (self.reason is not None):
            raise ValueError("unavailable refinement requires its own reason")
        names = [statement.fact.quantity for statement in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("operator facts require distinct quantities")
        receipts = {receipt.content_id: receipt for receipt in self.receipts}
        for statement in self.facts:
            name = statement.fact.quantity
            if name not in _QUANTITIES or statement.frame != _frame(self.manifest, self.unit, self.scope, name):
                raise ValueError("operator fact requires its exact input, quantity and frame")
            if statement.bindings:
                raise ValueError("structural operator facts have no realization binding")
            value = statement.fact.value
            if value is not None:
                if not isinstance(value, (Rational, Float64)):
                    raise ValueError("operator scalar facts require real values")
                signed = value.numerator if isinstance(value, Rational) else value.value
                if name in {"operator_norm_upper_bound", "centered_norm_upper_bound"} and signed < 0:
                    raise ValueError("operator norm bounds must be nonnegative")
            evidence = statement.fact.evidence
            if evidence is not None and evidence.status == "witnessed":
                if (evidence.subject_id != self.manifest.reference.content_id
                        or evidence.witnessed_scope != self.scope):
                    raise ValueError("operator evidence requires this exact subject and scope")
                if evidence.source == SOURCE and evidence.artifact not in receipts:
                    raise ValueError("acquired structural fact requires its stored acquisition receipt")
        values = {statement.fact.quantity: statement.fact.value for statement in self.facts}
        lower, upper = values.get("eigenvalue_lower_bound"), values.get("eigenvalue_upper_bound")
        if lower is not None and upper is not None and _fraction(lower) > _fraction(upper):
            raise ValueError("eigenvalue lower bound must not exceed the same-spectrum upper bound")
        if any((receipt.manifest, receipt.unit, receipt.scope) != (self.manifest, self.unit, self.scope)
               for receipt in self.receipts):
            raise ValueError("refinement receipt belongs to another operator or frame")
        return self


def _options(operator):
    """Return the scans that apply to an admitted Hermitian input with native data, or none."""
    if operator.structure != "hermitian" or operator._data is None:
        return ()
    manifest = operator.manifest
    if "pauli_terms" in manifest.access:
        return (RefinementOption(operation="certify_pauli_l1", access="admitted Hermitian Pauli terms",
            work_law="M*(q+1) label visits, then O(M) exact scalar operations",
            payload_law=("M*(q+16) label bytes and one t*(q+25)+q+4 byte label tile, t<=min(M,1024), "
                         "then a bounded exact-scalar frontier")),)
    if manifest.reference.representation in {"csr", "csc"} and "entries" in manifest.access:
        return (RefinementOption(operation="certify_sparse_rows", access="admitted Hermitian CSR/CSC storage",
            work_law="O(nnz+D) compressed-entry visits and exact scalar operations; no conversion",
            payload_law="borrow immutable native arrays plus a bounded exact-scalar frontier"),)
    return ()


class _PauliCensus(NamedTuple):
    """Binary64 identity coefficient, non-identity L1 norm and centered table of a Pauli operator.

    Attributes:
        center: ``fsum`` of the real parts of the identity-word coefficients.
        alpha: ``fsum`` of the absolute non-identity coefficients.
        centered: Ordered ``(label, coefficient)`` pairs of the nonzero non-identity terms.
    """

    center: float
    alpha: float
    centered: tuple


def _pauli_census(raw, num_qubits):
    """Binary64 identity coefficient, non-identity L1 norm and centered table.

    The Lanczos Method uses ``[center - alpha, center + alpha]`` as the
    spectral frame of its Chebyshev walk. ``fsum`` makes the sums correctly
    rounded, and exact-zero non-identity terms are omitted from the ordered
    table. ``_pauli_values`` computes the certified rational counterparts.
    """
    identity = "I" * num_qubits
    center = fsum(float(c.real) for label, c in raw if label == identity)
    centered = tuple((label, float(c.real)) for label, c in raw if label != identity and c != 0)
    alpha = fsum(abs(c) for _, c in centered)
    return _PauliCensus(center, alpha, centered)


def _scalar_frontier(count):
    """Byte allowance for the exact integers that a scan over ``count`` entries keeps alive at once.

    The allowance is 32 integer slots of ``3174 + bit_length(count)`` bits.
    Both the bit width and the slot count are derived below.
    """
    # Finite binary64 values share denominator 2**1074 and magnitude <2**1024.
    # A sum of count values therefore needs at most 2098+bit_length(count)
    # numerator bits. Cross comparison with a denominator needs another 1074.
    # The constant 3174 is 2098 + 2 + 1074. The two extra bits cover the
    # Gershgorin radius, which sums a real and an imaginary magnitude per
    # entry (2*count terms), and the diagonal plus or minus that radius.
    #
    # Slot count. Each Fraction holds two integers. The most integers are
    # alive in _sparse_values while |Re a| + |Im a| is formed for the radius
    # of a row after the first. Eleven Fractions are alive then: lower, upper
    # and norm, the previous row's lo, hi and row_norm, this row's diagonal,
    # radius and real, and the two magnitudes being added, 22 integers in
    # all. CPython's Fraction._add, like _sub, holds at most seven more
    # integers at once, its g, s, t, g2, t // g2, db // g2 and s * (db // g2),
    # so the peak is 29 and 32 slots cover it. _pauli_values peaks at 19,
    # the five Fractions full, center, centered, value and magnitude, one
    # finished difference and seven temporaries. Integers held inside
    # C-level int arithmetic and math.gcd, Python object overhead and CPU
    # time are not counted.
    return 32 * ((3174 + max(1, count).bit_length() + 7) // 8)


def _pauli_values(raw, num_qubits):
    """Exact rational Pauli L1 bounds of ``H = c_I I + sum_{P != I} c_P P``.

    Every Pauli word has spectral norm one, so the triangle inequality gives
    ``||H|| <= sum_P |c_P|`` and places the spectrum in
    ``[c_I - sum_{P != I} |c_P|, c_I + sum_{P != I} |c_P|]``. The stored
    binary64 coefficients are summed as exact fractions, so the bounds hold
    for the admitted operator without rounding. Returns the values and the
    derived scalar-operation count.
    """
    full = center = centered = Fraction(0)
    identity = "I" * num_qubits
    for label, coefficient in raw:
        if coefficient.imag != 0:
            raise ValueError("Hermitian Pauli refinement requires real coefficients")
        value = Fraction(float(coefficient.real))
        magnitude = abs(value)
        full += magnitude
        if label == identity:
            center += value
        else:
            centered += magnitude
    return dict(operator_norm_upper_bound=full, identity_coefficient=center,
        centered_norm_upper_bound=centered, eigenvalue_lower_bound=center-centered,
        eigenvalue_upper_bound=center+centered), 4*len(raw)+2


def _sparse_values(operator, max_bytes):
    """Exact rational Gershgorin bounds of an admitted Hermitian CSR/CSC matrix.

    By the Gershgorin circle theorem every eigenvalue lies in some interval
    ``[a_ii - R_i, a_ii + R_i]`` with ``R_i = sum_{j != i} |a_ij|``. For a
    Hermitian matrix the spectral norm equals the spectral radius, which is
    at most the largest absolute row sum. The diagonal of an exactly
    Hermitian matrix is real, and ``|Re a| + |Im a| >= |a|`` keeps the
    radius rational. Returns the values and the derived scalar-operation
    count.
    """
    data, indices, indptr = operator.sparse_entries(max_bytes=max_bytes)
    dimension = operator.basis.dimension
    lower = upper = None
    norm = Fraction(0)
    operations = 0
    # For Hermitian A, CSC columns give the same absolute sums and diagonal as
    # rows. |Re a|+|Im a| >= |a| provides a rational Gershgorin enclosure.
    for row in range(dimension):
        diagonal = radius = Fraction(0)
        for offset in range(int(indptr[row]), int(indptr[row + 1])):
            value = data[offset]
            real = Fraction(float(value.real))
            operations += 1
            if int(indices[offset]) == row:
                diagonal = real
            else:
                radius += abs(real) + abs(Fraction(float(value.imag)))
                operations += 5
        lo, hi = diagonal-radius, diagonal+radius
        row_norm = abs(diagonal)+radius
        lower = lo if lower is None else min(lower, lo)
        upper = hi if upper is None else max(upper, hi)
        norm = max(norm, row_norm)
        # Per row: lo, hi, |diagonal| and its sum with the radius, the min and
        # max updates after the first row, and the norm maximum. The entry loop
        # above charges one conversion per entry and five more operations per
        # off-diagonal radius term.
        operations += 4 + int(row > 0)*2 + 1
    return dict(operator_norm_upper_bound=norm, eigenvalue_lower_bound=lower,
                eigenvalue_upper_bound=upper), operations


def _publish(operator, unit, scope, selected, values, *, entries, input_bytes, operations):
    """Publish one scan's exact values as framed facts with a fresh receipt.

    Each call creates a new acquisition identity, so a repeated scan is a new
    acquisition and never relabels earlier work. The identity coefficient is
    an exact read of the input and the other quantities are certified bounds.
    """
    scalars = {name: Rational(numerator=value.numerator, denominator=value.denominator)
               for name, value in values.items()}
    output_bytes = sum((abs(value.numerator).bit_length()+7)//8 + (value.denominator.bit_length()+7)//8
                       for value in values.values())
    receipt = RefinementReceipt(acquisition_id=str(uuid4()), manifest=operator.manifest,
        unit=unit, scope=scope, operation=selected, input_entries=entries, input_bytes=input_bytes,
        scalar_operations=operations, output_bytes=output_bytes)
    work = WorkProvenance(stage="planning", description=f"{selected}: {entries} input entries, "
        f"{operations} derived scalar operations; {input_bytes} borrowed input bytes, "
        f"{output_bytes} output scalar bytes; CPU and RSS unmeasured", artifact=receipt.content_id)
    facts = []
    for name, scalar in scalars.items():
        evidence = Evidence(kind="proved_relation" if name == "identity_coefficient" else "certified_bound",
            source=SOURCE, status="witnessed", artifact=receipt.content_id, witnessed_scope=scope,
            subject_id=operator.reference.content_id, work=(work,))
        facts.append(FramedFact(frame=_frame(operator.manifest, unit, scope, name), bindings=(),
            fact=Fact(quantity=name, unit=unit, scope=scope, availability="concrete", value=scalar,
                      evidence=evidence)))
    return tuple(facts), receipt


def _report(operator, unit, scope, facts, receipts, *, parent_id=None):
    options = _options(operator)
    return OperatorFactReport(manifest=operator.manifest, unit=unit, scope=scope,
        facts=facts, options=options, receipts=receipts, parent_id=parent_id,
        reason=None if options else "no_known_refinement: supported Hermitian Pauli or CSR/CSC data is unavailable")


def refine_operator_facts(operator, *, unit: Unit, scope: Scope, max_bytes=DEFAULT_INPUT_BYTES,
                          refinements: tuple[Refinement, ...] = (),
                          previous: OperatorFactReport | None = None) -> OperatorFactReport:
    """Run only explicitly selected scans; reuse unchanged acquired facts.

    Selecting a scan again creates a new acquisition receipt. Empty refinements
    use metadata only. No matvec, eigensolve, backend or dense conversion runs.
    """
    from .inputs import OperatorInput

    if not isinstance(operator, OperatorInput):
        raise TypeError("operator facts require an OperatorInput")
    if previous is not None and not isinstance(previous, OperatorFactReport):
        raise TypeError("previous must be an OperatorFactReport")
    if type(refinements) is not tuple or len(refinements) != len(set(refinements)):
        raise ValueError("refinements must be an explicit finite tuple of distinct operations")
    available = tuple(option.operation for option in _options(operator))
    if any(selected not in available for selected in refinements):
        raise ValueError("selected refinement is unavailable for this operator access/structure")
    if previous is not None and (previous.manifest, previous.unit, previous.scope) != (operator.manifest, unit, scope):
        raise ValueError("previous report belongs to another operator or frame")
    # One 8-byte reference per new and kept receipt in the returned report.
    _check_bytes(8*(len(refinements)+(0 if previous is None else len(previous.receipts))), max_bytes,
                 "refinement receipt references")
    facts = (tuple(FramedFact(frame=_frame(operator.manifest, unit, scope, name), bindings=(),
        fact=Fact(quantity=name, unit=unit, scope=scope, availability="unknown",
                  reason="this operator quantity has no acquired fact")) for name in _QUANTITIES)
        if previous is None else previous.facts)
    receipts = () if previous is None else previous.receipts
    for selected in refinements:
        count = operator.manifest.shape[0] if selected == "certify_pauli_l1" else operator._data.nnz
        q = operator.basis.dimension.bit_length()-1
        # Materialized labels (q characters and a 16-byte coefficient per term,
        # the returned payload of the PauliTerms.labels law, whose decoding
        # tile labels() admits itself and releases before the scan) or the
        # borrowed CSR/CSC arrays, plus the live exact integers of the scan (_scalar_frontier).
        input_bytes = count*(q+16) if selected == "certify_pauli_l1" else operator.manifest.payload_bytes
        _check_bytes(input_bytes + _scalar_frontier(count), max_bytes, "operator structural bounds")
        values, operations = (_pauli_values(operator.pauli_terms().labels(max_bytes=max_bytes), q)
                              if selected == "certify_pauli_l1" else _sparse_values(operator, max_bytes))
        acquired, receipt = _publish(operator, unit, scope, selected, values,
            entries=count, input_bytes=input_bytes, operations=operations)
        replacements = {fact.fact.quantity: fact for fact in acquired}
        facts = tuple(replacements.pop(fact.fact.quantity, fact) for fact in facts) + tuple(replacements.values())
        receipts = (*receipts, receipt)
    return _report(operator, unit, scope, facts, receipts,
                   parent_id=None if previous is None else previous.content_id)
