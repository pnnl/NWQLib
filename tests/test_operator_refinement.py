"""Independent exact structural bounds and same-input acquisition history."""

from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

from nwqlib.core import Float64, Rational, Scope, Unit
from nwqlib.operators import OperatorFactReport, OperatorInput, ingest_pauli, operator_input, refine_operator_facts

UNIT = Unit(symbol="Ha", dimension="energy")
SCOPE = Scope(domain="finite Hermitian input")


def refine(operator, **kwargs):
    return refine_operator_facts(operator, unit=UNIT, scope=SCOPE, **kwargs)


def values(report):
    return {statement.fact.quantity: Fraction(statement.fact.value.numerator, statement.fact.value.denominator)
            for statement in report.facts if isinstance(statement.fact.value, Rational)}


def test_pauli_l1_and_centered_bounds_keep_binary64_residues():
    operator = ingest_pauli([("I", 2), ("X", .5), ("Z", -.25)], num_qubits=1)
    report = refine(operator, refinements=("certify_pauli_l1",))
    # H=2I+X/2-Z/4 has radius sqrt(5)/4; triangle enclosure is 3/4.
    assert values(report) == dict(operator_norm_upper_bound=Fraction(11, 4), identity_coefficient=2,
        centered_norm_upper_bound=Fraction(3, 4), eigenvalue_lower_bound=Fraction(5, 4), eigenvalue_upper_bound=Fraction(11, 4))
    tiny = ingest_pauli([("Z", 1), ("X", 2.**-53)], num_qubits=1)
    bound = values(refine(tiny, refinements=("certify_pauli_l1",)))["operator_norm_upper_bound"]
    assert bound == 1 + Fraction(1, 2**53) and bound > Fraction(float(bound))
    assert OperatorFactReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("factory", [sparse.csr_matrix, sparse.csc_array])
def test_sparse_gershgorin_zero_row_and_borrowed_scan(factory, monkeypatch):
    operator = operator_input(factory([[2, 1j, 0], [-1j, -1, 0], [0, 0, 0]]))
    arrays = operator.sparse_entries()
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    reads = [0, 0, 0]
    class Counted:
        def __init__(self, array, position):
            self.array, self.position = array, position
        def __len__(self):
            return len(self.array)
        def __getitem__(self, index):
            reads[self.position] += 1
            return self.array[index]
    watched = tuple(Counted(array, index) for index, array in enumerate(arrays))
    monkeypatch.setattr(OperatorInput, "sparse_entries", lambda self, **kwargs: watched)
    for name in ("entry", "dense_array", "matvec"):
        monkeypatch.setattr(OperatorInput, name, lambda *a, **k: pytest.fail("extra numerical access"))
    report = refine(operator, refinements=("certify_sparse_rows",))
    assert values(report) == dict(operator_norm_upper_bound=3, eigenvalue_lower_bound=-2, eigenvalue_upper_bound=3)
    assert reads == [4, 4, 6]
    assert report.receipts[0].input_entries == 4
    assert report.receipts[0].input_bytes == operator.manifest.payload_bytes


def test_metadata_reuses_previous_facts_and_repeated_scan_has_new_identity(monkeypatch):
    operator = ingest_pauli([("I", -2), ("Z", .5)], num_qubits=1)
    first = refine(operator, refinements=("certify_pauli_l1",))
    second = refine(operator, refinements=("certify_pauli_l1",), previous=first)
    assert second.receipts[0] == first.receipts[0] and len(second.receipts) == 2
    assert second.receipts[1].acquisition_id != first.receipts[0].acquisition_id
    assert second.receipts[1].scalar_operations == first.receipts[0].scalar_operations > 0
    estimate = first.facts[0].revise(fact=first.facts[0].fact.revise(value=Float64(value=2.5),
        evidence=first.facts[0].fact.evidence.revise(kind="numerical_estimate")))
    previous = first.revise(facts=(estimate, *first.facts[1:]))
    for name in ("pauli_terms", "sparse_entries", "dense_array", "entry", "matvec"):
        monkeypatch.setattr(OperatorInput, name, lambda *a, **k: pytest.fail("metadata scan"))
    result = refine(operator, previous=previous)
    assert result.facts == previous.facts and result.receipts == previous.receipts
    assert result.facts[0].fact.evidence.kind == "numerical_estimate"
    restored = OperatorInput.from_record(operator.to_record())
    assert not refine(restored).options
    with pytest.raises(ValueError, match="unavailable"):
        refine(restored, refinements=("certify_pauli_l1",))


def test_invalid_bound_or_association_rejects_and_small_cap_precedes_scan(monkeypatch):
    operator = ingest_pauli([("I", -2), ("Z", .5)], num_qubits=1)
    report = refine(operator, refinements=("certify_pauli_l1",))
    negative = report.facts[0].revise(fact=report.facts[0].fact.revise(value=Rational(numerator=-1, denominator=1)))
    with pytest.raises(ValueError, match="nonnegative"):
        report.revise(facts=(negative, *report.facts[1:]))
    another = operator_input(np.eye(2))
    with pytest.raises(ValueError, match="another operator"):
        refine(another, previous=report)
    monkeypatch.setattr(OperatorInput, "pauli_terms", lambda *a, **k: pytest.fail("read before byte admission"))
    with pytest.raises(ValueError, match="max_bytes"):
        refine(operator, refinements=("certify_pauli_l1",), max_bytes=1)
