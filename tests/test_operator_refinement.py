"""Independent exact structural bounds and same-input acquisition history."""

from fractions import Fraction

import pytest
from scipy import sparse

from nwqlib.core import Rational, Scope, Unit
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


@pytest.mark.parametrize("factory", [sparse.csr_matrix])
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
