"""Ordered algebra and before-work expansion checks without full-state references."""

import numpy as np
import pytest
from scipy import sparse

from nwqlib.operators import (
    FactorizedOperatorProduct, fermion_table, ingest_fermion, ingest_pauli, operator_input,
)


def test_fermion_order_exact_cancellation_and_jw_sign():
    small = 2.**-55
    operator = ingest_fermion([(1, ()), (small, ()), (-1, ())], num_modes=2)
    assert operator.fermion_terms().coefficients.tolist() == [small]
    left = ingest_fermion([(1, ((0, 0), (1, 1)))], num_modes=2)
    right = ingest_fermion([(1, ((1, 1), (0, 0)))], num_modes=2)
    assert left.reference != right.reference
    a = fermion_table([(1, ((1, 0),))], num_modes=2)
    creation = fermion_table([(1, ((1, 1),))], num_modes=2)
    assert a.to_pauli(mapping="jw").pauli_terms().labels() == (("XZ", .5), ("YZ", .5j))
    assert a.to_pauli(mapping="z_free").pauli_terms().labels() == (("XI", .5), ("YI", .5j))
    # a†a=(I-Z1)/2; this discriminates ladder order and the Y phase sign.
    assert creation.product(a).to_pauli(mapping="jw").pauli_terms().labels() == (("II", .5), ("ZI", -.5))
    anticommutator = fermion_table([(1, ((1, 0), (1, 1))), (1, ((1, 1), (1, 0)))], num_modes=2)
    assert anticommutator.to_pauli(mapping="jw").pauli_terms().labels() == (("II", 1),)
    for array in (a.offsets, a.modes, a.actions, a.coefficients):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_multiword_fermion_parity_and_bounded_mapping(monkeypatch):
    from nwqlib.operators._pauli import PauliTerms
    from nwqlib.operators._fermion import mapping_requirements
    a = fermion_table([(1, ((64, 0),))], num_modes=65)
    assert a.to_pauli(mapping="jw").pauli_terms().labels() == (("X"+"Z"*64, .5), ("Y"+"Z"*64, .5j))
    composite = fermion_table([(1, ((1, 0), (0, 1)))], num_modes=2)
    monkeypatch.setattr(PauliTerms, "product", lambda *a, **k: pytest.fail("mapping product before byte check"))
    with pytest.raises(ValueError, match="max_bytes"):
        composite.to_pauli(mapping="jw", max_bytes=100)
    with pytest.raises(ValueError, match="max_bytes"):
        mapping_requirements((10**10,), num_modes=2)


def test_fermion_mapping_admits_the_mask_identity_share_before_products(monkeypatch):
    from nwqlib.operators._pauli import PauliTerms
    from nwqlib.operators._fermion import mapping_requirements
    # Two ladders map to four rows. The final mask ingestion charges each row
    # the identity JSON share 6q + 360, so the mapping law includes it and a
    # budget one byte short fails before the first ladder product. The law is
    # the raw table 6(24 + 2*9 + 8) + 1, two ladder images of 4L, the product
    # candidates 2*4L + 32W and four final rows of 6L plus the share.
    composite = fermion_table([(1, ((1, 0), (0, 1)))], num_modes=2)
    law = 301 + 2 * 4 * 32 + (2 * 4 * 32 + 32) + 4 * (6 * 32 + 6 * 2 + 360)
    assert mapping_requirements((2,), num_modes=2)[1] == law
    assert composite.to_pauli(mapping="jw", max_bytes=law).pauli_terms().labels() == \
        composite.to_pauli(mapping="jw").pauli_terms().labels()
    monkeypatch.setattr(PauliTerms, "product", lambda *a, **k: pytest.fail("mapping product before byte check"))
    for budget in (law - 1, law - 4 * (6 * 2 + 360)):
        with pytest.raises(ValueError, match="operator expansion needs"):
            composite.to_pauli(mapping="jw", max_bytes=budget)


def test_factorized_action_order_and_cumulative_products():
    a = operator_input([[1, 1], [0, 1]])
    b = operator_input([[2, 0], [0, 3]])
    ab, ba = FactorizedOperatorProduct((a, b)), FactorizedOperatorProduct((b, a))
    vector = np.array([1., 2.])
    np.testing.assert_array_equal(ab.matvec(vector, max_products=8), [8, 6])
    np.testing.assert_array_equal(ba.matvec(vector, max_products=8), [6, 6])
    assert ab.factors[0] is a and ab.manifest.factors == (a.reference, b.reference)
    with pytest.raises(ValueError, match="max_products"):
        ab.matvec(None, max_products=7)
    with pytest.raises(ValueError, match="homogeneous"):
        ab.expand()


def test_sparse_and_pauli_expansion_preserve_order_and_refuse_before_products(monkeypatch):
    a = operator_input(sparse.csc_array([[1, 1], [0, 1]]))
    b = operator_input(sparse.csr_matrix([[2, 0], [0, 3]]))
    product = FactorizedOperatorProduct((a, b))
    expanded = product.expand(max_products=3)
    assert tuple(expanded.entry(i, j) for i in range(2) for j in range(2)) == (2, 3, 0, 3)
    monkeypatch.setattr(sparse.csr_matrix, "__matmul__", lambda *a: pytest.fail("sparse product before count check"))
    with pytest.raises(ValueError, match="max_products"):
        product.expand(max_products=2)
    x, y = ingest_pauli([("X", 1)], num_qubits=1), ingest_pauli([("Y", 1)], num_qubits=1)
    assert FactorizedOperatorProduct((x, y)).expand().pauli_terms().labels() == (("Z", 1j),)
    assert FactorizedOperatorProduct((y, x)).expand().pauli_terms().labels() == (("Z", -1j),)
    # Each chain step is charged the bytes that its PauliTerms.product call
    # checks (operators/_pauli.py::pauli_product_requirements), and the final
    # operator row adds four L-byte coalescing copies and the identity JSON
    # share 6q + 360 per product term, so an admitted chain runs every step.
    from nwqlib.operators import _pauli
    checked, check = [], _pauli._check_bytes

    def record(size, limit, operation):
        checked.extend([size] if operation == "Pauli product" else [])
        return check(size, limit, operation)

    monkeypatch.setattr(_pauli, "_check_bytes", record)
    z = ingest_pauli([("X", 1), ("Y", 2), ("Z", 3)], num_qubits=1)
    for factors, count in (((x, y), 1), ((z, z, z), 27)):
        checked.clear()
        law = FactorizedOperatorProduct(factors).expansion_requirements()[1]
        FactorizedOperatorProduct(factors).expand(max_bytes=law)
        assert len(checked) == len(factors) - 1
        assert law == sum(checked) + count * (4 * 32 + 6 * 1 + 360)
    law = FactorizedOperatorProduct((x, y)).expansion_requirements()[1]
    monkeypatch.setattr(type(x.pauli_terms()), "product", lambda *a, **k: pytest.fail("Pauli product before byte check"))
    with pytest.raises(ValueError, match="operator expansion needs"):
        FactorizedOperatorProduct((x, y)).expand(max_bytes=law - 1)


def test_declared_factor_is_metadata_only_and_raw_stream_checks_precede_consumption():
    from nwqlib.operators import OperatorInput
    native = operator_input([[1, 0], [0, 2]])
    stored = OperatorInput.from_record(native.to_record())
    product = FactorizedOperatorProduct((stored, native))
    for action in (lambda: product.matvec(None), product.expand):
        with pytest.raises(ValueError, match="native data is unavailable"):
            action()
    class Untouched:
        def __len__(self):
            return 1000
        def __iter__(self):
            pytest.fail("oversized explicit term stream was consumed")
    with pytest.raises(ValueError, match="max_bytes"):
        fermion_table(Untouched(), num_modes=2, max_bytes=100)
