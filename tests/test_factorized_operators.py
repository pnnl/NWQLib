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
    # Without max_bytes the limit is the physical memory, which the refusal names.
    with pytest.raises(ValueError, match="mapping exceeds the physical memory"):
        mapping_requirements((10**10,), num_modes=2)


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
    # share 3 max(2q + 120, 150) + 112 per product term, so an admitted chain
    # runs every step.
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
        assert law == sum(checked) + count * (4 * 32 + 3 * 150 + 112)
    law = FactorizedOperatorProduct((x, y)).expansion_requirements()[1]
    monkeypatch.setattr(type(x.pauli_terms()), "product", lambda *a, **k: pytest.fail("Pauli product before byte check"))
    with pytest.raises(ValueError, match="operator expansion needs"):
        FactorizedOperatorProduct((x, y)).expand(max_bytes=law // 2)


@pytest.mark.parametrize("left, right", [("csr", "csr"), ("csr", "csc"), ("csc", "csr"), ("csc", "csc")])
def test_sparse_chain_counts_each_stored_product_before_multiplying(left, right):
    """A two-factor sparse expansion is admitted at exactly its scalar products and refused one below.

    For A*B the products are the sum over k of the stored entries of A in
    column k times those of B in row k. Random 40 x 40 patterns in every
    CSR/CSC combination give that count independently from dense masks.
    """
    rng = np.random.default_rng(len(left) + 3 * len(right) + (left == "csc") + 2 * (right == "csc"))
    masks = [rng.random((40, 40)) < 0.1 for _ in range(2)]
    a, b = (getattr(sparse, f"{form}_matrix")(mask * rng.uniform(1, 2, (40, 40)))
            for form, mask in zip((left, right), masks))
    products = int(masks[0].sum(axis=0) @ masks[1].sum(axis=1))
    chain = FactorizedOperatorProduct((operator_input(a), operator_input(b)))
    expanded = chain.expand(max_products=products)
    np.testing.assert_allclose(expanded._data.toarray(), (a @ b).toarray(), rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match="max_products"):
        chain.expand(max_products=products - 1)


def test_sparse_triple_product_is_limited_by_its_product_count_alone():
    """A later sparse factor is counted exactly and checked only against max_products.

    For A*B*C the products are those of A*B plus, for each stored entry of
    A*B, at most the widest row of C. A random 5 x 5 CSC*CSR*CSC chain
    expands at exactly that count and is refused one below it by the
    product limit. With dimension 2**21, a full column of A and full rows of
    B and C need 2**42 + 2**63 products, which pass with no limit and with
    max_products=2**65, from the index pointers alone.
    """
    from types import SimpleNamespace
    from nwqlib.operators._factorized import _sparse_chain_requirements

    rng = np.random.default_rng(552)
    masks = [rng.random((5, 5)) < t for t in (.45, .55, .4)]
    chain = FactorizedOperatorProduct(tuple(
        operator_input(form(mask.astype(float)))
        for form, mask in zip((sparse.csc_matrix, sparse.csr_matrix, sparse.csc_matrix), masks)))
    first = int(masks[0].sum(axis=0) @ masks[1].sum(axis=1))
    products = first + min(25, first) * int(masks[2].sum(axis=1).max())
    expanded = chain.expand(max_products=products)
    np.testing.assert_array_equal(expanded._data.toarray(), masks[0] * 1 @ masks[1] @ masks[2])
    with pytest.raises(ValueError, match=f"needs {products} scalar products, exceeding max_products"):
        chain.expand(max_products=products - 1)

    d = 2**21
    full = np.concatenate(([0], np.full(d, d, dtype=np.int64)))
    column = SimpleNamespace(_data=SimpleNamespace(format="csc", indptr=full, nnz=d))
    row = SimpleNamespace(_data=SimpleNamespace(format="csr", indptr=full, nnz=d))
    for cap in (None, 2**65):
        items, _, _ = _sparse_chain_requirements((column, row, row), d, 1 << 72, cap)
        assert items == d * d
    with pytest.raises(ValueError, match=f"needs {2**42 + 2**63} scalar products"):
        _sparse_chain_requirements((column, row, row), d, 1 << 72, 2**42 + 2**63 - 1)


def test_pauli_input_reserves_the_identity_share_of_its_widest_term_layout():
    """S(q) = 3 max(2q + 120, 150) + 112 bytes per term (operators/inputs.py::_pauli_identity_bytes).

    A per-term record enters its parent's content hash as a 150-byte
    reference, so the share is constant up to q = 15, and each such record
    keeps its 112-byte content hash. At q = 6 a term costs q + 4L + S =
    6 + 128 + 562 = 696 bytes, so three labels are admitted at exactly 2,088
    bytes and refused one byte below.
    """
    from nwqlib.operators.inputs import _pauli_identity_bytes

    assert [_pauli_identity_bytes(q) for q in (1, 6, 14, 15, 16)] == [562, 562, 562, 562, 568]
    labels = [("XIIIIZ", 1.0), ("IYIIII", -0.5), ("ZZZZZZ", 0.25)]
    assert len(ingest_pauli(labels, num_qubits=6, max_bytes=3 * 696).pauli_terms().labels()) == 3
    with pytest.raises(ValueError, match="raw Pauli input exceeds max_bytes"):
        ingest_pauli(labels, num_qubits=6, max_bytes=3 * 696 - 1)
