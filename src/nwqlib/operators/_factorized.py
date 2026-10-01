"""Preserved factor tuples with explicit bounded classical action/expansion."""

from dataclasses import dataclass

from scipy import sparse

from .access import DEFAULT_INPUT_BYTES, ProductManifest, _check_bytes, _check_products
from .inputs import OperatorInput, _matvec_requirements, _coalesced_input, _pauli_identity_bytes, ingest_sparse
from ._fermion import _check, _count_product, _fermion_law, _coalesced_fermion
from ._pauli import pauli_product_requirements


@dataclass(frozen=True, eq=False, init=False, slots=True)
class FactorizedOperatorProduct:
    """Nonempty immutable ordered tuple of admitted compatible operator handles.

    Attributes:
        factors: Existing payload owners, referenced without traversal or copies.
        manifest: Ordered InputRefs and basis; structure is always general.
    """

    factors: tuple[OperatorInput, ...]
    manifest: ProductManifest

    def __init__(self, factors, *, max_bytes=DEFAULT_INPUT_BYTES):
        """Reference admitted handles in order, without reading their payloads.

        All factors must share one basis and dimension. Referenced payload
        bytes are summed per occurrence, or unknown if any factor's size is
        unknown.
        """
        if type(factors) is not tuple or not factors:
            raise ValueError("factorized product requires a nonempty native tuple")
        _check_bytes(8 * len(factors), max_bytes, "factor references")
        if any(not isinstance(f, OperatorInput) for f in factors):
            raise TypeError("factors must be admitted OperatorInput handles")
        basis = factors[0].manifest.basis
        metadata_bytes = 8 * len(factors) + (basis.dimension.bit_length() + 7) // 8
        _check_bytes(metadata_bytes, max_bytes, "factor references")
        if any(f.manifest.basis != basis for f in factors):
            raise ValueError("factorized product requires identical basis and dimension")
        sizes = tuple(f.manifest.payload_bytes for f in factors)
        manifest = ProductManifest(factors=tuple(f.manifest.reference for f in factors), basis=basis,
                                   payload_bytes=metadata_bytes,
                                   referenced_payload_bytes=None if None in sizes else sum(sizes))
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "manifest", manifest)

    def action_requirements(self):
        """Cumulative local action sizes plus the outer vector frontier.

        All capabilities are checked first from metadata. Two factor-result
        vectors are live across a step; original input may remain referenced.
        Pauli conversion/chunk space is included by the shared native law.
        """
        if any("matvec" not in f.manifest.access or
               (f.manifest.reference.representation == "pauli" and
                f.manifest.basis.dimension.bit_length() - 1 > 64) for f in self.factors):
            raise ValueError("classical matvec is unavailable for a factor")
        laws = tuple(_matvec_requirements(f) for f in self.factors)
        d = self.manifest.basis.dimension
        return d, 32 * d + sum(law[1] for law in laws), sum(law[2] for law in laws)

    def matvec(self, vector, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000,
               limit_name="max_products"):
        """Admit the entire action before vector access, then apply right first.

        ``limit_name`` names the caller's option that supplies ``max_products``
        in a refusal.
        """
        requirements = self.action_requirements()
        _check(max_bytes, requirements)
        _check_products(requirements[2], max_products, limit_name)
        current = vector
        for factor in reversed(self.factors):
            current = factor.matvec(current, max_bytes=max_bytes, max_products=max_products,
                                    limit_name=limit_name)
        return current

    def expansion_requirements(self, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
        """Admit homogeneous expansion before any product/native multiplication.

        Sparse C counts candidate scalar products, with
        C >= structural nnz >= final nnz. Later chain steps conservatively use
        prior nnz upper bound times next maximum row nnz. No pattern pass.
        A Pauli chain step multiplies the running M-row product by the next
        K-row factor and is charged ``pauli_product_requirements(M, K, W)``,
        the bytes that its PauliTerms.product call checks.
        """
        if any(f.manifest.identity_status != "ingested" for f in self.factors):
            raise ValueError("expansion is unavailable for a declared factor")
        for factor in self.factors:
            factor._require_data()
        representations = {f.manifest.reference.representation for f in self.factors}
        sparse_factors = representations <= {"csr", "csc"}
        if not sparse_factors and (len(representations) != 1 or not representations <= {"pauli", "fermion"}):
            raise ValueError("expansion requires homogeneous Pauli, Fermion or sparse factors")
        # Admit metadata/index traversal before touching sparse storage or offsets.
        d = self.manifest.basis.dimension
        if sparse_factors:
            _check_bytes(8 * d, max_bytes, "sparse row census")
            return _sparse_chain_requirements(self.factors, d, max_bytes, max_products)
        q = d.bit_length() - 1
        _check_bytes(8 * len(self.factors) + (q + 8) // 8, max_bytes, "factor expansion metadata")
        tables = tuple(f._data for f in self.factors)
        count = len(tables[0])
        p = len(tables[0].modes) if representations == {"fermion"} else 0
        payload, work, items = 0, max(q, len(tables)), max(count + p, len(tables))
        for table in tables[1:]:
            new_count = _count_product(count, len(table), max_bytes)
            if representations == {"pauli"}:
                w = (q + 63) // 64
                # The step's bytes are the law its PauliTerms.product call checks.
                law = (new_count, pauli_product_requirements(count, len(table), w), new_count * (w + 1))
            else:
                p = (_count_product(p, len(table), max_bytes)
                     + _count_product(count, len(table.modes), max_bytes))
                law = _fermion_law(new_count, p, q)
            count = new_count
            items, payload, work = max(items, law[0]), payload + law[1], work + law[2]
            _check(max_bytes, (items, payload, work))
        if representations == {"pauli"}:
            w = (q + 63) // 64
            # Final coalescing into an operator: the four copies of
            # _coalesced_input plus the identity JSON share that every Pauli
            # operator admission charges per term.
            final = (count, count * (4 * (16 * w + 16) + _pauli_identity_bytes(q)), max(q, count * (w + 1)))
        else:
            final = _fermion_law(count, p, q)
        law = max(items, final[0]), payload + final[1], work + final[2]
        _check(max_bytes, law)
        return law

    def expand(self, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
        """Explicit same-representation expansion, with final coalescing only."""
        self.expansion_requirements(max_bytes=max_bytes, max_products=max_products)
        representation = self.factors[0].manifest.reference.representation
        if representation in ("csr", "csc"):
            # Normalization/snapshot/workspace for every factor and intermediate
            # was admitted before this first native operation.
            current = sparse.csr_matrix(self.factors[0]._data, copy=True)
            for factor in self.factors[1:]:
                right = sparse.csr_matrix(factor._data, copy=True)
                current = current @ right
            # Do not trust inherited SciPy canonical flags after multiplication.
            current.has_sorted_indices = False
            current.has_canonical_format = False
            current.sum_duplicates()
            current.eliminate_zeros()
            current.sort_indices()
            return ingest_sparse(current, max_bytes=max_bytes)
        current = self.factors[0]._data
        for factor in self.factors[1:]:
            current = current.product(factor._data, max_bytes=max_bytes)
        return _coalesced_input(current) if representation == "pauli" else _coalesced_fermion(current, max_bytes=max_bytes)


def _sparse_layout(nnz, d):
    # Complex values plus int64 indices/pointers bound both native index widths.
    return 24 * nnz + 8 * (d + 1)


def _sparse_chain_requirements(factors, d, max_bytes, max_products):
    """Admit a whole sparse product chain from row counts, before multiplying.

    For the first product A*B the candidate count is the sum over stored
    ``A[i, k]`` of ``row_nnz(B, k)``, which bounds the structural and final
    nnz of A*B. Each later factor multiplies the previous nnz bound by its
    largest row nnz. Products, bytes and work are checked cumulatively, so no
    prefix product runs only to discover that a later step exceeds a limit.
    The laws are in docs/development/input_contracts.md.
    """
    scan = sum(f._data.nnz + d for f in factors)
    # All CSR/CSC normalization copies, including index conversion and scratch.
    payload = sum(2 * _sparse_layout(f._data.nnz, d) for f in factors)
    work, items = 2 * scan, max(f._data.nnz for f in factors)
    _check(max_bytes, (items, payload, work))
    _check_products(0, max_products)
    products = 0
    previous = factors[0]._data.nnz
    for step, factor in enumerate(factors[1:], start=1):
        right = factor._data
        # CSC needs a row-count histogram; explicitly covered by traversal law.
        row_counts = [0] * d
        if right.format == "csr":
            for row in range(d):
                row_counts[row] = int(right.indptr[row + 1]) - int(right.indptr[row])
        else:
            for row in right.indices:
                row_counts[int(row)] += 1
        if step == 1:
            left = factors[0]._data
            candidates = 0
            if left.format == "csr":
                for k in left.indices:
                    candidates += row_counts[int(k)]
                    _check_products(products + candidates, max_products)
            else:
                for k in range(d):
                    candidates += _count_product(int(left.indptr[k + 1]) - int(left.indptr[k]),
                                                 row_counts[k], max_products - products)
                    _check_products(products + candidates, max_products)
        else:
            candidates = _count_product(previous, max(row_counts, default=0), max_products - products)
        products += candidates
        _check_products(products, max_products)
        upper = min(d * d, candidates)
        # Inputs/intermediate CSR, multiplication workspace, canonical sorting,
        # finite scan, Hermiticity comparison and final immutable ingestion.
        payload += 4 * _sparse_layout(upper, d) + 32 * d
        work += candidates + previous + right.nnz + d + upper * max(1, upper.bit_length())
        items = max(items, upper)
        _check(max_bytes, (items, payload, work))
        previous = upper
    payload += 4 * _sparse_layout(previous, d)
    work += 4 * (previous + d) + previous * max(1, previous.bit_length())
    law = items, payload, work
    _check(max_bytes, law)
    return law
