"""Supplied real factor polynomials, with explicit bounded Jordan–Wigner conversion.

The input defines E0 + dΓ(T) + sum(dΓ(V diag(w) V.T)**2)/2. It does
not assert that the supplied columns are orthogonal or reconstruct a molecule.
"""

from dataclasses import dataclass, field
from itertools import chain, repeat
import sys
from typing import Literal

import numpy as np
from pydantic import model_validator

from nwqlib._validation import finite_real
from nwqlib.core.records import Basis, ContentID, Float64, InputRef, Rational, Record, Scope, Source, Unit
from nwqlib.evidence.error_model import ErrorFrame, FramedFact
from nwqlib.evidence.records import Evidence, Fact, WorkProvenance
from .access import Count, DEFAULT_INPUT_BYTES, InputManifest, _check_bytes, _check_products
from ._fermion import fermion_table, mapping_requirements
from .inputs import ORDER, OperatorInput, _array_input, _digest, _freeze_array

SOURCE = Source(name="supplied real DF conversion", version="1",
    domain="real binary64 factor polynomial; interleaved alpha/beta modes; q0 least significant",
    reference="H=E0+dGamma(T)+sum(dGamma(V diag(w) V.T)^2)/2; explicit JW")
CONSTRUCTION = "binary64; upper-triangle B and B-squared; unordered-pair Gram; shared Hermitian coefficients; JW"


class DFManifest(Record):
    """Detached factor snapshot identity and declared source/orbital/energy frame.

    payload_digest hashes actual T, ordered V/w arrays and their native shapes.
    constant_energy is E0, already including any supplied core/nuclear constant.
    columns keeps every factor column, including signed/zero inner weights.
    dtype/byte_order/payload_bytes describe T/V/w only; weights have sqrt(energy)
    units and vectors are dimensionless. reference identifies this whole record,
    so a source declaration never substitutes for the observed payload identity.
    """

    payload_digest: ContentID
    source: InputRef
    orbital_basis: Basis
    energy_unit: Unit
    constant_energy: Float64
    columns: tuple[Count, ...]
    dtype: Literal["float64"] = "float64"
    byte_order: Literal["little", "big"]
    payload_bytes: Count
    convention: Literal["E0+dGamma(T)+sum(dGamma(V diag(w) V.T)^2)/2"] = (
        "E0+dGamma(T)+sum(dGamma(V diag(w) V.T)^2)/2")

    @model_validator(mode="after")
    def _domain(self):
        """Require an energy unit, factors with at least one column and matching payload bytes.

        T has ``n*n`` float64 entries, and each factor column has ``n``
        vector entries and one weight, so the payload is
        ``8 * (n*n + (n + 1) * sum(columns))`` bytes.
        """
        n = self.orbital_basis.dimension
        if self.energy_unit.dimension != "energy" or any(r == 0 for r in self.columns):
            raise ValueError("DF requires an energy unit and positive column counts")
        if self.payload_bytes != 8 * (n*n + (n+1)*sum(self.columns)):
            raise ValueError("DF payload bytes must match T and the ordered factor shapes")
        return self

    @property
    def reference(self):
        """Small derived input identity; does not inspect native factor arrays."""
        return InputRef(identity=self.content_id, representation="supplied_real_df", source=SOURCE)


def _scope(manifest, output):
    return Scope(domain=f"DF {manifest.reference.content_id} explicitly converted to Pauli {output.reference.content_id}")


def _error_frame(manifest, output, scope):
    return ErrorFrame(quantity="factor_operator_minus_materialized_pauli", metric="spectral_norm",
        unit=manifest.energy_unit, scope=scope, domain="full_operator",
        conditioning=f"exact raw binary64 DF polynomial {manifest.reference.content_id}; "
                     f"actual Pauli input {output.reference.content_id}")


class DFConversionReceipt(Record):
    """input/output bind a conversion; no native factors or Pauli terms are stored.

    construction fixes arithmetic/symmetry/mapping. conversion_error is unknown
    in the full-operator frame, not a propagated energy-output guarantee.
    raw_fermion_rows/raw_jw_candidates count pre-simplification populations;
    pauli_terms counts the actual coalesced result. items/payload_bytes/work
    count the cumulative acquisition envelope before publication, not wall time,
    allocator bytes or final output size. payload_bytes includes the identity
    JSON share of ``inputs._pauli_identity_bytes`` for every raw mapped row,
    which bounds the Plan records that will hold the coalesced terms. Other
    text, JSON and Python object overhead are outside these logical counters.
    """

    input: DFManifest
    output: InputManifest
    construction: Literal[CONSTRUCTION] = CONSTRUCTION
    conversion_error: FramedFact
    raw_fermion_rows: Count
    raw_jw_candidates: Count
    pauli_terms: Count
    items: Count
    payload_bytes: Count
    work: Count

    @model_validator(mode="after")
    def _association(self):
        """Bind a loaded or new receipt to one conversion without replaying it.

        The output must be the ingested Hermitian Pauli operator at width 2n,
        the raw populations must equal the counts of the contraction and
        mapping, the counters must reach the raw-table floor, and the
        conversion error must stay unknown in its full-operator frame.
        """
        n = self.input.orbital_basis.dimension
        # Check the width's bit length without creating another exponential integer.
        d = self.output.basis.dimension
        if (self.output.reference.representation != "pauli" or self.output.identity_status != "ingested"
                or d.bit_length() != 2*n+1 or d & (d-1)
                or self.output.basis.identity != "computational" or self.output.basis.ordering != ORDER
                or self.output.shape != (self.pauli_terms,) or "Hermitian" not in self.output.checked):
            raise ValueError("DF receipt requires the actual Hermitian Pauli output at the interleaved width")
        rows, mapped = _populations(n, bool(self.input.columns))
        if (self.raw_fermion_rows, self.raw_jw_candidates) != (rows, mapped) or self.pauli_terms > mapped:
            raise ValueError("DF receipt populations must preserve raw contraction/mapping counts")
        # Necessary raw-table floor, not the full mapper envelope or provenance
        # authentication. P counts stored ladder entries, even for zero
        # coefficients. Record admission uses scalar arithmetic only, no replay.
        q = 2*n
        w = (q+63)//64
        p = 4*n*n + (16*n**4 if self.input.columns else 0)
        minimum_items = max(rows+p, mapped)
        minimum_bytes = 24*rows + 9*p + 8 + mapped*(16*w+16)
        minimum_work = max(q, rows+p) + max(q, mapped*(w+1))
        if (self.items < minimum_items or self.payload_bytes < minimum_bytes or self.work < minimum_work):
            raise ValueError("DF receipt counters are below the declared raw-table lower bound")
        scope = _scope(self.input, self.output)
        if (self.conversion_error.frame != _error_frame(self.input, self.output, scope)
                or self.conversion_error.bindings or self.conversion_error.fact.quantity != "conversion_error"
                or self.conversion_error.fact.availability != "unknown"):
            raise ValueError("conversion error must keep this input/output full-operator unknown frame")
        return self


@dataclass(frozen=True, slots=True)
class DFConversion:
    """Live output and portable provenance, with exact associations.

    Attributes:
        operator: Actual admitted Pauli target, reusable without conversion.
        receipt: Portable factor/output association and acquisition population.
        work_fact: Source-qualified planning_work for Candidate prior_work.
    """

    operator: OperatorInput
    receipt: DFConversionReceipt
    work_fact: Fact

    def __post_init__(self):
        """Require the operator, receipt and work fact to name one conversion.

        A work fact from another conversion, or an operator from another
        receipt, would attach cost or provenance to the wrong Pauli input.
        """
        if not isinstance(self.operator, OperatorInput) or self.operator.manifest != self.receipt.output:
            raise ValueError("DF conversion operator belongs to another receipt output")
        fact, receipt = self.work_fact, self.receipt
        scope = _scope(receipt.input, receipt.output)
        evidence = fact.evidence
        if (fact.quantity != "planning_work" or fact.availability != "concrete"
                or fact.unit != Unit(symbol="work_unit", dimension="count") or fact.scope != scope
                or not isinstance(fact.value, Rational) or (fact.value.numerator, fact.value.denominator) != (receipt.work, 1)
                or evidence is None or evidence.source != SOURCE or evidence.kind != "observed"
                or evidence.status != "witnessed" or evidence.artifact != receipt.content_id
                or evidence.subject_id != receipt.output.reference.content_id or evidence.witnessed_scope != scope
                or len(evidence.work) != 1 or evidence.work[0].artifact != receipt.content_id
                or evidence.work[0].stage != "planning"):
            raise ValueError("DF conversion work must name this acquisition, input/output and receipt")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FactorizedHamiltonian:
    """Factory-owned immutable real factor polynomial; no automatic materialization.

    Attributes:
        manifest: Detached source, native snapshot identity and scientific frame.
        _one_body: Immutable shifted T; never serialized.
        _factors: Immutable ordered V/w snapshots; never serialized.
    """

    manifest: DFManifest
    _one_body: np.ndarray = field(repr=False)
    _factors: tuple = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("factor Hamiltonians require ingest_df; independent field replacement is unsupported")

    def to_pauli(self, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000) -> DFConversion:
        """Explicit quartic materialization, with no decomposition or reference run."""
        n, columns = self.manifest.orbital_basis.dimension, self.manifest.columns
        r, k = len(columns), sum(columns)
        rows, mapped = _populations(n, bool(r))
        # Shape-only data bound before enumerating even the mapper's lengths:
        # 24 bytes per raw row, the per-string term of fermion_table's
        # 24M + 9P + 8 law (an int64 offset and a complex128 coefficient).
        _check_bytes(rows * 24, max_bytes, "DF raw operator rows")
        lengths = chain((0,), repeat(2, 2*n*n), repeat(4, 4*n**4 if r else 0))
        mapping = mapping_requirements(lengths, num_modes=2*n, max_bytes=max_bytes)
        pairs = n*(n+1)//2
        # Cumulative numerical payload: weighted V, reconstructed/mirrored B,
        # B-square and accumulation, pair table and full pair-Gram/symmetry
        # copies, and finite/comparison masks. 16 bytes/entry covers the float
        # value and its bounded masks; four n² layouts per factor cover both
        # contractions and triangular writes. No spin-orbital Q^4 tensor.
        entries = n*k + 4*r*n*n + r*pairs + 2*pairs*pairs*(r > 0) + 4*n*n
        # Work: twice the scalar products of V diag(w) V.T, B @ B and the
        # pair Gram (the three terms of the product law below), 8 per stored
        # entry for scans and copies, and one per raw row.
        numeric_work = 2*n*n*k + 2*r*n**3 + 2*r*pairs*pairs + 8*entries + rows
        sizes = dict(items=rows + entries + mapping[0], payload_bytes=16*entries + mapping[1],
                     work=rows + numeric_work + mapping[2])
        _check_bytes(sizes["payload_bytes"], max_bytes, "DF contraction and JW conversion")
        # Matrix multiplication scalar products; vendor CPU/BLAS workspace is
        # unknown. The finite cap is per explicit conversion, never metadata.
        # n*n*k: (V*w) @ V.T over all k columns. r*n**3: B @ B per factor.
        # r*pairs*pairs: the pair Gram pair_values.T @ pair_values.
        _check_products(n*n*k + r*n**3 + r*pairs*pairs, max_products)
        one_body, gram, pair_index = _contract(self._one_body, self._factors)
        raw = fermion_table(_rows(one_body, gram, pair_index, self.manifest.constant_energy.value),
                            num_modes=2*n, max_bytes=max_bytes)
        operator = raw.to_pauli(mapping="jw", max_bytes=max_bytes)
        if operator.structure != "hermitian":
            raise ArithmeticError("shared Hermitian coefficients failed exact JW imaginary cancellation")
        return _publish(self.manifest, operator, rows, mapped, sizes)


def ingest_df(shifted_one_body, factors, *, constant_energy, orbital_basis: Basis,
              energy_unit: Unit, source: InputRef, max_bytes=DEFAULT_INPUT_BYTES) -> FactorizedHamiltonian:
    """Snapshot finite native real T/V/w, without contraction or chemistry access.

    T must be exactly symmetric. V columns need not be orthogonal; signed and
    zero w are lawful. The empty factor tuple defines E0+dGamma(T). All array
    scans, snapshots and hashes are admitted cumulatively before reading data.

    Here ``dGamma(A) = sum_{pq,sigma} A_pq a_{p sigma}^dagger a_{q sigma}``.
    T is the one-body coefficient of this particular polynomial. For an
    electronic Hamiltonian with one-body integrals h and chemists'-notation
    two-body integrals ``g_pqrs = sum_l B_l[p,q] B_l[r,s]``, rewriting the
    two-body term as ``sum_l dGamma(B_l)**2/2`` shifts the one-body
    coefficient to ``T_ps = h_ps - (1/2) sum_q g_pqqs``. The caller supplies
    that shifted T. No integrals are read here.
    """
    if not isinstance(orbital_basis, Basis) or not isinstance(source, InputRef):
        raise TypeError("DF requires an explicit orbital Basis and supplied source InputRef")
    if not isinstance(energy_unit, Unit) or energy_unit.dimension != "energy":
        raise ValueError("DF requires a declared energy unit")
    if type(factors) is not tuple:
        raise TypeError("DF factors must be an explicit finite tuple")
    constant = finite_real(constant_energy, "constant_energy")
    n = orbital_basis.dimension
    _check_bytes(8*(len(factors)+1), max_bytes, "DF factor references")
    if type(shifted_one_body) is not np.ndarray or shifted_one_body.shape != (n, n):
        raise ValueError("shifted_one_body must be a native square array matching orbital_basis")
    arrays, columns = [shifted_one_body], []
    for factor in factors:
        if type(factor) is not tuple or len(factor) != 2:
            raise TypeError("each factor must be a (vectors, weights) tuple")
        vectors, weights = factor
        if (type(vectors) is not np.ndarray or type(weights) is not np.ndarray
                or vectors.ndim != 2 or vectors.shape[0] != n or vectors.shape[1] < 1
                or weights.shape != (vectors.shape[1],)):
            raise ValueError("DF factor requires vectors[n,r] and weights[r] with r>0")
        columns.append(vectors.shape[1])
        arrays.extend((vectors, weights))
    size = sum(a.size for a in arrays)
    # Snapshot bytes + finite/symmetry masks; hashing borrows immutable memory.
    # 10 bytes per entry: the float64 snapshot and one byte for each mask.
    _check_bytes(10*size, max_bytes, "DF factor snapshot")
    if any(a.dtype.kind not in "iuf" for a in arrays):
        raise TypeError("DF arrays must contain real numerical data")
    frozen = tuple(_freeze_array(_array_input(a, ndim=a.ndim, max_bytes=max_bytes)) for a in arrays)
    if not all(np.isfinite(a).all() for a in frozen):
        raise ValueError("DF arrays must contain finite values")
    if not np.array_equal(frozen[0], frozen[0].T):
        raise ValueError("shifted_one_body must be exactly symmetric; no repair is performed")
    constant_record = Float64(value=constant)
    manifest = DFManifest(
        payload_digest=_digest("supplied_real_df.T.ordered_V_w", (n, *columns), frozen),
        source=source, orbital_basis=orbital_basis, energy_unit=energy_unit,
        constant_energy=constant_record, columns=tuple(columns), byte_order=sys.byteorder,
        payload_bytes=8*size)
    result = object.__new__(FactorizedHamiltonian)
    object.__setattr__(result, "manifest", manifest)
    object.__setattr__(result, "_one_body", frozen[0])
    object.__setattr__(result, "_factors", tuple(zip(frozen[1::2], frozen[2::2], strict=True)))
    return result


def _populations(n, has_factors):
    """Return raw Fermion rows and raw JW rows emitted by ``_rows``.

    Rows: one constant, ``2*n**2`` spin-diagonal one-body strings and
    ``4*n**4`` two-body strings (four spin pairs). A string of p ladder
    operators maps to ``2**p`` Pauli rows before coalescing.
    """
    quartic = 4*n**4 if has_factors else 0
    return 1+2*n*n+quartic, 1+8*n*n+16*quartic


def _symmetric_upper(matrix):
    """Use one stored upper-triangle coefficient for both orientations."""
    for p in range(len(matrix)):
        matrix[p+1:, p] = matrix[p, p+1:]
    return matrix


def _contract(shifted_one_body, factors):
    """Normal-order the squared factors into one-body and pair-Gram tables.

    With ``B = V diag(w) V.T`` and the anticommutator
    ``a_{q sigma} a_{r tau}^dagger = delta_qr delta_sigma,tau
    - a_{r tau}^dagger a_{q sigma}``,

        dGamma(B)**2 / 2 = dGamma(B @ B) / 2
            + (1/2) sum B_pq B_rs a_{p sigma}^dagger a_{r tau}^dagger
              a_{s tau} a_{q sigma}.

    The first term is added to T. The second is summed over factors as
    ``gram[pq, rs] = sum_l B_l[p,q] B_l[r,s]`` over unordered spatial pairs.
    This is standard anticommutator algebra. Each stored upper-triangle
    value serves both orientations, so the one-body table and gram are
    exactly symmetric in binary64. Then every JW coefficient pairs with an
    exactly conjugate partner, which the Hermitian check in ``to_pauli``
    relies on.
    """
    n = len(shifted_one_body)
    one_body = shifted_one_body.copy()
    # Unique unordered spatial pairs preserve exact stored eightfold g symmetry.
    pair_index = np.empty((n, n), dtype=np.int64)
    pairs = tuple((p, q) for p in range(n) for q in range(p, n))
    for index, (p, q) in enumerate(pairs):
        pair_index[p, q] = pair_index[q, p] = index
    pair_values = np.empty((len(factors), len(pairs)), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for index, (vectors, weights) in enumerate(factors):
            b = _symmetric_upper((vectors*weights) @ vectors.T)
            correction = _symmetric_upper(b @ b)
            one_body += 0.5*correction
            for pair, (p, q) in enumerate(pairs):
                pair_values[index, pair] = b[p, q]
        gram = _symmetric_upper(pair_values.T @ pair_values) if factors else None
    if not np.isfinite(one_body).all() or (gram is not None and not np.isfinite(gram).all()):
        raise ValueError("DF contraction produced nonfinite coefficients")
    return one_body, gram, pair_index


def _rows(one_body, gram, pair_index, constant):
    """Yield ordered Fermion rows of the normal-ordered polynomial.

    Spin orbital ``(p, sigma)`` is mode ``2*p + sigma`` (alpha and beta
    interleaved). Two-body rows use the order
    ``a_{p sigma}^dagger a_{r tau}^dagger a_{s tau} a_{q sigma}`` with
    coefficient ``gram[pq, rs]/2``.
    """
    n = len(one_body)
    yield constant, ()
    for p in range(n):
        for q in range(n):
            for spin in range(2):
                yield one_body[p, q], ((2*p+spin, 1), (2*q+spin, 0))
    if gram is None:
        return
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    coefficient = 0.5*gram[pair_index[p, q], pair_index[r, s]]
                    for sigma in range(2):
                        for tau in range(2):
                            yield coefficient, ((2*p+sigma, 1), (2*r+tau, 1),
                                                (2*s+tau, 0), (2*q+sigma, 0))


def _publish(manifest, operator, rows, mapped, sizes):
    """Bind the Pauli output to its factor input with a receipt and work fact.

    The conversion error stays unknown in the full-operator frame because no
    independently justified enclosure of the binary64 contraction error
    exists. The work fact is concrete and names this receipt.
    """
    scope = _scope(manifest, operator.manifest)
    frame = _error_frame(manifest, operator.manifest, scope)
    unavailable = Fact(quantity="conversion_error", unit=manifest.energy_unit,
        scope=scope, availability="unknown", reason="no independently justified enclosure of binary64 conversion error")
    error = FramedFact(frame=frame, bindings=(), fact=unavailable)
    receipt = DFConversionReceipt(input=manifest, output=operator.manifest,
        conversion_error=error, raw_fermion_rows=rows, raw_jw_candidates=mapped,
        pauli_terms=len(operator.pauli_terms()), **sizes)
    work = WorkProvenance(stage="planning", artifact=receipt.content_id,
        description=f"DF conversion: {receipt.work} admitted scalar/mask size-law units and "
        f"{receipt.payload_bytes} cumulative logical bytes before publication; supplied-factor acquisition unknown")
    evidence = Evidence(kind="observed", source=SOURCE, status="witnessed",
        artifact=receipt.content_id, subject_id=operator.manifest.reference.content_id,
        witnessed_scope=scope, work=(work,))
    value = Rational(numerator=receipt.work, denominator=1)
    work_unit = Unit(symbol="work_unit", dimension="count")
    fact = Fact(quantity="planning_work", unit=work_unit,
        scope=scope, availability="concrete", value=value, evidence=evidence)
    return DFConversion(operator, receipt, fact)
