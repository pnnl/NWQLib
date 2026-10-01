"""Explicit bounded ingestion and classical access, with no quantum oracle."""

from collections.abc import Sized
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Complex
import sys

import numpy as np
from scipy import sparse

from nwqlib._validation import finite_real
from nwqlib.core.records import Basis, InputRef, Source
from .access import DEFAULT_INPUT_BYTES, InputManifest, _check_bytes, _check_products
from ._pauli import PauliTerms, apply_terms, combine_terms, pauli_action_requirements


# Little-endian convention shared with Qiskit: coordinate index
# k = sum_q b_q 2**q, so qubit 0 is the least significant bit and the
# rightmost tensor factor. Every admitted operator and state uses it.
ORDER = "coordinate index increasing; qubit 0 is rightmost tensor/bit position"


def _basis(dimension):
    return Basis(identity="computational", dimension=dimension, ordering=ORDER)


def _freeze_array(array):
    """One immutable snapshot; even setflags cannot make its bytes writable."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _digest(kind, shape, arrays=()):
    """Hash native content once at admission, including dtype/shape/order.

    Equal numbers stored with another dtype, shape or ordering produce a
    different identity, so a Plan or saved result bound to one input cannot
    be silently associated with a differently represented one.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps([kind, shape, ORDER], separators=(",", ":")).encode())
    for array in arrays:
        digest.update(json.dumps([array.dtype.str, array.shape, "C"]).encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def _pauli_identity(table):
    """Bind width, word order, and ordered native x/z/coefficient bytes."""
    return _digest("pauli.words.q0-first.x-z", (table.num_qubits, table.x.shape[1]),
                   (table.x, table.z, table.coefficients))


def _manifest(kind, dimension, shape, dtype, payload_bytes, digest, access, checked, law):
    source = Source(name="nwqlib bounded ingestion", version="1", domain=kind,
                    reference="sha256 of admitted native dtype/shape/order/content")
    return InputManifest(
        reference=InputRef(identity=digest, representation=kind, source=source),
        basis=_basis(dimension), identity_status="ingested", dtype=dtype, shape=shape,
        byte_order=("not_applicable" if dtype == "uint8" else sys.byteorder),
        payload_bytes=payload_bytes, access=access, checked=checked, work_law=law,
    )


def _array_input(value, *, ndim, max_bytes, extra_bytes_per_item=0):
    """Convert only known numerical containers, after their output-size check.

    Integers and real data become float64; complex data become complex128.
    No unknown array protocol, object conversion or unsized stream is called.
    The caller adds its known per-entry storage, such as a normalized direction.
    """
    if type(value) is np.ndarray:
        if value.ndim != ndim:
            raise ValueError(f"expected {ndim}-dimensional numerical data")
        if value.dtype.kind not in "iufc":
            raise TypeError("numerical data must contain real or complex numbers, not bool/object/string values")
        dtype = np.dtype("complex128" if value.dtype.kind == "c" else "float64")
        _check_bytes(value.size * (dtype.itemsize + extra_bytes_per_item), max_bytes)
    elif type(value) in (list, tuple):
        rows = (value,) if ndim == 1 else value
        _check_bytes(len(value) * (8 + extra_bytes_per_item), max_bytes)
        width = None
        size = 0
        complex_data = False
        for row in rows:
            if type(row) not in (list, tuple):
                raise ValueError(f"expected rectangular {ndim}-dimensional numerical data")
            if width is not None and len(row) != width:
                raise ValueError("numerical data must be rectangular; ragged rows are unsupported")
            width = len(row)
            size += width
            _check_bytes(size * (8 + extra_bytes_per_item), max_bytes)
            for item in row:
                if type(item) not in (int, float, complex) and not (
                    isinstance(item, np.number) and item.dtype.kind in "iufc"
                ):
                    raise TypeError("numerical data must contain real or complex numbers, not bool/object/string values")
                complex_data |= isinstance(item, (complex, np.complexfloating))
        dtype = np.dtype("complex128" if complex_data else "float64")
        _check_bytes(size * (dtype.itemsize + extra_bytes_per_item), max_bytes)
    else:
        raise TypeError("numerical input requires an ordinary ndarray, list or tuple")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.asarray(value, dtype=dtype)
    except (OverflowError, FloatingPointError, ValueError) as error:
        raise ValueError("numerical input is not representable as finite float64/complex128 data") from error
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim}-dimensional numerical data")
    return array


@dataclass(frozen=True, slots=True)
class PeriodicStencil:
    """Exact family parameters for mass I + diffusion (2I-S-S†) + i potential Z0.

    Attributes:
        num_qubits: Positive q defining a periodic grid of 2**q points.
        mass: Nonnegative real coefficient of the identity.
        diffusion: Nonnegative real coefficient multiplying 2I-S-S† for cyclic shift S.
        potential: Real coefficient of iZ0; zero gives a Hermitian mass/diffusion operator.
    """

    num_qubits: int
    mass: float
    diffusion: float
    potential: float = 0.0

    def __post_init__(self):
        if type(self.num_qubits) is not int or self.num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer")
        for name in ("mass", "diffusion", "potential"):
            value = finite_real(getattr(self, name), name)
            if name != "potential" and value < 0:
                raise ValueError("periodic stencil mass and diffusion must be nonnegative")
            object.__setattr__(self, name, 0.0 if value == 0 else value)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class OperatorInput:
    """Factory-owned operator with correlated immutable metadata/native storage.

    Fields cannot be independently replaced. Exported record revisions remain
    standalone declarations and do not change this native owner.

    Attributes:
        manifest: Detached metadata and admitted input identity.
        structure: Checked Hermitian or general, or a declared external claim.
        _data: Private native storage; never serialized in a record.
    """

    manifest: InputManifest
    structure: str
    _data: object = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("native inputs require an ingestion/declaration factory; field replacement is unsupported")

    @classmethod
    def _from_admitted(cls, manifest, structure, data):
        instance = object.__new__(cls)
        object.__setattr__(instance, "manifest", manifest)
        object.__setattr__(instance, "structure", structure)
        object.__setattr__(instance, "_data", data)
        return instance

    @property
    def basis(self):
        return self.manifest.basis

    @property
    def reference(self):
        return self.manifest.reference

    def to_record(self):
        """Describe this handle without copying, hashing or storing native data."""
        return dict(format="nwqlib.operator_input/2",
                    manifest=self.manifest.model_dump(mode="json", exclude_computed_fields=True),
                    structure=self.structure)

    @classmethod
    def from_record(cls, record):
        """Read inert metadata; executable access requires a separate saved binding."""
        if (type(record) is not dict or set(record) != {"format", "manifest", "structure"}
                or record["format"] != "nwqlib.operator_input/2"):
            raise ValueError("unsupported operator input record")
        if record["structure"] not in ("hermitian", "unitary", "general"):
            raise ValueError("unsupported operator structure declaration")
        return cls._from_admitted(InputManifest.model_validate(record["manifest"]), record["structure"], None)

    def _require_data(self):
        if self._data is None:
            raise ValueError("operator native data is unavailable; a descriptive record does not restore data access")

    def pauli_terms(self):
        """Return immutable x/z words and coefficients without expansion."""
        self._require_data()
        if "pauli_terms" not in self.manifest.access:
            raise ValueError("structured Pauli access is unavailable")
        return self._data

    def dense_array(self):
        """Access the immutable admitted dense snapshot without conversion or copy."""
        self._require_data()
        if self.manifest.reference.representation != "dense" or "entries" not in self.manifest.access:
            raise ValueError("dense access requires an ingested dense operator; no implicit densification")
        return self._data

    def fermion_terms(self):
        """Return immutable ordered ladder arrays, without mapping or action."""
        self._require_data()
        if "fermion_terms" not in self.manifest.access:
            raise ValueError("ordered Fermion access is unavailable")
        return self._data

    def periodic_stencil(self):
        """Read the admitted compact family parameters; no classical action."""
        self._require_data()
        if not isinstance(self._data, PeriodicStencil):
            raise ValueError("periodic stencil parameters require their native input factory")
        return self._data

    def entry(self, row: int, column: int):
        """Read one existing dense/sparse entry without dense conversion."""
        self._require_data()
        if "entries" not in self.manifest.access:
            raise ValueError("entry access is unavailable")
        dimension = self.manifest.basis.dimension
        if any(type(i) is not int or not 0 <= i < dimension for i in (row, column)):
            raise ValueError("entry indices must lie in the operator dimension")
        return complex(self._data[row, column])

    def sparse_entries(self, *, max_bytes=DEFAULT_INPUT_BYTES):
        """Borrow immutable (data, indices, indptr), preserving CSR/CSC orientation.

        Native storage bytes are checked before exposing any arrays. No
        conversion, traversal, copy or mutable sparse container is returned.
        """
        self._require_data()
        if self.manifest.reference.representation not in {"csr", "csc"} or "entries" not in self.manifest.access:
            raise ValueError("compressed entries require an ingested CSR/CSC operator")
        _check_bytes(self.manifest.payload_bytes, max_bytes, "sparse entries")
        return self._data.data, self._data.indices, self._data.indptr

    def matvec(self, vector: np.ndarray, *, max_bytes=DEFAULT_INPUT_BYTES,
               max_products=1_000_000_000, limit_name="max_products") -> np.ndarray:
        """Explicit classical action with vector/output bytes and scalar products.

        Work law is D² for dense, nnz+D for sparse, and for Pauli terms the
        grouped-action law of ``_matvec_requirements``, which includes the
        possible ordered retry. Live numerical storage is O(D) beyond the
        owned operator; the shared Pauli kernel additionally uses bounded
        coordinate tiles and O(M) grouping metadata. The default
        ``max_products`` is registered in ENGINEERING_CONSTANTS.md,
        "Explicit input operation defaults".
        ``limit_name`` names the caller's option that supplies
        ``max_products`` in a refusal.
        """
        _, payload_bytes, products = _matvec_requirements(self)
        _check_bytes(payload_bytes, max_bytes, "matrix-vector action")
        _check_products(products, max_products, limit_name)
        dimension = self.manifest.basis.dimension
        if type(vector) is not np.ndarray or vector.shape != (dimension,):
            raise ValueError("matvec requires a native vector matching operator dimension")
        if vector.dtype not in (np.dtype("float64"), np.dtype("complex128")) or not np.isfinite(vector).all():
            raise ValueError("matvec vector must have finite float64/complex128 entries")
        if self.manifest.reference.representation == "pauli":
            return apply_terms(self._data, vector, num_qubits=dimension.bit_length() - 1)
        return self._data @ vector

def _matvec_requirements(operator):
    """Return shared ``(D, bytes, products)`` admission allowances.

    Capability is checked before vector access. Dense and CSR/CSC
    storage reserve 33D bytes for an input of at most complex128 size,
    a complex128 output and a one-byte finite mask. Work is D**2 for
    dense storage and nnz+D for CSR/CSC. These are logical envelopes,
    excluding the owned operator payload and allocator or RSS overhead.

    For M stored Pauli terms use g = min(M, D), an upper bound on
    distinct single-word flip masks obtained from counts alone. Call
    ``pauli_action_requirements(D, M, g)`` for one column with possible
    input conversion. With t = min(D, _PAULI_TILE), bytes are
    ``49*D + 24*M + 16*g + 8 + max(9*M, 96*t)`` and work is
    ``(M+g) + 3*M + D*(M+g+1) + D + M*D``. The shared owner includes
    the pre-tile Y-count frontier, grouped action and possible ordered
    retry. Admission inspects metadata without a flip-mask census.
    """
    operator._require_data()
    if "matvec" not in operator.manifest.access:
        raise ValueError("classical matvec is unavailable for this handle")
    d = operator.manifest.basis.dimension
    representation = operator.manifest.reference.representation
    if representation == "pauli":
        if (d.bit_length() - 1 + 63) // 64 != 1:
            raise ValueError("full-state action requires a single-word Pauli table")
        terms = operator.manifest.shape[0]
        size, work = pauli_action_requirements(d, terms, min(terms, d))
        return d, size, work
    work = operator._data.nnz + d if representation in ("csr", "csc") else d * d
    return d, 33 * d, work


def _scaled_observable_requirements(operator):
    """Return allowances for one storage-preserving scaled vector action.

    Let N be the state dimension, P the original operator payload bytes,
    and K the stored-entry count (Pauli terms, sparse nnz or dense N**2).
    For ``(_, B_action, W_action) = _matvec_requirements(operator)``,
    return ``B_scaled = B_action + 32*N + 3*P + 48*K`` and
    ``W_scaled = W_action + 2*N + 18*K``.

    Payload copies cover compressed indices and the immutable Pauli
    snapshot. Entry scratch covers component maxima and lost-component
    masks. The vector reserve includes a separate framed input and the
    final dot product. Work covers up to four maximum/reduction visits,
    two ldexp visits, eight mask visits, one initial copy and three
    immutable Pauli-array copies per entry.

    For Pauli storage K=L and the shared action owner uses
    g=min(L,N), possible input conversion and one column. It already
    includes the 9L pre-tile frontier and 3L preprocessing work.
    The coefficients are scaled once per operator, so the 3P+48L
    storage and 18L work do not depend on the number of state columns.
    This helper itself admits one vector action.
    """
    d, action_bytes, action_work = _matvec_requirements(operator)
    count = (
        len(operator.pauli_terms()) if "pauli_terms" in operator.manifest.access
        else operator._data.nnz if operator.reference.representation in ("csr", "csc")
        else d * d
    )
    return (
        action_bytes + 32 * d + 3 * operator.manifest.payload_bytes + 48 * count,
        action_work + 2 * d + 18 * count,
    )


def _observable_exponent(operator):
    """Inspect existing storage; a binary scale does not change coordinates."""
    from math import frexp

    values = (operator.pauli_terms().coefficients if "pauli_terms" in operator.manifest.access
              else operator._data.data if operator.reference.representation in ("csr", "csc")
              else operator.dense_array())
    maximum = max(float(np.max(np.abs(values.real), initial=0)),
                  float(np.max(np.abs(values.imag), initial=0)))
    return frexp(maximum)[1]


def _scaled_observable(operator, exponent):
    """Copy O/2**exponent in its native storage; report lost nonzero entries."""
    pauli = "pauli_terms" in operator.manifest.access
    original = operator.pauli_terms().coefficients if pauli else operator._data
    scaled = original.copy()
    compressed = operator.reference.representation in ("csr", "csc")
    before, values = (original.data, scaled.data) if compressed else (original, scaled)
    for component in ("real", "imag") if np.iscomplexobj(values) else ("real",):
        part = getattr(values, component)
        np.ldexp(part, -exponent, out=part)
        if np.any((part == 0) & (getattr(before, component) != 0)):
            return None
    if pauli:
        table = operator.pauli_terms()
        return PauliTerms._snapshot(table.num_qubits, table.x, table.z, scaled)
    return scaled


def _scaled_observable_moment(operator, vector):
    """Return (v†(O/2**e)v, e) using one native action, without recovery.

    When a common scale would erase a stored component, choose the original
    action before applying anything. This preserves its supported heterogeneous
    range without a second action or a silently pruned observable.
    """
    exponent = _observable_exponent(operator)
    scaled = _scaled_observable(operator, exponent)
    if scaled is None:
        scaled, exponent = operator._data, 0
    with np.errstate(over="ignore", invalid="ignore"):
        action = (apply_terms(scaled, vector, num_qubits=scaled.num_qubits)
                  if "pauli_terms" in operator.manifest.access else scaled @ vector)
        value = float(np.vdot(vector, action).real)
    return (value if np.isfinite(value) else None), exponent


def ingest_periodic_stencil(num_qubits: int, mass, diffusion, potential=0.0, *, max_bytes=DEFAULT_INPUT_BYTES) -> OperatorInput:
    """Admit mass I + diffusion (2I-S-S†) + i potential Z0, S|j>=|j+1 mod 2**q>.

    q is an exact positive integer, mass/diffusion are nonnegative finite real
    scalars, and potential is real and finite. q=1 keeps both wrap edges.
    The payload is three float64 parameters and q's integer magnitude. Before
    forming dimension metadata, admission covers parameters plus their hash
    array (48 bytes), q's magnitude, and two q-bit dimension integer slots.
    Work is q+4 and items are four parameters. These are logical size laws,
    not Python object/RSS accounting. No matrix, Pauli expansion, or vector
    access is implied. Spectral endpoints of the Hermitian potential=0 family
    are mass and mass+4*diffusion; their numerical use belongs to a consumer.
    """
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    integer_bytes = max(1, (num_qubits.bit_length() + 7) // 8)
    dimension_bytes = (num_qubits + 8) // 8  # 2**q has q+1 bits.
    _check_bytes(48 + integer_bytes + 2 * dimension_bytes, max_bytes, "periodic parameters")
    parameters = PeriodicStencil(num_qubits, mass, diffusion, potential)
    mass, diffusion, potential = parameters.mass, parameters.diffusion, parameters.potential
    identity = _digest("periodic_stencil", (num_qubits,),
                       (np.array((mass, diffusion, potential), dtype=np.float64),))
    structure = "hermitian" if potential == 0 else "general"
    manifest = _manifest("periodic_stencil", 1 << num_qubits, (4,), "integer+3*float64",
        24 + integer_bytes, identity, (),
        ("finite parameters", "mass/diffusion nonnegative", "analytic periodic Laplacian PSD",
         "Hermitian" if structure == "hermitian" else "general"),
        "four parameters; O(q) dimension metadata; no entries, matvec or Pauli expansion")
    return OperatorInput._from_admitted(manifest, structure, parameters)


def _pauli_width(num_qubits):
    if type(num_qubits) is not int or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    return (num_qubits + 63) // 64


def _pauli_identity_bytes(num_qubits):
    """Bytes per Pauli term for the identity JSON of the Plan records that hold the term.

    A Method that plans with an admitted Pauli operator copies its terms
    into records, and ``Record.content_id`` then holds two JSON copies (the
    text and its UTF-8 bytes) beside the records' own data. The widest
    per-term layout, in an Expectation Plan with a provider estimate, is a
    ``PauliCoefficient`` record, laid out like ``PauliTerm`` as
    ``{"coefficient":c,"label":"...","parent_id":null,"schema_version":1},``,
    q + 88 bytes with a binary64 repr of at most 24 characters, plus the
    label (q + 3 bytes) and the coefficient (25 bytes) again in its readout
    spec. With 4 bytes for the brackets of those lists, J = 2q + 120 bytes
    per copy. J also bounds QPE's kept label and float64 coefficient (one
    string in a label tuple and eight bytes in one base64 array), an ADAPT pool term
    ``["...",c]`` with an imaginary ``Complex128`` (q + 106), an ADAPT
    Hamiltonian or commutator term with its readout group label (at most
    2q + 94) and the Lanczos label, coefficient and readout row (at most
    q + 55 + digits(2**q - 1)). The records' own label and coefficient data
    are at most one copy, so the share is 3J = 6q + 360 bytes per term.
    Plans whose Program also grows with the terms (Expectation with sampled
    counts, FixedGCIM, LCHS, and QLS with a ``QuadraticForm`` or
    ``NormalizedExpectation`` output) are capped earlier by
    ``ir.expressions.AdmissionLimits``. ADAPT's commutators are products
    that ingestion does not see, so ADAPT charges its reconstruction as a
    running total in ``gcim/adapt_inputs.py``.
    """
    return 3 * (2 * num_qubits + 120)


def _pauli_label_law(num_qubits):
    """Return W, native bytes/term, and conservative label bytes/work per term.

    ``pauli_table`` admits labels with these byte and work laws. Raw
    conversion without coalescing can reserve 3L separately. These are
    logical payload envelopes, not allocator-byte or RSS measurements.

    Returns ``(W, L, q + 4L + S, q + W + 1)``. ``W = ceil(q/64)`` uint64 words
    hold one mask. ``L = 16W + 16`` bytes store one term: its x and z masks
    and one complex128 coefficient. A label admission charges its q
    characters, the four L-byte copies alive at the peak of
    ``_coalesced_input`` and the share S = ``_pauli_identity_bytes(q)`` for
    the identity JSON of the Plan records that will hold the term, and
    ``q + W + 1`` work units for the characters, mask words and coefficient.
    """
    w = _pauli_width(num_qubits)
    layout = 16 * w + 16
    return w, layout, num_qubits + 4 * layout + _pauli_identity_bytes(num_qubits), num_qubits + w + 1


def _pauli_label_limit(num_qubits, max_bytes):
    """Admit width and return the raw limit before any label traversal."""
    _, _, payload_bytes, _ = _pauli_label_law(num_qubits)
    _check_bytes((num_qubits + 8) // 8, max_bytes, "Pauli width metadata")
    return max_bytes // payload_bytes


def pauli_table(terms, *, num_qubits: int, max_bytes=DEFAULT_INPUT_BYTES) -> PauliTerms:
    """Admit an immutable raw label table, preserving duplicate/zero indices.

    Raw layout L=16W+16 bytes/term, W=ceil(q/64). Admission reserves
    R*(q+4L+S) logical bytes. The q label bytes and the four L-byte copies
    alive at the peak of ``_coalesced_input`` cover conversion and the
    subsequent stable coalescing, and S is the identity JSON share of
    ``_pauli_identity_bytes``. Python object overhead is not RSS accounting.
    Work size law is R*(q+W+1). Unsized streams keep at most the raw limit
    plus one lookahead.

    Labels follow Qiskit order: the rightmost character acts on qubit 0.
    """
    w = _pauli_width(num_qubits)
    raw_limit = _pauli_label_limit(num_qubits, max_bytes)
    if isinstance(terms, Sized) and len(terms) > raw_limit:
        raise ValueError("raw Pauli input exceeds max_bytes")
    xs, zs, coefficients = [], [], []
    for index, item in enumerate(terms):
        if index >= raw_limit:
            raise ValueError("raw Pauli input exceeds max_bytes")
        label, coefficient = item
        if type(label) is not str or len(label) != num_qubits or any(c not in "IXYZ" for c in label):
            raise ValueError("Pauli labels require exactly num_qubits I/X/Y/Z characters")
        if isinstance(coefficient, (bool, np.bool_)) or not isinstance(coefficient, Complex):
            raise TypeError("Pauli coefficients must be numerical scalars")
        value = complex(coefficient)
        if not np.isfinite(value):
            raise ValueError("Pauli coefficients must be finite")
        x, z = [0] * w, [0] * w
        for bit, factor in enumerate(reversed(label)):
            if factor in "XY":
                x[bit // 64] |= 1 << (bit % 64)
            if factor in "YZ":
                z[bit // 64] |= 1 << (bit % 64)
        xs.append(x)
        zs.append(z)
        coefficients.append(value)
    return PauliTerms._snapshot(num_qubits, np.array(xs, dtype=np.uint64).reshape(-1, w),
                                np.array(zs, dtype=np.uint64).reshape(-1, w),
                                np.array(coefficients, dtype=np.complex128))


def _coalesced_input(table):
    """Admit a Pauli table as an operator after exact-label coalescing.

    Identical words are summed with stable summation and only an exact-zero
    sum is removed, so no nonzero coefficient is pruned. Canonical Pauli
    words are Hermitian and linearly independent, so the sum is Hermitian
    exactly when every coalesced coefficient is real. That exact test sets
    ``structure`` without a tolerance.

    With L = 16W + 16 bytes per term, the peak is four L-byte copies alive
    together when the snapshot is taken: the input table, the coalesced
    (masks, value) pairs, the coalesced arrays and their snapshot. Callers
    charge this as 4L per term.
    """
    q, w = table.num_qubits, table.x.shape[1]
    combined = combine_terms(((x.tobytes(), z.tobytes()), value)
                             for x, z, value in zip(table.x, table.z, table.coefficients, strict=True))
    x, z = np.empty((len(combined), w), dtype=np.uint64), np.empty((len(combined), w), dtype=np.uint64)
    coefficients = np.empty(len(combined), dtype=np.complex128)
    for i, ((xb, zb), value) in enumerate(combined):
        x[i], z[i], coefficients[i] = np.frombuffer(xb, dtype=np.uint64), np.frombuffer(zb, dtype=np.uint64), value
    # Test the coefficients before the snapshot copies them byte for byte, so
    # the one-byte-per-term comparison mask is freed before the fourth copy
    # exists.
    hermitian = bool(np.all(coefficients.imag == 0))
    data = PauliTerms._snapshot(q, x, z, coefficients)
    manifest = _manifest("pauli", 1 << q, (len(data),), "uint64[x,z]+complex128",
                         len(data) * (16 * w + 16),
                         _pauli_identity(data),
                         ("matvec", "pauli_terms"), ("finite", "Hermitian" if hermitian else "general"),
                         "W=ceil(q/64); raw R*(16W+16) bytes; label scan O(Rq), mask scan O(RW); "
                         "stable coalescing O(RW) work/O(RW) workspace, snapshot copies admitted; action O(M*2**q)")
    return OperatorInput._from_admitted(manifest, "hermitian" if hermitian else "general", data)


def ingest_pauli(terms, *, num_qubits: int, max_bytes=DEFAULT_INPUT_BYTES) -> OperatorInput:
    """Snapshot bounded labels into native masks, stably coalescing exact zeros.

    Uses pauli_table's conversion/coalescing envelope. No coefficient pruning.
    """
    return _coalesced_input(pauli_table(terms, num_qubits=num_qubits, max_bytes=max_bytes))


def ingest_pauli_masks(x, z, coefficients, *, num_qubits: int, max_bytes=DEFAULT_INPUT_BYTES) -> OperatorInput:
    """Admit native uint64 (R,W) masks and complex128 (R,) coefficients.

    Reject invalid unused high bits and nonnative dtypes. Work size law
    max(q, R*(W+1)) admits width metadata even for zero terms. The logical
    byte envelope R*(4(16W+16)+S) beyond caller-owned arrays covers the four
    L-byte copies alive at the peak of ``_coalesced_input`` and the identity
    JSON share S of ``_pauli_identity_bytes``. The final content identity
    is shared with label ingestion of identical ordered terms.
    """
    w = _pauli_width(num_qubits)
    if any(type(a) is not np.ndarray for a in (x, z, coefficients)):
        raise TypeError("Pauli masks and coefficients must be native ndarrays")
    if x.ndim != 2 or x.shape[1] != w or z.shape != x.shape or coefficients.shape != (len(x),):
        raise ValueError("Pauli masks require (R,W) x/z and (R,) coefficients")
    # The snapshot of the caller's arrays and the three further L-byte copies
    # of _coalesced_input, L = 16W + 16 as in _pauli_label_law, the identity
    # JSON share of the records that will hold each term, and the dimension
    # integer 2**q of (q + 8) // 8 bytes.
    _check_bytes(len(x) * (4 * (16 * w + 16) + _pauli_identity_bytes(num_qubits)) + (num_qubits + 8) // 8,
                 max_bytes, "Pauli mask conversion")
    if x.dtype != np.dtype("uint64") or z.dtype != np.dtype("uint64") or coefficients.dtype != np.dtype("complex128"):
        raise TypeError("Pauli masks require native uint64 and complex128 dtypes")
    if num_qubits % 64 and np.any((x[:, -1] | z[:, -1]) >> np.uint64(num_qubits % 64)):
        raise ValueError("Pauli masks contain unused high bits")
    if not np.isfinite(coefficients).all():
        raise ValueError("Pauli coefficients must be finite")
    return _coalesced_input(PauliTerms._snapshot(num_qubits, x, z, coefficients))


def ingest_dense(matrix, *, max_bytes=DEFAULT_INPUT_BYTES) -> OperatorInput:
    """Snapshot a native small square array; no padding or symmetrization.

    Admission scans O(D²) entries, keeping one snapshot. Finite and exact
    stored Hermiticity checks use O(D²) temporary comparison storage, never
    eigensolves. max_bytes bounds the numerical snapshot size.
    """
    matrix = _array_input(matrix, ndim=2, max_bytes=max_bytes)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be nonempty and square")
    data = _freeze_array(matrix)
    if not np.isfinite(data).all():
        raise ValueError("operator entries must be finite")
    hermitian = np.array_equal(data, data.conj().T)
    manifest = _manifest("dense", data.shape[0], data.shape, str(data.dtype), data.nbytes,
                         _digest("dense", data.shape, (data,)), ("matvec", "entries"),
                         ("finite", "Hermitian" if hermitian else "general"), "O(D²) admission and action")
    return OperatorInput._from_admitted(manifest, "hermitian" if hermitian else "general", data)


def ingest_sparse(matrix, *, max_bytes=DEFAULT_INPUT_BYTES) -> OperatorInput:
    """Snapshot canonical CSR/CSC storage; sparse never becomes dense.

    Only native sorted, duplicate-free CSR/CSC arrays/matrices are supported.
    The caller explicitly canonicalizes other forms. Admission and Hermiticity
    comparison cost O(nnz+D), with O(nnz+D) temporary sparse storage.
    """
    if type(matrix) not in (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array):
        raise TypeError("sparse ingestion requires native canonical CSR or CSC")
    if matrix.ndim != 2:
        raise ValueError("operator must be a two-dimensional sparse array")
    if matrix.dtype.kind not in "iufc":
        raise TypeError("sparse data must contain real or complex numbers, not bool/object values")
    dtype = np.dtype("complex128" if matrix.dtype.kind == "c" else "float64")
    size = matrix.nnz * dtype.itemsize + matrix.indices.nbytes + matrix.indptr.nbytes
    _check_bytes(size, max_bytes, "sparse snapshot")
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be nonempty and square")
    native_values = _array_input(matrix.data, ndim=1, max_bytes=max_bytes)
    arrays = tuple(_freeze_array(a) for a in (native_values, matrix.indices, matrix.indptr))
    if not np.isfinite(arrays[0]).all():
        raise ValueError("operator entries must be finite")
    data = type(matrix)(arrays, shape=matrix.shape, copy=False)
    # SciPy may downcast small int64 indices into writable arrays despite
    # copy=False. Keep the actual admitted frozen storage and index dtype.
    data.data, data.indices, data.indptr = arrays
    # Inspect the snapshot, not potentially stale caller canonical-format flags.
    data.check_format(full_check=True)
    if not data.has_canonical_format:
        raise ValueError("sparse input must have canonical sorted duplicate-free storage")
    hermitian = (data != data.conj().T).nnz == 0
    manifest = _manifest(matrix.format, data.shape[0], data.shape, str(data.dtype), size,
                         _digest(matrix.format, data.shape, arrays), ("matvec", "entries"),
                         ("finite", "Hermitian" if hermitian else "general"), "O(nnz+D) admission and action")
    return OperatorInput._from_admitted(manifest, "hermitian" if hermitian else "general", data)


def operator_input(value, *, max_bytes=DEFAULT_INPUT_BYTES) -> OperatorInput:
    """Admit a known operator representation without implicit densification."""
    _check_bytes(0, max_bytes)
    if isinstance(value, OperatorInput):
        return value
    if type(value) in (np.ndarray, list, tuple):
        return ingest_dense(value, max_bytes=max_bytes)
    if type(value) in (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array):
        return ingest_sparse(value, max_bytes=max_bytes)
    if type(value) is PeriodicStencil:
        return ingest_periodic_stencil(value.num_qubits, value.mass, value.diffusion,
                                       value.potential, max_bytes=max_bytes)
    if (type(value).__module__ == "qiskit.quantum_info.operators.symplectic.sparse_pauli_op"
            and type(value).__name__ == "SparsePauliOp"):
        # Only this explicit SDK value loads its SDK owner. The public Pauli
        # phase is relative to canonical I/X/Y/Z labels, including Y itself.
        # Qiskit stores each word as (-i)**phase times its label, so the
        # coefficient of the canonical label is coeff * (-i)**phase.
        from qiskit.quantum_info import SparsePauliOp
        if type(value) is SparsePauliOp:
            q, count = value.num_qubits, len(value)
            words = _pauli_width(q)
            # With L = 16W + 16 bytes per term, the uint64 x and z masks and the
            # complex128 coefficients built here are one L-byte table. It stays
            # alive through ingest_pauli_masks, which adds its own four L-byte
            # copies and the (q + 8) // 8-byte dimension integer, so the peak is
            # 5L per term plus that integer. Earlier steps hold less. The mask
            # loop holds the masks and two 8-byte temporaries per term, and the
            # phase product holds the masks and at most q + 48 more bytes per
            # term, both below 5L because q <= 64W. The SDK's boolean rows are
            # caller-owned storage and are not charged. The law also includes the
            # identity JSON share that ingest_pauli_masks charges per term, so an
            # operator that the mask check would refuse is refused before the
            # masks are built.
            _check_bytes(count * (5 * (16 * words + 16) + _pauli_identity_bytes(q)) + (q + 8) // 8,
                         max_bytes, "SparsePauliOp conversion")
            x = np.zeros((count, words), dtype=np.uint64)
            z = np.zeros_like(x)
            for bit in range(q):
                x[:, bit // 64] |= value.paulis.x[:, bit].astype(np.uint64) << np.uint64(bit % 64)
                z[:, bit // 64] |= value.paulis.z[:, bit].astype(np.uint64) << np.uint64(bit % 64)
            coefficients = np.asarray(value.coeffs, dtype=np.complex128) * (-1j) ** value.paulis.phase
            return ingest_pauli_masks(x, z, coefficients, num_qubits=q, max_bytes=max_bytes)
    raise TypeError("operator input requires numerical square data, canonical CSR/CSC, SparsePauliOp, PeriodicStencil or OperatorInput")
