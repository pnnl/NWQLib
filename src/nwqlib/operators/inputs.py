"""Conversion of matrices and Pauli sums into OperatorInput objects, and their classical access.

Each conversion is checked against a byte limit before it copies data, and
no quantum oracle is built.
"""

from collections.abc import Sized
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Complex
import sys

import numpy as np
from scipy import sparse

from nwqlib._limits import ResourceLimitError, byte_allowance, exceeds, memory_limit
from nwqlib._validation import finite_real, integer
from nwqlib.core.records import Basis, InputRef, Source
from .access import InputManifest, _check_bytes, _check_products
from ._pauli import PauliTerms, apply_terms, combine_terms, pauli_action_requirements


# Little-endian convention shared with Qiskit: coordinate index
# k = sum_q b_q 2**q, so qubit 0 is the least significant bit and the
# rightmost tensor factor. Every admitted operator and state uses it.
ORDER = "coordinate index increasing; qubit 0 is rightmost tensor/bit position"


def _basis(dimension):
    return Basis(identity="computational", dimension=dimension, ordering=ORDER)


def _same_storage(array, stored):
    """Whether ``array`` reads the elements of the immutable ``stored`` array: the same buffer, dtype, shape and strides.

    The accessors return new views of a stored snapshot, so this replaces a
    comparison of ndarray objects.
    """
    return isinstance(array, np.ndarray) and array.__array_interface__ == stored.__array_interface__


def _freeze_array(array):
    """Return an immutable byte copy of the array, or a new view of it when its bytes already are immutable.

    Even setflags cannot make its bytes writable. A C-contiguous array whose
    storage is an immutable bytes object, such as an earlier result of this
    function, cannot change, so its storage is shared without a second copy.
    The result is still a new ndarray, so a later change to the dtype or
    shape of the caller's array does not reinterpret it. Any other array,
    read-only or not, is copied, because a writable view or buffer can still
    change it.
    """
    owner = array
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if type(owner) is bytes and array.flags.c_contiguous:
        return array.view()
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _freeze_numbers(array, max_bytes, held=0):
    """Return ``_freeze_array`` of a native float64 or complex128 input, with each -0.0 real or imaginary part as +0.0.

    -0.0 and +0.0 are one real number, and an input's identity hashes its bytes
    (``_digest``), so ``-A`` with a zero entry would otherwise differ from the same
    operator written with +0.0, and a comparison that needs matching inputs would
    refuse it. ``core.records._finite_zero`` applies this rule to scalars and
    ``FrozenArray`` to float64 arrays. An input without -0.0 takes only the copy
    ``_freeze_array`` makes. One with -0.0 needs a writable copy beside that
    snapshot, which ``max_bytes`` counts together with ``held``, the bytes of
    earlier snapshots of the same input that are live now. Under IEEE 754 round
    to nearest, ``x + 0.0`` is ``x`` for every nonzero ``x``, a subnormal included,
    and +0.0 for -0.0, so every other component keeps its bits.
    """
    stored = _freeze_array(array)
    # -0.0 is the one float64 whose bits read as this int64 (core.records._scan_float64).
    if not (stored.reshape(-1).view(np.int64) == -(1 << 63)).any():
        return stored
    _check_bytes(held + 2 * stored.nbytes, max_bytes, "negative zero normalization")
    canonical = stored + 0.0
    del stored
    return _freeze_array(canonical)


def _digest(kind, shape, arrays=()):
    """Hash the array content once, when the input object is created, including dtype, shape and order.

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
    """Hash the width, the word order and the ordered x, z and coefficient bytes."""
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


_PLAIN_NUMBERS = frozenset((int, float, complex))


def _array_input(value, *, ndim, max_bytes, extra_bytes_per_item=0):
    """Convert only known numerical containers, after their output-size check.

    Integers and real data become float64, and complex data become complex128.
    No other array protocol, object conversion or iterator without a length is used.
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
            # The exact item types decide admission, so bool, its NumPy
            # counterpart and int or float subclasses stay rejected.
            types = set(map(type, row))
            if not types <= _PLAIN_NUMBERS and not all(
                    t in _PLAIN_NUMBERS or (issubclass(t, np.number) and np.dtype(t).kind in "iufc")
                    for t in types):
                raise TypeError("numerical data must contain real or complex numbers, not bool/object/string values")
            complex_data = complex_data or any(issubclass(t, (complex, np.complexfloating)) for t in types)
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
    """Parameters of the periodic operator `mass*I + diffusion*(2I - S - S†) + i*potential*Z_0`.

    `S` is the cyclic shift `S|j> = |j+1 mod 2**q>` on a grid of `2**q`
    points, and `Z_0` is Pauli Z on qubit 0. For `q = 1` both wrap-around
    edges are kept. With `potential = 0` the operator is Hermitian, with
    spectral endpoints `mass` and `mass + 4*diffusion`.

    Build it with `PeriodicStencil(num_qubits, mass, diffusion, potential)`
    and pass it as a Problem's matrix, for example `LinearDynamics(A=...)`,
    or to [`operator_input`][nwqlib.operators.inputs.operator_input]. Only
    the four parameters are stored, never a matrix or a Pauli expansion.

    Args:
        num_qubits: Positive integer q, a Python or NumPy integer, kept as a
            Python int. The grid has `2**q` points.
        mass: Nonnegative finite coefficient of the identity.
        diffusion: Nonnegative finite coefficient of `2I - S - S†`.
        potential: Finite real coefficient of `i*Z_0`. Zero gives a
            Hermitian operator.

    Raises:
        ValueError: If `num_qubits` is not a positive integer, if `mass` or
            `diffusion` is negative, or if a coefficient is not finite.
    """

    num_qubits: int
    mass: float
    diffusion: float
    potential: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "num_qubits", integer(self.num_qubits, "num_qubits", 1))
        for name in ("mass", "diffusion", "potential"):
            value = finite_real(getattr(self, name), name)
            if name != "potential" and value < 0:
                raise ValueError("periodic stencil mass and diffusion must be nonnegative")
            object.__setattr__(self, name, 0.0 if value == 0 else value)


@dataclass(frozen=True, slots=True)
class CartesianParts:
    """The Hermitian parts of a generator `A = L + iH`, each held as its own operator input.

    [`ingest_cartesian`][nwqlib.operators.inputs.ingest_cartesian] builds it
    and `OperatorInput.cartesian_parts()` returns it. With
    `L = (A + A†)/2` and `H = (A - A†)/(2i)`, both parts are Hermitian and
    `A = L + iH`. The parts define A, so A is not kept separately.

    Attributes:
        dissipative: The Hermitian part L, an exactly Hermitian Pauli sum,
            periodic band operator or supplied block encoding, or `None`
            when L is exactly zero.
        hamiltonian: The part H, in the same forms, or `None` when H is
            exactly zero.
        dissipative_lower_bound: The lower bound on the smallest eigenvalue
            of L that you stated to `ingest_cartesian`, or `None`. NWQLib
            does not verify it.
    """

    dissipative: "OperatorInput | None"
    hamiltonian: "OperatorInput | None"
    dissipative_lower_bound: float | None = None


@dataclass(frozen=True, slots=True)
class SuppliedBlockEncoding:
    """A block-encoding circuit supplied for an operator X, with the properties asserted about X.

    [`ingest_block_encoding`][nwqlib.operators.inputs.ingest_block_encoding]
    builds it and `OperatorInput.supplied_block_encoding()` returns it. The
    circuit promises `||X - alpha (<0_a| tensor I) U (|0_a> tensor I)|| <= delta`
    with `alpha = encoding.alpha` and `delta = encoding.error_bound`, which
    is `None` when unknown. Then `||X|| <= alpha + delta`, because the
    compressed block has norm at most one. With an unknown
    delta, alpha is not a bound on `||X||`. NWQLib neither simulates the
    circuit nor checks these assertions.

    Attributes:
        encoding: The `BlockEncoding`, holding a copy of the supplied
            circuit with its metadata removed.
        hermitian: Whether X is asserted to be exactly Hermitian.
        lower_bound: Asserted lower bound on the smallest eigenvalue of a
            Hermitian X, or `None`.
    """

    encoding: object
    hermitian: bool
    lower_bound: float | None


@dataclass(frozen=True, eq=False, init=False, slots=True)
class OperatorInput:
    """An operator ready to use in a Problem: an immutable copy of its data and its metadata.

    [`operator_input`][nwqlib.operators.inputs.operator_input] and the
    `ingest_*` functions return it. Constructing it directly raises
    `TypeError`, and its fields cannot be replaced. It keeps the data in the
    representation it was given (dense, CSR, CSC, Pauli terms, ordered
    fermionic terms or periodic-stencil parameters) and never converts it to
    a dense matrix implicitly. Pass it wherever a matrix or operator is
    accepted, for example `Eigenproblem(A=handle)`, where `handle` is the
    input object. Reusing one input object reuses its copy of the data and
    its content hash.

    The methods below read the data or apply it classically. Each raises
    `ValueError` when the representation does not provide that access, or
    when the object was rebuilt by `from_record` and holds metadata only.
    The fields below are read-only. Below, `D` is the operator's
    dimension, `basis.dimension`.

    Attributes:
        manifest: The [`InputManifest`][nwqlib.operators.access.InputManifest]:
            content hash, representation, basis, stored size, available
            classical operations and the checks made when the input object
            was created.
        structure: `"hermitian"` or `"general"`. For stored data it comes
            from an exact check with no tolerance: equality of a stored
            matrix with its conjugate transpose, real coefficients of a Pauli
            sum, conjugate-symmetric coefficients of a periodic band operator,
            or zero `potential` of a periodic stencil. Fermionic terms are
            always `"general"`. A supplied block encoding takes it from your
            `hermitian` argument, which NWQLib does not check. A Cartesian
            input is `"hermitian"` when it has no part H. An object rebuilt by
            `from_record` may also declare `"unitary"`.
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
        """The computational [`Basis`][nwqlib.core.records.Basis]: dimension and coordinate order, qubit 0 rightmost."""
        return self.manifest.basis

    @property
    def reference(self):
        """The input's reference: content hash, representation and source."""
        return self.manifest.reference

    def to_record(self):
        """Return a JSON-ready description of this input object without its numerical data.

        Returns:
            record (dict): The keys `format`, `manifest` and `structure`.
                Building it copies and hashes no numerical data.
        """
        return dict(format="nwqlib.operator_input/2",
                    manifest=self.manifest.model_dump(mode="json", exclude_computed_fields=True),
                    structure=self.structure)

    @classmethod
    def from_record(cls, record):
        """Rebuild, from a `to_record()` description, an input object that holds metadata only.

        The object reports the saved metadata, but every data method raises
        `ValueError`, even for operations its manifest lists. Saved Results
        and Runs restore their numerical inputs separately.

        Args:
            record (dict): A description returned by `to_record()`.

        Returns:
            handle (OperatorInput): The rebuilt input object, named `handle`
                here, which holds no numerical data.

        Raises:
            ValueError: If `record` is not such a description.
        """
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
        """Return the stored Pauli table without expanding it.

        The table holds uint64 `x` and `z` masks and complex128
        `coefficients`, one row per Pauli word. Its `labels()` method returns
        `(label, coefficient)` pairs in Qiskit order, with the rightmost
        character acting on qubit 0.

        Returns:
            table (PauliTerms): The immutable stored table.

        Raises:
            ValueError: If the operator is not a Pauli sum.
        """
        self._require_data()
        if "pauli_terms" not in self.manifest.access:
            raise ValueError("structured Pauli access is unavailable")
        return self._data

    def dense_array(self):
        """Return a read-only view of the stored dense matrix, without copying it.

        Returns:
            matrix (numpy.ndarray): The stored float64 or complex128 matrix,
                in a new view, so that changing its shape or dtype leaves the
                operator unchanged.

        Raises:
            ValueError: If the operator was not given as a dense matrix. No
                other representation is converted to dense.
        """
        self._require_data()
        if self.manifest.reference.representation != "dense" or "entries" not in self.manifest.access:
            raise ValueError("dense access requires an ingested dense operator; no implicit densification")
        return self._data.view()

    def fermion_terms(self):
        """Return the stored ordered fermionic table, without mapping it to qubits.

        Returns:
            table (FermionTerms): The stored
                [`FermionTerms`][nwqlib.operators._fermion.FermionTerms].
                Its `to_pauli(mapping=...)` maps it to a Pauli operator.

        Raises:
            ValueError: If the operator is not a fermionic operator.
        """
        self._require_data()
        if "fermion_terms" not in self.manifest.access:
            raise ValueError("ordered Fermion access is unavailable")
        return self._data

    def periodic_stencil(self):
        """Return the stored [`PeriodicStencil`][nwqlib.operators.inputs.PeriodicStencil] parameters.

        The OperatorInput of a periodic stencil provides no classical
        matrix-vector product.

        Returns:
            stencil (PeriodicStencil): The stored parameters.

        Raises:
            ValueError: If the operator is not a periodic stencil.
        """
        self._require_data()
        if not isinstance(self._data, PeriodicStencil):
            raise ValueError("periodic stencil parameters require their native input factory")
        return self._data

    def band(self):
        """Return the bands of a periodic band operator.

        Returns:
            bands (BandSpecification): The coalesced offsets, in
                `(-D/2, D/2]` and increasing order, and their nonzero
                coefficients, after the coalescing of
                [`ingest_band`][nwqlib.operators.inputs.ingest_band].

        Raises:
            ValueError: If the operator is not a periodic band operator.
        """
        self._require_data()
        if self.manifest.reference.representation != "band":
            raise ValueError("periodic band access requires an operator from ingest_band")
        return self._data

    def cartesian_parts(self):
        """Return the Hermitian parts L and H of `A = L + iH` given to `ingest_cartesian`.

        Returns:
            parts (CartesianParts): The two part inputs, `None` for an
                exactly zero part.

        Raises:
            ValueError: If the operator was not built by
                [`ingest_cartesian`][nwqlib.operators.inputs.ingest_cartesian].
        """
        self._require_data()
        if self.manifest.reference.representation != "cartesian":
            raise ValueError("Cartesian part access requires an operator from ingest_cartesian")
        return self._data

    def supplied_block_encoding(self):
        """Return the supplied block-encoding circuit and the asserted properties of its operator.

        Returns:
            encoding (SuppliedBlockEncoding): The copy of the circuit,
                its `alpha`, ancilla count and error bound, and the asserted
                Hermiticity and lower eigenvalue bound.

        Raises:
            ValueError: If the operator was not built by
                [`ingest_block_encoding`][nwqlib.operators.inputs.ingest_block_encoding].
        """
        self._require_data()
        if self.manifest.reference.representation != "block_encoding":
            raise ValueError("supplied block-encoding access requires an operator from ingest_block_encoding")
        return self._data

    def entry(self, row: int, column: int):
        """Return one matrix entry, without converting a sparse matrix to dense.

        Args:
            row (int): Row index, `0 <= row < D`.
            column (int): Column index, `0 <= column < D`.

        Returns:
            value (complex): The stored entry, zero where a sparse matrix
                stores none.

        Raises:
            ValueError: If the operator is not a dense or sparse matrix, or
                an index is outside the dimension.
        """
        self._require_data()
        if "entries" not in self.manifest.access:
            raise ValueError("entry access is unavailable")
        dimension = self.manifest.basis.dimension
        try:
            row, column = integer(row, "row", 0), integer(column, "column", 0)
        except ValueError:
            raise ValueError("entry indices must lie in the operator dimension") from None
        if row >= dimension or column >= dimension:
            raise ValueError("entry indices must lie in the operator dimension")
        return complex(self._data[row, column])

    def sparse_entries(self, *, max_bytes=None):
        """Return the stored `(data, indices, indptr)` arrays of a CSR or CSC matrix.

        The arrays are the stored read-only arrays, in their original CSR or
        CSC orientation and index dtype. Their stored bytes are checked
        against `max_bytes` first. Nothing is converted, traversed or copied.

        Args:
            max_bytes (int | None): Limit on the stored bytes. Default `None`, the
                physical memory of this computer.

        Returns:
            arrays (tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]): The
                values, indices and index pointers.

        Raises:
            ValueError: If the operator is not a CSR or CSC matrix, or its
                stored bytes exceed `max_bytes`.
        """
        self._require_data()
        if self.manifest.reference.representation not in {"csr", "csc"} or "entries" not in self.manifest.access:
            raise ValueError("compressed entries require an ingested CSR/CSC operator")
        _check_bytes(self.manifest.payload_bytes, max_bytes, "sparse entries")
        # New views share the stored arrays without exposing their headers.
        return self._data.data.view(), self._data.indices.view(), self._data.indptr.view()

    def matvec(self, vector: np.ndarray, *, max_bytes=None,
               max_products=None, limit_name="max_products") -> np.ndarray:
        """Apply the operator to a vector classically and return `A @ vector`.

        Before it reads the vector, the call checks the bytes of the vector,
        output and work arrays against `max_bytes`, and the number of scalar
        products against `max_products`. For operator dimension `D`, sparse
        stored-entry count `nnz` and `M` stored Pauli terms, the product count is
        ``D**2`` for a dense matrix, ``nnz + D`` for a sparse one, and the
        grouped-action count on [Input cost
        controls](../development/input_contracts.md) for a Pauli operator.
        That count includes a possible retry of all terms in stored order
        after grouped arithmetic overflows or produces an invalid result.
        It is derived from the representation, not measured time. Beyond
        the stored operator the action keeps `O(D)` numerical storage, and
        a Pauli action also uses workspace for bounded batches of vector
        coordinates, called coordinate tiles, and `O(M)` grouping metadata.

        Args:
            vector (numpy.ndarray): One-dimensional float64 or complex128
                array of length D with finite entries.
            max_bytes (int | None): Limit on the checked bytes. Default `None`, the
                physical memory of this computer.
            max_products (int | None): Limit on the scalar products. Default
                `None`, no limit.
            limit_name (str): Name of the caller's option that supplies
                `max_products`, used in the error message.

        Returns:
            result (numpy.ndarray): The product `A @ vector`.

        Raises:
            ValueError: If the input object has no classical action (a
                periodic stencil, fermionic terms or an object rebuilt from
                metadata only), if a
                Pauli operator acts on more than 64 qubits, if `vector` has
                another shape, dtype or a nonfinite entry, or if a limit is
                exceeded.
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
    """Return the ``(D, bytes, products)`` allowances checked before a matrix-vector product.

    Whether the representation provides the product is checked before the
    vector is read. Dense and CSR/CSC storage reserve 33D bytes for an
    input of at most complex128 size, a complex128 output and a one-byte
    finite mask. Work is D**2 for dense storage and nnz+D for CSR/CSC.
    These are logical upper bounds, excluding the operator's own data and
    the overhead of the allocator or the resident memory of the process.

    For M stored Pauli terms use g = min(M, D), an upper bound on
    distinct single-word flip masks obtained from counts alone. Call
    ``pauli_action_requirements(D, M, g)`` for one column with possible
    input conversion. With t = min(D, _PAULI_TILE), bytes are
    ``49*D + 254*M + 1282 + max(9*M, 96*t)`` and work is
    ``(M+g) + 3*M + D*(M+g+1) + D + M*D``. ``pauli_action_requirements``
    covers the Y counts formed before the tiles, the Python containers
    that group the terms, the grouped action and a possible retry in term
    order. The check reads metadata only and does not count the distinct
    flip masks.
    """
    operator._require_data()
    if "matvec" not in operator.manifest.access:
        raise ValueError("classical matvec is unavailable for this handle")
    d = operator.manifest.basis.dimension
    representation = operator.manifest.reference.representation
    if representation == "pauli":
        if (d.bit_length() - 1 + 63) // 64 != 1:
            raise ResourceLimitError(f"a full-state action on {d.bit_length() - 1} qubits holds {d} amplitudes, more "
                                     f"than a 64-bit index addresses and any memory holds")
        terms = operator.manifest.shape[0]
        size, work = pauli_action_requirements(d, terms, min(terms, d))
        return d, size, work
    work = operator._data.nnz + d if representation in ("csr", "csc") else d * d
    return d, 33 * d, work


def _scaled_observable_requirements(operator):
    """Return the allowances for one action of the scaled operator, kept in its own representation, on a vector.

    Let N be the state dimension, P the original operator payload bytes,
    and K the stored-entry count (Pauli terms, sparse nnz or dense N**2).
    For ``(_, B_action, W_action) = _matvec_requirements(operator)``,
    return ``B_scaled = B_action + 32*N + 3*P + 48*K`` and
    ``W_scaled = W_action + 2*N + 18*K``.

    The 3P+48K bytes and the 18K work reserve the exceptional scaled copy
    of the operator in ``_scaled_observable_action``. It is formed when
    ``|e| > 512``, the range to which that function's exactness argument is
    restricted, when for ``e > 0`` the sufficient test of
    ``_scales_exactly`` does not establish that every stored component
    scales exactly (the test can fail for a component that would), or when
    for ``e > 0`` the framed vector does not scale back exactly. Neither
    guard asserts that the scaling loses bits. Other scales act unscaled or
    move the exact scaling onto the vector and make no copy, so for them
    the reservation exceeds the actual use. The payload term covers the copy,
    its compressed indices and the immutable copy of the Pauli table. The
    entry term covers the lost-component masks, and the tiled exactness
    scan of ``_scales_exactly`` fits inside it. The vector reserve covers
    the framed input, its reversibility check and the final dot product.
    Work covers four maximum/minimum visits for the exponent, two ldexp
    visits, eight mask visits, one initial copy and three immutable
    Pauli-array copies per entry.

    For Pauli storage K=L, and ``pauli_action_requirements`` uses
    g=min(L,N), possible input conversion and one column. It already
    includes the 9L bytes formed before the tiles and the 3L
    preprocessing work. The coefficients are scaled once per operator, so
    the 3P+48L storage and 18L work do not depend on the number of state
    columns. This helper itself covers one vector action.
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
    """Return the binary exponent of the largest absolute real or imaginary stored entry.

    Scaling by a power of two leaves the coordinates unchanged. The largest
    magnitude is the larger of the largest component and the negated
    smallest one, read from the real and imaginary views without an
    array-sized absolute value.
    """
    from math import frexp

    values = (operator.pauli_terms().coefficients if "pauli_terms" in operator.manifest.access
              else operator._data.data if operator.reference.representation in ("csr", "csc")
              else operator.dense_array())
    parts = (values.real, values.imag) if np.iscomplexobj(values) else (values,)
    maximum = max(max(float(np.max(part, initial=0.0)), -float(np.min(part, initial=0.0))) for part in parts)
    return frexp(maximum)[1]


def _scaled_observable(operator, exponent):
    """Return a copy of O/2**exponent in its native storage, or None when the scaling turns a nonzero component into zero."""
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


# Stored components scanned per tile by _scales_exactly (512 KiB of float64).
_SCALE_SCAN_TILE = 1 << 16


def _scales_exactly(values, exponent):
    """Whether every real and imaginary component of ``values`` times 2**-exponent is exact.

    For ``exponent <= 0`` the factor is at least one and, with the largest
    component below ``2**exponent``, every product is below one, so none
    overflows and a subnormal is scaled up exactly. For ``exponent > 0`` the
    sufficient test is that no nonzero component lies below
    ``2**(exponent-1022)``, so every product is normal or zero. A component
    that could lose bits, or become zero, fails the test. It reads the
    components in tiles without a full-size temporary.
    """
    from math import ldexp

    if exponent <= 0:
        return True
    smallest = ldexp(1.0, exponent - 1022)
    flat = np.ascontiguousarray(values).reshape(-1)
    if np.iscomplexobj(flat):
        flat = flat.view(flat.real.dtype)
    for start in range(0, flat.size, _SCALE_SCAN_TILE):
        part = np.abs(flat[start:start + _SCALE_SCAN_TILE])
        if np.any((part < smallest) & (part != 0)):
            return False
    return True


def _stored_components(operator):
    """Return the stored coefficients or entries that scaling an observable multiplies."""
    if "pauli_terms" in operator.manifest.access:
        return operator.pauli_terms().coefficients
    return operator._data.data if operator.reference.representation in ("csr", "csc") else operator._data


def _scaled_copy_certain(operator, exponent):
    """Whether ``_scaled_observable_action`` forms the scaled copy of the operator for every vector.

    It does when ``|exponent| > 512`` or when ``_scales_exactly`` cannot
    establish that O*2**(-exponent) is exact. Both are sufficient-condition
    guards decided by the operator alone before any state exists, so the
    copy can also be formed for an operator whose scaling would be exact.
    """
    return not (abs(exponent) <= 512 and _scales_exactly(_stored_components(operator), exponent))


def _scaled_observable_action(operator, exponent, direction, admit_copy=None, certain=None):
    """Return ``(v, (O/2**exponent) v)`` for ``v = direction()``, or None when the scaling turns a nonzero component of O into zero.

    The action is the one of ``_scaled_observable`` in the operator's own
    representation, bit for bit, without its full copy when both scalings
    are exact. Then ``O/2**e`` and ``v/2**e`` are exact, every product of a
    stored component and a vector component is the same real number
    ``o*v/2**e`` in both forms and rounds identically, and the sums of those
    products follow the same operation order. A sum of Pauli coefficients
    alone is the exact ``2**e`` multiple of the scaled one, because
    ``|e| <= 512`` leaves no overflow and an addition with a subnormal result
    is exact. When ``_scaled_copy_certain`` holds, the scaled copy is formed
    for every vector, and the caller checks its limits beforehand. When
    only the vector does not scale back exactly, which needs ``e > 0`` and
    a vector component small enough to lose bits when divided by ``2**e``,
    the copy is formed after ``admit_copy()``, when given, has checked its
    limits. ``direction`` is called only when no component is lost.
    ``certain`` is the caller's earlier ``_scaled_copy_certain`` result for
    this operator and exponent, computed here when None.
    """
    pauli = "pauli_terms" in operator.manifest.access
    if certain is None:
        certain = _scaled_copy_certain(operator, exponent)
    if not certain:
        vector = direction()
        framed = np.array(vector, copy=True)
        for component in ("real", "imag") if np.iscomplexobj(framed) else ("real",):
            part = getattr(framed, component)
            np.ldexp(part, -exponent, out=part)
            if not np.array_equal(np.ldexp(part, exponent), getattr(vector, component)):
                break
        else:
            if pauli:
                table = operator.pauli_terms()
                return vector, apply_terms(table, framed, num_qubits=table.num_qubits)
            return vector, operator._data @ framed
        if admit_copy is not None:
            admit_copy()
        scaled = _scaled_observable(operator, exponent)
    else:
        scaled = _scaled_observable(operator, exponent)
        if scaled is None:
            return None
        vector = direction()
    if pauli:
        return vector, apply_terms(scaled, vector, num_qubits=scaled.num_qubits)
    return vector, scaled @ vector


def _scaled_observable_moment(operator, vector):
    """Return (v†(O/2**e)v, e) from one action in the operator's own representation, without multiplying back by 2**e.

    An operator whose largest stored component is at least 1/2 (e >= 0)
    acts unscaled, and the result is returned with e = 0. Its products are
    those of O/2**e times 2**e, so the unscaled action underflows no more
    and differs only by that exact factor, and an overflow leaves a
    non-finite value. Only then, or for a smaller operator, does the scaled
    action of ``_scaled_observable_action`` run. It moves the exact scaling
    onto a copy of the vector when that is reversible, which for
    ``-512 <= e < 0`` it always is, and otherwise forms the scaled copy of
    the operator. When a common scale would turn a nonzero stored component
    into zero, the unscaled operator is used before applying anything. This
    keeps the supported range of magnitudes without a silently pruned
    observable.
    """
    pauli = "pauli_terms" in operator.manifest.access
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = _observable_exponent(operator)
        if exponent >= 0:
            data = operator._data
            value = float(np.vdot(vector, apply_terms(data, vector, num_qubits=data.num_qubits)
                                  if pauli else data @ vector).real)
            if np.isfinite(value):
                return value, 0
        scaled = _scaled_observable_action(operator, exponent, lambda: vector)
        if scaled is None:
            if exponent >= 0:
                return None, 0
            data, exponent = operator._data, 0
            scaled = vector, (apply_terms(data, vector, num_qubits=data.num_qubits) if pauli else data @ vector)
        value = float(np.vdot(vector, scaled[1]).real)
    return (value if np.isfinite(value) else None), exponent


def ingest_periodic_stencil(num_qubits: int, mass, diffusion, potential=0.0, *, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput for `mass*I + diffusion*(2I - S - S†) + i*potential*Z_0`.

    This is the operator of [`PeriodicStencil`][nwqlib.operators.inputs.PeriodicStencil]
    on a periodic grid of `2**q` points, with the cyclic shift
    `S|j> = |j+1 mod 2**q>`. For `q = 1` both wrap-around edges are kept.
    Only the four parameters are stored, so the input object has no matrix,
    Pauli expansion or matrix-vector access. With `potential = 0` the operator is
    Hermitian, with spectral endpoints `mass` and `mass + 4*diffusion`.

    Args:
        num_qubits (int): Positive integer q.
        mass (float): Nonnegative finite coefficient of the identity.
        diffusion (float): Nonnegative finite coefficient of `2I - S - S†`.
        potential (float): Finite real coefficient of `i*Z_0`.
        max_bytes (int | None): Limit on the bytes checked before the parameters
            are stored. Default `None`, the physical memory of this computer.

    Returns:
        operator (OperatorInput): Structure `"hermitian"` when `potential`
            is zero, `"general"` otherwise.

    Raises:
        ValueError: If `num_qubits` is not a positive integer, if `mass` or
            `diffusion` is negative, or if a coefficient is not finite.
    """
    # Size law. The payload is three float64 parameters and q's integer
    # magnitude. Before the dimension metadata is formed, the check covers
    # the parameters plus their hash array (48 bytes), q's magnitude and two
    # q-bit dimension integer slots. Work is q+4 and items are four
    # parameters. These are logical size laws, not Python object or RSS
    # accounting.
    num_qubits = integer(num_qubits, "num_qubits", 1)
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


def _signed_offset(offset, dimension):
    """Return the representative of ``offset`` modulo ``dimension`` in ``(-dimension/2, dimension/2]``."""
    value = offset % dimension
    return value - dimension if 2 * value > dimension else value


def ingest_band(offsets, coefficients, *, num_qubits: int, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput for the periodic band operator `X = sum_b beta_b S^b`, without forming a matrix.

    `S` is the cyclic shift `S|x> = |x+1 mod D>` on `D = 2**q` grid points.
    Offsets that are equal modulo D are summed with stable summation, and a
    band whose sum is exactly zero is removed, so no nonzero coefficient is
    pruned. The operator keeps the representatives of the offsets in
    `(-D/2, D/2]`, in increasing order. Because `(S^b)† = S^-b`, X is Hermitian exactly when
    `beta_{-b} = conj(beta_b)` for every band. The offsets 0 and, for
    `D >= 2`, `D/2` are their own partners, so their coefficients must be
    real. That exact test, without a tolerance, sets `structure`. Since S is unitary, `||X|| <= sum_b |beta_b|`,
    and for Hermitian X every eigenvalue lies in
    `[beta_0 - sum_{b != 0} |beta_b|, beta_0 + sum_{b != 0} |beta_b|]`
    which LCHS evaluates exactly from the coefficients.

    Args:
        offsets (Sequence[int]): Integer band offsets b. Band b couples `|x>`
            to `|x + b mod D>`.
        coefficients (Sequence[complex]): One finite coefficient per offset.
        num_qubits (int): Positive number of qubits q.
        max_bytes (int | None): Byte limit. Default `None`, the physical
            memory of this computer.

    Returns:
        operator (OperatorInput): The band operator, representation
            `"band"`, with no matrix-vector access.

    Raises:
        ValueError: If `num_qubits` is not a positive integer, the lengths
            differ, a coefficient is not finite, a stored offset lies outside
            the int64 range in which the input digest and the archive store
            offsets, or the bands exceed `max_bytes`.
        TypeError: If an offset is not an integer or a coefficient is not a
            number.
    """
    from nwqlib.subroutines.block_encoding.banded import BandSpecification

    num_qubits = integer(num_qubits, "num_qubits", 1)
    if not isinstance(offsets, Sized) or not isinstance(coefficients, Sized) or len(offsets) != len(coefficients):
        raise ValueError("band offsets and coefficients require equal finite lengths")
    # Size law. Each band holds one int64 offset and one complex128
    # coefficient (24 bytes), and coalescing keeps the input pair, one bucket
    # entry and the stored arrays, 96 bytes per band, plus the q-bit
    # dimension integer.
    _check_bytes(96 * len(offsets) + (num_qubits + 8) // 8, max_bytes, "periodic band conversion")
    dimension = 1 << num_qubits
    pairs = []
    for offset, coefficient in zip(offsets, coefficients, strict=True):
        if isinstance(offset, (bool, np.bool_)) or not isinstance(offset, (int, np.integer)):
            raise TypeError("band offsets must be integers")
        if isinstance(coefficient, (bool, np.bool_)) or not isinstance(coefficient, Complex):
            raise TypeError("band coefficients must be numerical scalars")
        pairs.append((_signed_offset(int(offset), dimension), complex(coefficient)))
    combined = sorted(combine_terms(pairs))
    # The input digest and the execution archive store the offsets as int64.
    if any(not -(1 << 63) <= offset < 1 << 63 for offset, _ in combined):
        raise ValueError("stored band offsets must fit the int64 offset storage of the input digest and archive")
    stored = dict(combined)
    hermitian = all(value == stored.get(_signed_offset(-offset, dimension), 0).conjugate()
                    for offset, value in combined)
    offset_array = np.array([offset for offset, _ in combined], dtype=np.int64)
    coefficient_array = np.array([value for _, value in combined], dtype=np.complex128)
    data = BandSpecification(offsets=tuple(int(value) for value in offset_array),
                             coefficients=tuple(complex(value) for value in coefficient_array),
                             num_qubits=num_qubits)
    manifest = _manifest("band", dimension, (len(combined),), "int64+complex128", 24 * len(combined),
        _digest("band.periodic.signed-offsets", (num_qubits, len(combined)), (offset_array, coefficient_array)),
        (), ("finite coefficients", "offsets summed modulo 2**q", "exact-zero bands removed",
             "Hermitian" if hermitian else "general"),
        "B bands; O(B) coalescing; no entries, matvec or Pauli expansion")
    return OperatorInput._from_admitted(manifest, "hermitian" if hermitian else "general", data)


def ingest_cartesian(dissipative=None, hamiltonian=None, *, dissipative_lower_bound=None,
                     max_bytes=None) -> OperatorInput:
    """Return an OperatorInput for `A = L + iH` from its two Hermitian parts.

    Use it for `LinearDynamics(A=...)` when you have L and H separately, for
    example a periodic diffusion band for L and a local Pauli potential for
    H. Each part is an exactly Hermitian Pauli sum, a Hermitian periodic band
    operator from [`ingest_band`][nwqlib.operators.inputs.ingest_band] or a
    supplied block encoding from
    [`ingest_block_encoding`][nwqlib.operators.inputs.ingest_block_encoding]
    that asserts Hermiticity, or anything that
    [`operator_input`][nwqlib.operators.inputs.operator_input] turns into
    one of these. `None` means that part is exactly zero. The two parts
    define A, so no dense A is formed and no second representation of A is
    kept. The content identifier hashes the identifiers of the parts in the
    order L, H. The default coherent LCHS estimator evolves Pauli and band
    parts and builds no evolution under a supplied-encoding part, so for a
    positive elapsed time and a nonzero initial state or source its
    planning raises `ApplicabilityError` for such a part. The sampled
    estimator, `LCHS(estimator="sampled_pairs")` on the
    `"qsp_block_encoding"` backend, evolves all three forms.

    `dissipative_lower_bound` states a lower bound `ell_u` on the smallest
    eigenvalue of L that the stored coefficients do not show, for example
    0 for a Laplacian written as a Pauli sum. Otherwise LCHS has only the
    structural bound `c_I - sum_{P != I} |c_P|` of a Pauli sum, or its
    counterpart for a band operator, which can lie far below the smallest
    eigenvalue. LCHS uses `ell = max(ell_structural, ell_u)`,
    `psd_shift = max(0, -ell)` and `scalar_decay = max(0, ell)`, from
    `exp(-A t) = exp(-ell t) exp(-(A - ell I) t)`. These hold only if
    `ell_u <= lambda_min(L)`. NWQLib does not verify this and computes no
    spectrum. When `ell_u` improves on the structural bound,
    `Plan.reconstruction.psd_premise` reads `"user_assertion"`, and the shift
    and the error bounds that depend on it are conditional on your
    statement. The [LCHS guide](../algorithms/lchs.md#declared-lower-bound)
    gives its effect on shots, which routes gain from it and what a wrong
    value does. The value enters the content identifier, because it decides
    the shift and the kernel grid.

    Args:
        dissipative (object): The Hermitian part L, or `None`.
        hamiltonian (object): The Hermitian part H, or `None`.
        dissipative_lower_bound (float | None): Lower bound on the smallest
            eigenvalue of L, in the units of A. Default `None`, no statement.
            L must be a Pauli sum or band operator, since a supplied encoding
            states its own `lower_bound` in
            [`ingest_block_encoding`][nwqlib.operators.inputs.ingest_block_encoding].
            Planning refuses a value above the mean eigenvalue `tr(L)/D` of L,
            which is `c_I` of a Pauli sum or the offset-0 coefficient of a band,
            and a value equal to it unless L is a multiple of the identity,
            because `lambda_min(L)` is at most that mean, with equality only
            for a multiple of the identity. A false value below the mean plans
            as if it were true.
        max_bytes (int | None): Byte limit of any conversion. Default `None`, the
            physical memory of this computer.

    Returns:
        operator (OperatorInput): Representation `"cartesian"`, structure
            `"hermitian"` when H is `None` and `"general"` otherwise.

    Raises:
        ValueError: If both parts are `None`, a part is not one of the forms
            above, is not exactly Hermitian, holds no data, or the parts use
            different bases, or if `dissipative_lower_bound` is not finite, is
            given without L or is given for a supplied encoding of L.
    """
    parts = []
    for name, value in (("dissipative", dissipative), ("hamiltonian", hamiltonian)):
        if value is None:
            parts.append(None)
            continue
        part = operator_input(value, max_bytes=max_bytes)
        part._require_data()
        if part.manifest.reference.representation not in ("pauli", "band", "block_encoding"):
            raise ValueError(f"{name} part must be a Pauli sum, periodic band operator or supplied block encoding")
        if part.structure != "hermitian":
            raise ValueError(f"{name} part must be exactly Hermitian")
        parts.append(part)
    present = [part for part in parts if part is not None]
    if not present:
        raise ValueError("Cartesian input requires at least one part")
    basis = present[0].basis
    if any(part.basis != basis for part in present):
        raise ValueError("Cartesian parts must use one basis")
    identifiers = [None if part is None else part.reference.identity for part in parts]
    hashed, work_law = "the ordered part identifiers", "two part references; the parts keep their own representations"
    if dissipative_lower_bound is not None:
        if parts[0] is None:
            raise ValueError("dissipative_lower_bound bounds the part L, so it needs a dissipative part")
        if parts[0].manifest.reference.representation == "block_encoding":
            raise ValueError("a supplied encoding of L states its bound through lower_bound of ingest_block_encoding")
        # Adding 0.0 turns a negative zero into 0.0, which bounds the same operators and must hash the same.
        dissipative_lower_bound = finite_real(dissipative_lower_bound, "dissipative_lower_bound") + 0.0
        # The bound decides the shift and the kernel grid, so it enters the identifier, as a shortest round-trip JSON
        # number. Without a declaration the identifier hashes the two part identifiers alone.
        identifiers.append(dissipative_lower_bound)
        hashed += " and the declared lower bound of L"
        work_law += "; declared lower bound of L asserted, not verified"
    digest = hashlib.sha256(json.dumps(["cartesian.dissipative-hamiltonian", basis.dimension, ORDER, *identifiers],
                                       separators=(",", ":")).encode())
    sizes = [part.manifest.payload_bytes for part in present]
    # The Hermiticity of a supplied encoding is the caller's statement, so only Pauli and band parts were checked.
    checked = tuple(f"{name} part Hermitian as asserted for its supplied encoding, not checked"
                    if part.manifest.reference.representation == "block_encoding" else f"{name} part exactly Hermitian"
                    for name, part in zip(("dissipative", "hamiltonian"), parts, strict=True) if part is not None)
    manifest = InputManifest(
        reference=InputRef(identity="sha256:" + digest.hexdigest(), representation="cartesian",
                           source=Source(name="nwqlib bounded ingestion", version="1", domain="cartesian",
                                         reference="sha256 of " + hashed)),
        basis=basis,
        identity_status="ingested" if all(part.manifest.identity_status == "ingested" for part in present) else "declared",
        dtype=None, shape=(2,), byte_order=None,
        payload_bytes=None if None in sizes else sum(sizes), access=(),
        checked=(*checked, "parts share one basis"), work_law=work_law)
    return OperatorInput._from_admitted(manifest, "hermitian" if parts[1] is None else "general",
                                        CartesianParts(*parts, dissipative_lower_bound))


def ingest_block_encoding(encoding, *, reference: InputRef, hermitian: bool, lower_bound=None,
                          max_bytes=None) -> OperatorInput:
    """Return an OperatorInput for an operator X given only by a block-encoding circuit you supply.

    The circuit promises `||X - alpha (<0_a| tensor I) U (|0_a> tensor I)|| <= delta`
    with `alpha = encoding.alpha` and `delta = encoding.error_bound`. NWQLib
    computes no content hash of circuit data, so `reference` is a declared
    identifier of X, as for
    [`bind_preparation_circuit`][nwqlib.problems.inputs.bind_preparation_circuit].
    The call copies the circuit and records the assertions without
    simulating the circuit or checking them. A known delta gives
    `||X|| <= alpha + delta`. With an unknown delta, alpha is not a norm
    bound, and an algorithm that needs one refuses the input.

    LCHS evolves a supplied encoding on its `"qsp_block_encoding"` backend,
    which `ham_evo="auto"` chooses for this input. Its
    default coherent estimator evolves a general A (`hermitian=False`) from
    this one circuit and builds no evolution under an encoding asserted
    Hermitian, given as A or as a part of
    [`ingest_cartesian`][nwqlib.operators.inputs.ingest_cartesian]. For a
    positive elapsed time and a nonzero initial state or source, its
    planning raises `ApplicabilityError` for such an encoding. The sampled
    estimator (`LCHS(estimator="sampled_pairs")`) evolves both
    forms. Both estimators refuse a general A, or a Hermitian part used as
    L, whose delta is unknown, because the shift that makes L, the Hermitian
    part of A, positive semidefinite and the kernel grid use the eigenvalue
    bounds of L that delta gives. An unknown delta of a supplied H leaves
    the sampled estimate's bias bound unknown. `hermitian=True` is the
    caller's statement that X is Hermitian. It is not checked and does not
    make the encoded block Hermitian, so the error bounds of a QSP evolution
    under X are conditional on it, and the evolution builders warn that they
    assume it.

    Args:
        encoding (BlockEncoding): The circuit, `alpha`, ancilla count,
            system width and error bound, for example from
            [`build_block_encoding`][nwqlib.subroutines.block_encoding.build_block_encoding].
        reference (InputRef): Declared identifier of X, with
            `representation="block_encoding"`.
        hermitian (bool): Whether X is asserted to be exactly Hermitian.
        lower_bound (float | None): Asserted lower bound on the smallest
            eigenvalue of a Hermitian X, or `None`. It may be negative, which
            does not assert that X has a negative eigenvalue.
        max_bytes (int | None): Limit on the coarse byte estimate of the circuit
            copy, which counts 256 bytes per instruction and qubit of each
            copied circuit plus the parameter data of gates other than
            Qiskit's standard gates. Default `None`, the physical memory of this computer.

    Returns:
        operator (OperatorInput): Representation `"block_encoding"`, with no
            classical access.

    Raises:
        TypeError: If `encoding` is not a `BlockEncoding` or `hermitian` is
            not a bool.
        ValueError: If `reference` does not name a block encoding, the
            circuit width differs from its declared widths or has classical
            bits or free parameters, `lower_bound` is given for a
            non-Hermitian X or is not finite, or the copy exceeds `max_bytes`.
    """
    from dataclasses import replace
    from nwqlib.blocks._qiskit_intake import snapshot_circuit
    from nwqlib.subroutines.block_encoding.core import BlockEncoding

    if type(encoding) is not BlockEncoding:
        raise TypeError("ingest_block_encoding requires a BlockEncoding")
    if type(hermitian) is not bool:
        raise TypeError("hermitian must be a bool")
    if not isinstance(reference, InputRef) or reference.representation != "block_encoding":
        raise ValueError("a supplied block encoding requires a reference with representation 'block_encoding'")
    circuit = encoding.circuit
    if circuit.num_qubits != encoding.num_ancillas + encoding.system_qubits or encoding.system_qubits < 1:
        raise ValueError("block-encoding circuit width must equal its ancillas plus at least one system qubit")
    if circuit.num_clbits or circuit.num_parameters:
        raise ValueError("block-encoding circuit requires no classical bits or free parameters")
    if lower_bound is not None:
        if not hermitian:
            raise ValueError("a lower eigenvalue bound requires an asserted Hermitian operator")
        lower_bound = finite_real(lower_bound, "lower_bound")
    native, data_bytes = snapshot_circuit(circuit, max_bytes=max_bytes)
    manifest = InputManifest(reference=reference, basis=_basis(1 << encoding.system_qubits),
        identity_status="declared", dtype=None, shape=None, byte_order=None, payload_bytes=data_bytes,
        access=(), checked=("width equals ancillas plus system qubits", "no classical bits or free parameters",
                            "finite positive alpha; finite nonnegative or unknown error bound",
                            "Hermitian asserted" if hermitian else "general"),
        work_law="supplied native circuit; coarse snapshot byte estimate; block relation asserted, not checked")
    data = SuppliedBlockEncoding(encoding=replace(encoding, circuit=native, metadata={}),
                                 hermitian=hermitian, lower_bound=lower_bound)
    return OperatorInput._from_admitted(manifest, "hermitian" if hermitian else "general", data)


def _pauli_width(num_qubits):
    return (integer(num_qubits, "num_qubits", 1) + 63) // 64


def _pauli_identity_bytes(num_qubits):
    """Bytes per Pauli term for the content-hash JSON of the Plan records that hold the term.

    A Method that plans with an admitted Pauli operator copies its terms
    into records, and ``Record.content_id`` then holds two JSON copies (the
    text and its UTF-8 bytes) beside the records' own data. Each record's
    content hash is computed on its own, a nested record first, so one copy
    holds a term in its widest single layout J. A per-term record enters its
    parent's JSON as its reference ``{"content_id":"sha256:...","type":...}``,
    145 bytes for an Expectation ``PauliCoefficient``, which with the
    separator and the 4 bytes of list brackets gives 150. The inline layouts
    keep the earlier bound 2q + 120: the label (q + 3 bytes) and coefficient
    (25 bytes) that an Expectation readout spec repeats, QPE's kept label
    (its coefficient array enters by digest), an ADAPT pool term ``["...",c]``
    with an imaginary ``Complex128`` (q + 106 written inline, at most q + 134
    if the coefficient enters by its reference), an ADAPT
    Hamiltonian term with its readout group label (at most
    2q + 94) and the Lanczos label, coefficient and readout row (at most
    q + 55 + digits(2**q - 1)). So J = max(2q + 120, 150), which is 150 for
    q <= 15. The records' own label and coefficient data are at most one
    copy, and each per-term record keeps its computed content hash, a
    71-character string of 112 bytes on CPython 3.12, as long as the Plan.
    The share is therefore 3J + 112 bytes per term. Like the rest of this
    count it covers JSON and data bytes, not Python object overhead.
    Input conversion charges the same share before the Method is chosen, even
    on routes without per-term records. Recheck when a per-term record type's
    name, the reference form or the Python version changes. Plans whose Program also grows with the
    terms (Expectation with sampled counts, FixedGCIM, LCHS, and QLS with a
    ``QuadraticForm`` or ``NormalizedExpectation`` output) are capped earlier
    by ``ir.expressions.AdmissionLimits``. Quantum ADAPT stores its commutators
    as arrays. Their content hashes read the array digests. During construction,
    ``gcim/adapt_inputs.py`` counts the completed arrays alongside the
    workspace for the next commutator.
    """
    return 3 * max(2 * num_qubits + 120, 150) + 112


def _pauli_label_law(num_qubits):
    """Return W, native bytes/term, and conservative label bytes/work per term.

    ``pauli_table`` checks labels against these byte and work formulas.
    Raw conversion without summing identical words can reserve 3L
    separately. These are logical upper bounds on the data bytes, not
    measurements of allocated bytes or resident memory.

    Returns ``(W, L, q + 4L + S, q + W + 1)``. ``W = ceil(q/64)`` uint64 words
    hold one mask. ``L = 16W + 16`` bytes store one term, which is its x and z
    masks and one complex128 coefficient. Each label counts its q
    characters, the four L-byte copies alive at the peak of
    ``_coalesced_input`` and the share S = ``_pauli_identity_bytes(q)`` for
    the content-hash JSON of the Plan records that will hold the term, and
    ``q + W + 1`` work units for the characters, mask words and coefficient.
    """
    num_qubits = integer(num_qubits, "num_qubits", 1)
    w = _pauli_width(num_qubits)
    layout = 16 * w + 16
    return w, layout, num_qubits + 4 * layout + _pauli_identity_bytes(num_qubits), num_qubits + w + 1


def _pauli_label_limit(num_qubits, max_bytes):
    """Check the width metadata bytes and return the largest number of raw terms that ``max_bytes`` allows, before any label is read.

    ``max_bytes=None`` allows the physical memory, and returns None, no count limit, when the platform does not report it.
    """
    _, _, payload_bytes, _ = _pauli_label_law(num_qubits)
    _check_bytes((num_qubits + 8) // 8, max_bytes, "Pauli width metadata")
    limit = memory_limit(max_bytes)
    return None if limit is None else limit // payload_bytes


def pauli_table(terms, *, num_qubits: int, max_bytes=None) -> PauliTerms:
    """Return the raw Pauli table of `(label, coefficient)` pairs, keeping duplicates and zeros.

    The terms keep their order, repeated labels and zero coefficients. Use
    [`ingest_pauli`][nwqlib.operators.inputs.ingest_pauli] for an operator
    with identical labels summed. Labels follow Qiskit order, in which the
    rightmost character acts on qubit 0. The number of terms is checked against
    `max_bytes` before they are read, and an iterator of terms is read at
    most one term past the number that fits (row "Pauli labels" of [Input
    cost controls](../development/input_contracts.md)).

    Args:
        terms (Iterable[tuple[str, complex]]): Pairs of a label of exactly
            `num_qubits` characters from `IXYZ` and a finite real or complex
            coefficient.
        num_qubits (int): Positive number of qubits q.
        max_bytes (int | None): Byte limit. Default `None`, the physical
            memory of this computer.

    Returns:
        table (PauliTerms): Immutable uint64 `x` and `z` masks and complex128
            `coefficients`, one row per term.

    Raises:
        ValueError: If a label has another length or character, a
            coefficient is not finite, or the terms exceed `max_bytes`.
        TypeError: If a coefficient is not a number.
    """
    # Size law. Raw layout L = 16W + 16 bytes per term, W = ceil(q/64). The
    # check reserves R*(q+4L+S) logical bytes. The q label bytes and the four
    # L-byte copies alive at the peak of _coalesced_input cover conversion
    # and the subsequent stable coalescing, and S is the identity JSON share
    # of _pauli_identity_bytes. Python object overhead is not RSS
    # accounting. Work is R*(q+W+1). An unsized stream keeps at most the raw
    # limit plus one lookahead.
    num_qubits = integer(num_qubits, "num_qubits", 1)
    w = _pauli_width(num_qubits)
    raw_limit = _pauli_label_limit(num_qubits, max_bytes)
    per_term = _pauli_label_law(num_qubits)[2]
    if isinstance(terms, Sized) and exceeds(len(terms), raw_limit):
        raise ResourceLimitError(f"raw Pauli input exceeds {byte_allowance(max_bytes)}: its {len(terms)} terms at "
                                 f"{per_term} bytes each need {len(terms) * per_term} bytes")
    xs, zs, coefficients = [], [], []
    for index, item in enumerate(terms):
        if raw_limit is not None and index >= raw_limit:
            raise ResourceLimitError(f"raw Pauli input exceeds {byte_allowance(max_bytes)}: its first {index + 1} "
                                     f"terms at {per_term} bytes each need {(index + 1) * per_term} bytes")
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
    """Return the OperatorInput of a Pauli table after summing the terms with identical words.

    Identical words are summed with stable summation and only an exact-zero
    sum is removed, so no nonzero coefficient is pruned. Canonical Pauli
    words are Hermitian and linearly independent, so the sum is Hermitian
    exactly when every summed coefficient is real. That exact test sets
    ``structure`` without a tolerance.

    With L = 16W + 16 bytes per term, the peak is four L-byte copies alive
    together when the immutable copy is made, namely the input table, the
    summed (masks, value) pairs, the summed arrays and their immutable
    copy. Callers count this as 4L per term.
    """
    q, w = table.num_qubits, table.x.shape[1]
    combined = combine_terms(((x.tobytes(), z.tobytes()), value)
                             for x, z, value in zip(table.x, table.z, table.coefficients, strict=True))
    x, z = np.empty((len(combined), w), dtype=np.uint64), np.empty((len(combined), w), dtype=np.uint64)
    coefficients = np.empty(len(combined), dtype=np.complex128)
    for i, ((xb, zb), value) in enumerate(combined):
        x[i], z[i], coefficients[i] = np.frombuffer(xb, dtype=np.uint64), np.frombuffer(zb, dtype=np.uint64), value
    return _distinct_pauli_input(q, x, z, coefficients)


def _distinct_pauli_input(q, x, z, coefficients):
    """Return the OperatorInput of Pauli rows whose words are distinct and whose coefficients are nonzero.

    The rows are copied byte for byte, so the result has the content hash of
    ingesting the same ordered terms. The caller admits the copy, L = 16W +
    16 bytes per term.
    """
    w = x.shape[1]
    # Test the coefficients before the snapshot copies them byte for byte, so
    # the one-byte-per-term comparison mask is freed before the next copy
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


def ingest_pauli(terms, *, num_qubits: int, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput for the Pauli sum `sum_k c_k P_k` given as `(label, coefficient)` pairs.

    Terms with identical labels are summed with stable summation, and only a
    sum that is exactly zero is removed. No nonzero coefficient is pruned.
    Because distinct Pauli words are Hermitian and linearly independent, the
    operator is Hermitian exactly when every summed coefficient is real, and
    its `structure` records that test without a tolerance. Labels follow
    Qiskit order, in which the rightmost character acts on qubit 0. The byte
    check is that of [`pauli_table`][nwqlib.operators.inputs.pauli_table].

    Args:
        terms (Iterable[tuple[str, complex]]): Pairs of a label of exactly
            `num_qubits` characters from `IXYZ` and a finite real or complex
            coefficient.
        num_qubits (int): Positive number of qubits q.
        max_bytes (int | None): Byte limit. Default `None`, the physical
            memory of this computer.

    Returns:
        operator (OperatorInput): The Pauli operator, with identical words
            summed.

    Raises:
        ValueError: If a label has another length or character, a
            coefficient is not finite, or the terms exceed `max_bytes`.
        TypeError: If a coefficient is not a number.

    Examples:
        Two `ZI` terms add up to one, with coefficient 1.25:

        >>> from nwqlib.operators import ingest_pauli
        >>> H = ingest_pauli([("ZI", 1.0), ("XX", 0.5), ("ZI", 0.25)],
        ...                  num_qubits=2)
        >>> H.pauli_terms().labels()
        (('ZI', (1.25+0j)), ('XX', (0.5+0j)))
        >>> H.structure
        'hermitian'
    """
    return _coalesced_input(pauli_table(terms, num_qubits=num_qubits, max_bytes=max_bytes))


def ingest_pauli_masks(x, z, coefficients, *, num_qubits: int, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput from packed Pauli masks and coefficients.

    Row k with masks `(x, z)` denotes the word `i**popcount(x & z) X**x Z**z`,
    so a qubit with both bits set carries Y, and its coefficient multiplies
    that canonical I/X/Y/Z word. Bit `b` of word `w` refers to qubit
    `64*w + b`. Identical words are summed as in
    [`ingest_pauli`][nwqlib.operators.inputs.ingest_pauli], and the result
    has the same content hash as label input of the same ordered terms. The
    byte check is the row "Pauli masks" of [Input cost
    controls](../development/input_contracts.md).

    Args:
        x (numpy.ndarray): uint64 X masks of shape `(R, W)`, with
            `W = ceil(q/64)`.
        z (numpy.ndarray): uint64 Z masks of the same shape.
        coefficients (numpy.ndarray): complex128 coefficients of shape `(R,)`,
            all finite.
        num_qubits (int): Positive number of qubits q.
        max_bytes (int | None): Byte limit. Default `None`, the physical
            memory of this computer.

    Returns:
        operator (OperatorInput): The Pauli operator, with identical words
            summed.

    Raises:
        TypeError: If an argument is not a NumPy array of the stated dtype.
        ValueError: If the shapes disagree, a bit above qubit q-1 is set, a
            coefficient is not finite, or the arrays exceed `max_bytes`.
    """
    # Size law. Work max(q, R*(W+1)) covers the width metadata even for zero
    # terms. The logical byte bound R*(4(16W+16)+S) beyond caller-owned
    # arrays covers the four L-byte copies alive at the peak of
    # _coalesced_input and the identity JSON share S of
    # _pauli_identity_bytes.
    num_qubits = integer(num_qubits, "num_qubits", 1)
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


def ingest_dense(matrix, *, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput holding an immutable copy of a square dense matrix.

    Integer and real data become float64 and complex data complex128. The
    matrix is neither padded nor symmetrized. `D` is the matrix dimension.
    The call scans the `O(D**2)`
    entries once, checking that they are finite and whether the stored
    matrix equals its conjugate transpose exactly. Those checks use `O(D**2)`
    temporary storage and no eigensolver.

    Args:
        matrix (numpy.ndarray | list | tuple): A nonempty square NumPy array,
            or nested lists or tuples of real or complex numbers.
        max_bytes (int | None): Limit on the bytes of the stored copy. Default `None`,
            the physical memory of this computer.

    Returns:
        operator (OperatorInput): Structure `"hermitian"` or `"general"`.

    Raises:
        TypeError: If `matrix` is another type or holds booleans, objects or
            strings.
        ValueError: If `matrix` is empty, not square, ragged or not finite,
            or its copy exceeds `max_bytes`.
    """
    matrix = _array_input(matrix, ndim=2, max_bytes=max_bytes)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be nonempty and square")
    data = _freeze_numbers(matrix, max_bytes)
    if not np.isfinite(data).all():
        raise ValueError("operator entries must be finite")
    hermitian = np.array_equal(data, data.conj().T)
    manifest = _manifest("dense", data.shape[0], data.shape, str(data.dtype), data.nbytes,
                         _digest("dense", data.shape, (data,)), ("matvec", "entries"),
                         ("finite", "Hermitian" if hermitian else "general"), "O(D²) admission and action")
    return OperatorInput._from_admitted(manifest, "hermitian" if hermitian else "general", data)


def ingest_sparse(matrix, *, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput holding an immutable copy of a SciPy CSR or CSC matrix, kept sparse.

    The matrix must be a `csr_matrix`, `csc_matrix`, `csr_array` or
    `csc_array` in canonical form, with sorted indices and no duplicate
    entries. Canonicalize other storage before the call. The copy keeps the
    CSR or CSC orientation and index dtype. `D` is the matrix dimension.
    Checking finiteness and exact
    equality with the conjugate transpose costs O(nnz + D) work and O(nnz +
    D) temporary sparse storage. The matrix never becomes dense.

    Args:
        matrix (scipy.sparse.csr_matrix | scipy.sparse.csc_matrix): A
            nonempty square canonical CSR or CSC matrix or array of real or
            complex numbers.
        max_bytes (int | None): Limit on the stored values, indices and index
            pointers. Default `None`, the physical memory of this computer.

    Returns:
        operator (OperatorInput): Structure `"hermitian"` or `"general"`.

    Raises:
        TypeError: If `matrix` is another type or holds booleans or objects.
        ValueError: If `matrix` is empty, not square, not canonical or not
            finite, or its copy exceeds `max_bytes`.
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
    arrays = (_freeze_numbers(native_values, max_bytes), _freeze_array(matrix.indices), _freeze_array(matrix.indptr))
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


def operator_input(value, *, max_bytes=None) -> OperatorInput:
    """Return an OperatorInput for a matrix or operator, keeping its representation.

    The call dispatches on the type of `value`:

    - a NumPy array, or nested lists or tuples: [`ingest_dense`][nwqlib.operators.inputs.ingest_dense]
    - a SciPy CSR or CSC matrix or array: [`ingest_sparse`][nwqlib.operators.inputs.ingest_sparse]
    - a [`PeriodicStencil`][nwqlib.operators.inputs.PeriodicStencil]:
      [`ingest_periodic_stencil`][nwqlib.operators.inputs.ingest_periodic_stencil]
    - a Qiskit `SparsePauliOp`: its Pauli terms, summed as in
      [`ingest_pauli`][nwqlib.operators.inputs.ingest_pauli]
    - an `OperatorInput`: returned unchanged

    Qiskit stores each Pauli word as `(-i)**phase` times its label, so the
    coefficient of the canonical I/X/Y/Z label is `coeff * (-i)**phase`.
    Qiskit is imported only for a `SparsePauliOp` argument. Sparse and Pauli
    input is never converted to a dense matrix.

    Args:
        value (object): The matrix or operator.
        max_bytes (int | None): Byte limit of the dispatched function. Default `None`,
            the physical memory of this computer.

    Returns:
        operator (OperatorInput): The operator, ready to use in a Problem.

    Raises:
        TypeError: If `value` has none of the types above, or a dispatched
            function rejects its contents.
        ValueError: If the dispatched function rejects `value`.

    Examples:
        A NumPy matrix stays dense, and a `SparsePauliOp` stays a Pauli sum:

        >>> import numpy as np
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from nwqlib.operators import operator_input
        >>> A = operator_input(np.array([[2.0, 1.0], [1.0, 2.0]]))
        >>> A.manifest.reference.representation, A.structure
        ('dense', 'hermitian')
        >>> A.matvec(np.array([1.0, 1.0]))
        array([3., 3.])
        >>> H = operator_input(SparsePauliOp(["ZI", "XX"], coeffs=[1.0, 0.5]))
        >>> H.manifest.reference.representation, H.basis.dimension
        ('pauli', 4)
        >>> H.pauli_terms().labels()
        (('ZI', (1+0j)), ('XX', (0.5+0j)))
    """
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
