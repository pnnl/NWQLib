"""Immutable numerical output and its actual acquisition provenance."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from contextlib import closing
from dataclasses import dataclass, field
from hashlib import sha256
from math import prod
from threading import RLock
from typing import Literal

from pydantic import model_validator

from nwqlib.core.records import Basis, ContentID, Record, Source, Text
from nwqlib.operators.access import Count


# Little-endian C-order storage of each admitted ArrayOutput dtype: the
# manifest encoding, the NumPy dtype string and the bytes per entry.
_ARRAY_ENCODINGS = {"complex128": ("complex128-le-c", "<c16", 16), "float64": ("float64-le-c", "<f8", 8)}

# PayloadRef formats of published arrays, amplitude and readout arrays alike.
# A Run charges these payloads through its array byte counter, not again as
# other binary payloads.
ARRAY_PAYLOAD_FORMATS = frozenset(f"nwqlib.array/{dtype}" for dtype in ("<c16", "<f8", "<u8"))

# Entries per block of the finiteness check at publication, a 64 KiB Boolean
# mask. Registered in docs/ENGINEERING_CONSTANTS.md ("Local journal scalar admission").
_FINITE_CHECK_ENTRIES = 1 << 16


def _check_finite_blocks(array):
    """Reject a nonfinite entry, reading the array in blocks of leading-axis slices.

    Each block holds at most ``_FINITE_CHECK_ENTRIES`` entries, or one slice
    along the leading axis when that slice is longer, so the Boolean mask
    ``np.isfinite`` allocates stays that size instead of the array's.
    """
    import numpy as np
    rows = array.shape[0] if array.ndim else 1
    per_row = array.size // rows if rows else 0
    step = max(1, _FINITE_CHECK_ENTRIES // max(1, per_row))
    for start in range(0, rows, step):
        if not np.isfinite(array[start:start + step]).all():
            raise ValueError("artifact payload must be finite")


class ArrayOutput(Record):
    """One requested complex128 or float64 array in its original coordinates and frame.

    A complex128 vector may be declared as a unit direction or modulo a global
    phase. A float64 array holds real values in its coordinates. It admits the
    ``physical`` or ``unit`` frame for a vector and the ``physical`` frame for
    an operator, and only ``global_phase="physical"``: its signs are stored
    values, and ``modulo_global_phase`` describes the phase freedom of a
    complex state direction, which a real array does not declare.

    Attributes:
        name: Unique output name within its selected kernel/readout.
        kind: vector or operator, determining one or two basis-sized axes.
        basis: Original coordinate basis and dimension.
        frame: physical magnitude or normalized unit-vector frame.
        global_phase: physical phase or explicitly modulo_global_phase vector meaning.
        dtype: complex128 or float64 numerical output representation.
    """

    name: Text
    kind: Literal["vector", "operator"]
    basis: Basis
    frame: Literal["physical", "unit"]
    global_phase: Literal["physical", "modulo_global_phase"]
    dtype: Literal["complex128", "float64"] = "complex128"

    @model_validator(mode="after")
    def _operator_frame(self):
        """Require the physical frame and phase for an operator output, and a physical phase for a real array.

        A vector output may be declared as a unit direction or modulo a global
        phase, because a normalized state direction is a meaningful quantity by
        itself. An operator's scale and phase are part of its matrix action, so
        an operator output keeps both.
        """
        if self.kind == "operator" and (self.frame != "physical" or self.global_phase != "physical"):
            raise ValueError("operators must keep their physical action and phase")
        if self.dtype == "float64" and self.global_phase != "physical":
            raise ValueError("a float64 array stores its signs and declares global_phase='physical'")
        return self

    @property
    def shape(self):
        """``(D,)`` for a vector or ``(D, D)`` for an operator, with D the basis dimension."""
        return (self.basis.dimension,) * (2 if self.kind == "operator" else 1)

    @property
    def encoding(self):
        """Persistence encoding: ``complex128-le-c`` or ``float64-le-c``."""
        return _ARRAY_ENCODINGS[self.dtype][0]

    @property
    def storage_dtype(self):
        """Little-endian NumPy dtype string of the stored payload, ``<c16`` or ``<f8``."""
        return _ARRAY_ENCODINGS[self.dtype][1]

    @property
    def data_bytes(self):
        """Payload size in bytes, 16 per complex128 entry and 8 per float64 entry."""
        return _ARRAY_ENCODINGS[self.dtype][2] * prod(self.shape)

    def validate_array(self, array):
        """Check shape and the declared complex128 or float64 representation before reading the payload."""
        import numpy as np
        kind, itemsize = ("c", 16) if self.dtype == "complex128" else ("f", 8)
        if (type(array) is not np.ndarray or array.shape != self.shape
                or array.dtype.kind != kind or array.dtype.itemsize != itemsize):
            raise ValueError("native output shape/dtype differs from its selected declaration")


# Little-endian C-order storage of each readout-array component: its dtype,
# the manifest encoding and the NumPy dtype string. Every entry has 8 bytes.
_READOUT_COMPONENTS = {"probabilities": ("float64", "float64-le-c", "<f8"),
                       "indices": ("uint64", "uint64-le-c", "<u8")}


class ReadoutArray(Record):
    """One component array of a probability readout, in C order.

    It is not an ``ArrayOutput``: a probability array has no amplitude's
    unit-vector or global-phase meaning. Let k be the number of observed
    qubits, D = 2^k and s the number of nonzero returned probabilities.
    Outcome integer j denotes the bit string ``format(j, f"0{k}b")``, and
    ``observation.qubits[0]`` supplies its least significant bit. The
    components are

    - ``probabilities``: float64. Dense, of shape ``(D,)``: every outcome
      exists by position, including exact zeros. Sparse, of shape ``(s,)``:
      the values, strictly positive after canonical removal of exact zeros.
    - ``indices``: uint64 outcome indices of a sparse marginal, strictly
      increasing, of shape ``(s,)`` for a width of at most 64 bits, or
      ``(s, W)`` with ``W = ceil(width/64)`` words, word zero least
      significant, compared and sorted most-significant word first, with the
      padding bits of the last word zero.

    A length can be zero (an empty sparse return). The two dtypes of a sparse
    readout stay separate arrays: a homogeneous two-column float array loses
    integer indices or counts above 2**53. The manifest records the actual
    encoding, shape, dtype, byte count, payload digest and acquisition
    provenance of each component.

    Attributes:
        component: probabilities or indices.
        dtype: float64 or uint64, the one the component selects.
        shape: One axis, or ``(entries, words)`` for packed indices.
    """

    component: Literal["probabilities", "indices"]
    dtype: Literal["float64", "uint64"]
    shape: tuple[Count, ...]

    @model_validator(mode="after")
    def _layout(self):
        """Require the component's dtype, one axis, or two for packed indices with at least two words."""
        if (self.dtype != _READOUT_COMPONENTS[self.component][0]
                or len(self.shape) not in ((1, 2) if self.component == "indices" else (1,))
                or (len(self.shape) == 2 and self.shape[1] < 2)):
            raise ValueError("a readout array has its component's dtype and one axis, or (entries, words) "
                             "packed indices with at least two words")
        return self

    @property
    def encoding(self):
        """Persistence encoding: ``float64-le-c`` or ``uint64-le-c``."""
        return _READOUT_COMPONENTS[self.component][1]

    @property
    def storage_dtype(self):
        """Little-endian NumPy dtype string of the stored payload: ``<f8`` or ``<u8``."""
        return _READOUT_COMPONENTS[self.component][2]

    @property
    def data_bytes(self):
        """Payload size in bytes, 8 per entry."""
        return 8 * prod(self.shape)

    def validate_array(self, array):
        """Check the shape and the declared 8-byte representation before reading the payload."""
        import numpy as np
        if (type(array) is not np.ndarray or array.shape != self.shape
                or array.dtype.kind != {"float64": "f", "uint64": "u"}[self.dtype]
                or array.dtype.itemsize != 8):
            raise ValueError("readout array shape/dtype differs from its declaration")


def probability_readout(values, indices=None, *, width):
    """Check one probability marginal at publication and choose its stored encoding.

    Publication owns the value checks. Loading checks the declared layout and
    associations without rereading the values. ``values`` is float64 and
    either dense (``indices`` None, shape ``(2**width,)``, outcome j at
    position j) or sparse with ``indices`` of shape ``(m,)`` for a width of at
    most 64 or ``(m, W)`` packed words with ``W = ceil(width/64)``. The checks
    are:

    - Type and layout: exact supported dtype, shape, contiguity and endian
      conversion before publication, O(1) headers and O(m) only when a
      conversion copy is required.
    - Finite probabilities: blockwise ``np.isfinite(values).all()``, O(m)
      work and O(C) Boolean scratch for block length C.
    - Nonnegative probabilities: blockwise ``values >= 0``, before canonical
      zero removal.
    - Unique legal keys: dense positions give uniqueness. Sparse input is
      sorted, then ``index[1:] > index[:-1]`` and a width/domain check. This
      is O(m) for sorted input and O(m log m) with an O(m) index workspace to
      sort unsorted input. For width 64 every uint64 index is in range, and
      for a width below 64 the indices are compared with ``np.uint64(1 <<
      width)``. Cardinality and duplicates are tested before zero removal.

    Sparse storage is chosen when ``8*(W+1)*s < 8*2**width``, which for a
    single word is ``2*s < D``, otherwise dense. Both represent the exact same
    binary64 entries, with zero values implicit in the sparse form. This is
    the raw-byte crossover ``B_dense = 8D``, ``B_sparse = 16s`` (``8*(W+1)*s``
    with W index words), not an empirical speed threshold. A dense input keeps
    its buffer on the dense branch and forms indices only on the sparse
    branch. The adaptive policy guarantees ``B_actual <= 8D``. The stored-entry
    count D of a dense marginal and its nonzero count s are different
    quantities.

    The total is ``math.fsum`` of the stored values, computed once here and
    saved, so the unit endpoint ``fsum(p_j) <= fl(1+w)`` of
    ``ObservationChunk.validate_unit_bound`` is the exact acceptance
    predicate. ``np.sum`` with the same comparison could change acceptance for
    a total at the endpoint. It is O(m) scalar conversion work, and it creates
    no per-bin records. Zero-width probabilities stay invalid.

    Returns:
        ``(indices, values, summary)``: the stored uint64 indices (None when
        dense) and float64 values, both little-endian and C-contiguous, and a
        mapping with ``encoding`` (dense or sparse), ``entries`` (stored
        entries), ``nonzero``, ``mass`` (the ``math.fsum`` total) and
        ``maximum``.
    """
    import numpy as np
    from itertools import chain
    from math import fsum

    if type(width) is not int or width < 1:
        raise ValueError("a probability readout observes at least one qubit")
    words = -(-width // 64)
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind != "f" or values.dtype.itemsize != 8:
        raise ValueError("probabilities are one float64 value per entry")
    values = np.ascontiguousarray(values, dtype="<f8")
    step = _FINITE_CHECK_ENTRIES
    for start in range(0, len(values), step):
        block = values[start:start + step]
        if not np.isfinite(block).all():
            raise ValueError("probabilities must be finite")
        if (block < 0).any():
            raise ValueError("probabilities must be nonnegative")
    if indices is None:
        if len(values) != 1 << width:
            raise ValueError(f"a dense probability readout of width {width} holds 2**{width} values")
        nonzero = int(np.count_nonzero(values))
        if (words + 1) * nonzero < len(values):
            positions = np.flatnonzero(values)
            indices, values = positions.view(np.uint64), values[positions]
            del positions
    else:
        indices = np.asarray(indices)
        packed = words > 1
        if (indices.dtype.kind not in "iu" or indices.dtype.itemsize != 8 or indices.shape[0:1] != values.shape
                or indices.shape[1:] != ((words,) if packed else ())):
            raise ValueError(f"indices of a width-{width} readout are one 64-bit integer per value, or "
                             f"(entries, {words}) packed words above width 64")
        if indices.dtype.kind == "i" and (indices < 0).any():
            raise ValueError("outcome indices are nonnegative")
        indices = np.ascontiguousarray(indices, dtype="<u8")
        # The most significant word holds width - 64*(W-1) bits. Set bits
        # above them are out of range (padding bits must be zero).
        top = width - 64 * (words - 1)
        high = indices[:, -1] if packed else indices
        if top < 64 and len(high) and (high >= np.uint64(1 << top)).any():
            raise ValueError(f"an outcome index exceeds the readout width {width}")
        if packed:
            # np.lexsort's last key is the primary one, so the most
            # significant word (the last column) orders the rows first.
            order = np.lexsort(indices.T)
            if (np.diff(order) != 1).any():
                indices, values = indices[order], values[order]
            if len(indices) > 1 and np.all(indices[1:] == indices[:-1], axis=1).any():
                raise ValueError("duplicate outcome indices in one probability readout")
        else:
            if len(indices) > 1 and not (indices[1:] > indices[:-1]).all():
                order = np.argsort(indices, kind="stable")
                indices, values = indices[order], values[order]
                if not (indices[1:] > indices[:-1]).all():
                    raise ValueError("duplicate outcome indices in one probability readout")
        keep = values != 0
        if not keep.all():
            indices, values = indices[keep], values[keep]
        del keep
        nonzero = len(values)
        if not packed and (words + 1) * nonzero >= 1 << width:
            dense = np.zeros(1 << width, dtype="<f8")
            dense[indices] = values
            indices, values = None, dense
    total = fsum(chain.from_iterable(values[start:start + step].tolist() for start in range(0, len(values), step)))
    summary = dict(encoding="dense" if indices is None else "sparse", entries=len(values), nonzero=nonzero,
                   mass=total, maximum=float(values.max(initial=0.0)))
    return indices, values, summary


class ArtifactManifest(Record):
    """The description of a saved array: what it is, where it came from and its content hash.

    A Result names its arrays by their manifests, for example
    `LCHSAnalysis.artifact`, and `result.data.artifact(manifest)` returns
    the array's handle. A manifest gives no access to the array itself. The
    fields below are read-only.

    Attributes:
        plan_id: Content hash of the Plan.
        realization_id: Content hash of the concrete parameter values of the
            experiment.
        construction_id: Content hash of the construction that produced the
            output, or for a readout array the preparation record of its
            measurement.
        producer_id: Content hash of the classical computation, or of the
            readout description, that produced the array.
        output: The array's shape, basis and normalization and phase
            convention (`ArrayOutput`), or the component of a readout array
            (`ReadoutArray`).
        acquisition: Four text labels of the measurement that produced the
            array: the Run's `run_id`, the attempt, the job and the result or
            chunk within that job. For a classical computation, the job label
            repeats the attempt.
        source: The code and method that produced the array (`Source`).
        digest: Content hash of the saved numerical bytes. It does not prove
            that the values are scientifically correct.
        data_bytes: Size of the array data in bytes, which matches the
            output's shape and dtype.
        encoding: Element type, little-endian byte order and C (row-major)
            layout of the saved data: `"complex128-le-c"`, `"float64-le-c"`
            or `"uint64-le-c"`, as the output's dtype selects.
    """

    schema_version: Literal[4] = 4
    plan_id: ContentID
    realization_id: ContentID
    construction_id: ContentID
    producer_id: ContentID
    output: ArrayOutput | ReadoutArray
    acquisition: tuple[Text, Text, Text, Text]
    source: Source
    digest: ContentID
    data_bytes: Count
    encoding: Literal["complex128-le-c", "float64-le-c", "uint64-le-c"] = "complex128-le-c"

    @model_validator(mode="after")
    def _provenance(self):
        """Require the byte count and encoding implied by the output shape and dtype."""
        if self.data_bytes != self.output.data_bytes or self.encoding != self.output.encoding:
            raise ValueError("artifact bytes disagree with the selected array shape/dtype")
        return self


class UnavailableOutput(Record):
    """A declared array output that its producer did not produce, and the reason.

    An observation lists it in place of the array, so a missing output is
    explicit instead of looking like an empty result.

    Attributes:
        output: Selected named array declaration that was not produced.
        reason: Nonempty explanation of why that output is unavailable.
    """

    output: ArrayOutput
    reason: Text


@dataclass(frozen=True, eq=False, init=False, slots=True)
class ArtifactHandle:
    """A saved array with its manifest.

    `result.data.artifact(manifest)` and
    `result.data.artifacts` give handles, and `handle.array` reads the
    values as a read-only NumPy array. A handle cannot be built directly. A
    handle of a reopened Run or loaded Result reads its array from the saved
    data on first access. The fields below are read-only.

    Attributes:
        manifest: The array's
            [`ArtifactManifest`][nwqlib.artifacts.ArtifactManifest]: content
            hash, output convention and the measurement that produced it.
    """

    manifest: ArtifactManifest
    _array: object = field(repr=False)
    _store: object = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("artifact handles require publication through their live store")

    @property
    def array(self):
        """The array, as a read-only NumPy array.

        For an array in memory, reading it copies and computes nothing. For a
        reopened Run, the array is read from the saved data once, at the first
        access. For a loaded Result, it is a read-only memory map of the saved
        file, so its values are read from disk as they are used.

        Raises:
            ValueError: If the array data is not available.
        """
        if self._array is None and self._store is not None:
            self._store.get(self.manifest)
        if self._array is None:
            raise ValueError("this artifact description has no available array data")
        return self._array

    @property
    def available(self):
        """Whether the array's values can be read, checked without reading them."""
        return self._array is not None or (self._store is not None and self._store.available(self.manifest))


class ArtifactStore:
    """One finite numerical-data lifetime, shared by execution and analysis.

    The data-byte cap bounds stored arrays. Publication can use at most two
    array-sized temporary copies, for endian or layout conversion and for
    immutable ownership, and none for a handed-over private array
    (``_publish``). That finite size follows from the admitted array, not a
    metadata work ledger. The cap does not claim to bound process RSS or all
    undocumented SDK workspace.

    A restored inventory is registered lazily: a manifest costs O(1)
    metadata and zero hydrated payload bytes until its array is first read
    (``get``). Hydration keeps one copy per payload digest for the store's
    lifetime. Lazy loading does not imply eviction, so with already hydrated
    payloads totalling H the store's payload peak is H + B + C for a new
    payload of B bytes read in journal blocks of C bytes.
    """

    def __init__(self, *, max_bytes=DEFAULT_MAX_BYTES):
        """Create an empty store with a cap of ``max_bytes`` array bytes.

        ``_persistence`` is None for a standalone store. A Run installs its
        reservation and publication callbacks there, so each array commits in
        the same journal transaction as the observation that produced it.
        ``_loader`` is None for a standalone store. A Run installs a callable
        that returns ``(read_chunks, chunk_bytes)`` for a manifest whose bytes
        its journal holds, or None, so a restored array hydrates on first use.
        """
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("artifact storage requires a finite nonnegative byte cap")
        self._cap = max_bytes
        self._data_bytes = 0
        self._handles = {}
        self._lock = RLock()
        self._persistence = None
        self._loader = None
        self._hydrated = {}

    def _handle(self, manifest, array):
        """A new handle of this store for ``manifest``, holding ``array`` or None until it is read."""
        handle = object.__new__(ArtifactHandle)
        object.__setattr__(handle, "manifest", manifest)
        object.__setattr__(handle, "_array", array)
        object.__setattr__(handle, "_store", self)
        return handle

    def _restore(self, manifests):
        """Restore the charged inventory without inventing missing native data.

        Each manifest is registered without reading its payload. The array is
        read on first use through ``_loader`` (``get``).
        """
        with self._lock:
            for manifest in manifests:
                self.check_capacity(manifest.data_bytes)
                self._handles[manifest.content_id] = self._handle(manifest, None)
                self._data_bytes += manifest.data_bytes

    def __copy__(self):
        raise TypeError("copying a live artifact store cannot reset its data cap")

    def __deepcopy__(self, memo):
        raise TypeError("copying a live artifact store cannot reset its data cap")

    @property
    def data_bytes(self):
        return self._data_bytes

    def snapshot(self):
        """Freeze the inventory as handle references, without copying any array."""
        with self._lock:
            return tuple(self._handles.values())

    def _set_limit(self, max_bytes):
        """Apply an already recorded Run limit amendment without resetting usage."""
        with self._lock:
            if type(max_bytes) is not int or max_bytes < self._cap or max_bytes < self._data_bytes:
                raise ValueError("artifact data cap can only increase and must cover existing data")
            self._cap = max_bytes

    def get(self, manifest):
        """Return the handle for ``manifest`` when its array is available, hydrating it on first use.

        Lookup needs the whole manifest, not only its identity, and the stored
        manifest must equal it. A description cannot reach data published
        under different provenance. A restored manifest whose bytes the
        store's loader can read is hydrated once (``_hydrate``).
        """
        if type(manifest) is not ArtifactManifest:
            raise TypeError("artifact access requires its full provenance-qualified manifest")
        with self._lock:
            handle = self._handles.get(manifest.content_id)
            if handle is None:
                raise ValueError("this store has no data associated with the supplied manifest")
            if handle._array is None:
                source = None if self._loader is None else self._loader(manifest)
                if manifest.digest not in self._hydrated and source is None:
                    raise ValueError("this artifact description has no available array data")
                read_chunks, chunk_bytes = source or (None, 16)
                self._hydrate(manifest, read_chunks, chunk_bytes=chunk_bytes)
            return handle

    def available(self, manifest):
        """Whether the store can supply array data for ``manifest``, not only its description.

        A payload is accessible when the store holds it or its loader can
        read it. Answering reads no payload bytes.
        """
        if type(manifest) is not ArtifactManifest:
            raise TypeError("artifact access requires its full manifest")
        with self._lock:
            handle = self._handles.get(manifest.content_id)
            if handle is None:
                raise ValueError("this store has no inventory for the supplied manifest")
            return (handle._array is not None or manifest.digest in self._hydrated
                    or (self._loader is not None and self._loader(manifest) is not None))

    def _hydrate(self, manifest, read_chunks, *, chunk_bytes):
        """Materialize one registered payload on first access into a single private buffer.

        The stored dtype, shape, byte count and block order determine its
        layout. The returned array exposes a read-only view. With previous
        hydrated arrays excluded, the explicit payload-buffer peak is the
        payload byte count plus one journal block when both the consumer and
        the iterator release each block before reading the next. Stream the
        block inventory or account for its separate metadata storage.

        One private byte buffer of size B is allocated, each journal block is
        copied directly into a memoryview of it and released before the next
        block is read, and the final NumPy view is created over a read-only
        memoryview, so the explicit peak payload memory of one hydration is
        B + C for blocks of C bytes (16N + C for N complex128 entries).
        Loading is trusted: the shape, encoding, block order, block sizes and
        completeness are checked, and no value scan, digest recomputation or
        summary recheck runs. Publication checked the values and computed the
        summaries from the bytes that were hashed. ``manifest.data_bytes``
        stays B. The Run charges stored bytes, not this transient workspace.
        The buffer is kept per payload digest for the store's lifetime, so a
        second restored manifest with the same bytes, of the same or another
        dtype and shape, gets its own typed view of that buffer without a
        second read.
        """
        import numpy as np
        if type(chunk_bytes) is not int or chunk_bytes <= 0 or chunk_bytes % 16:
            raise ValueError("array chunk_bytes must be a positive multiple of 16")
        output = manifest.output
        with self._lock:
            handle = self._handles.get(manifest.content_id)
            if handle is None:
                raise ValueError("this store has no inventory for the supplied manifest")
            if handle._array is not None:
                return handle
            buffer = self._hydrated.get(manifest.digest)
            if buffer is None or len(buffer) != manifest.data_bytes:
                size = manifest.data_bytes
                if size > self._cap:
                    raise ValueError("saved array exceeds the admitted data cap")
                count = (size + chunk_bytes - 1) // chunk_bytes
                data = bytearray(size)
                target = memoryview(data)
                offset = index = 0
                with closing(read_chunks()) as chunks:
                    # The block is released before the iterator is advanced,
                    # so at most one block is alive beside the buffer.
                    for item in chunks:
                        ordinal, block = item
                        del item
                        expected = min(chunk_bytes, size - offset)
                        if (type(ordinal) is not int or ordinal != index or type(block) is not bytes
                                or expected <= 0 or len(block) != expected):
                            raise ValueError("saved artifact has missing, reordered or incorrectly sized chunks")
                        target[offset:offset + expected] = block
                        del block
                        offset += expected
                        index += 1
                target.release()
                if offset != size or index != count:
                    raise ValueError("saved artifact payload is incomplete")
                buffer = memoryview(data).toreadonly()
                del data
                self._hydrated[manifest.digest] = buffer
            array = np.frombuffer(buffer, dtype=output.storage_dtype).reshape(output.shape)
            object.__setattr__(handle, "_array", array)
            return handle

    def check_capacity(self, data_bytes):
        """Reject ``data_bytes`` more array bytes when stored plus new would exceed the cap."""
        if type(data_bytes) is not int or data_bytes < 0:
            raise ValueError("data capacity requires nonnegative integer bytes")
        if self._data_bytes + data_bytes > self._cap:
            raise ValueError(f"array publication needs {data_bytes} bytes with {self._data_bytes} already stored; limit is {self._cap}")

    def _publish(self, array, *, output, provenance, private=False):
        """Publish one declared array as immutable little-endian data.

        An ``ArrayOutput`` is complex128 or float64. A ``ReadoutArray`` is
        float64 or uint64 and has had its value checks
        (``probability_readout``) before it arrives here. Shape and dtype
        are checked before any copy. Nonfinite values of an
        ``ArrayOutput`` reject. The check reads the array in blocks of at most
        ``_FINITE_CHECK_ENTRIES`` entries (one operator row when a row is
        longer), so its Boolean scratch does not grow with the array. A Run
        reserves the bytes against the acquisition's output reservation before
        the copy. The array is then converted to the declared little-endian
        dtype (``<c16``, ``<f8`` or ``<u8``) and frozen, and only then
        hashed, so the digest and the manifest identity do not depend on the
        host byte order. A manifest already in the store (same bytes and same
        acquisition) returns the existing handle without a second charge.

        Ownership: with ``private=True`` the caller hands over an array that no
        other code references and gives up its own reference. When that array
        is the only NumPy array over its C-contiguous data in the stored dtype
        (it owns the data, or its base is a non-array buffer owner such as a
        simulator's result buffer), publication marks it read-only and keeps it
        instead of copying it, and the published handle holds a view of it
        that cannot be made writeable. The kept array itself is not copied into
        immutable bytes, so code that reached it through the view's ``base``
        could make it writeable again. Publication relies on the handover
        instead. Any other array is copied into immutable bytes, after a
        conversion copy when its byte order or layout differs from the stored
        one, so later changes by the caller cannot reach the published data.
        ``reduce_amplitudes`` returns a new array that nothing else holds, and
        amplitude publication hands it over. A probability readout hands over
        the arrays that ``probability_readout`` returns, the adapter's dense
        buffer included. Points of one trajectory that share one saved marginal
        hand over the same buffer, which stays read-only and is published once
        per point under that point's manifest.
        """
        import numpy as np
        from nwqlib.operators.inputs import _freeze_array

        output.validate_array(array)
        with self._lock:
            self.check_capacity(array.nbytes)
            if array.dtype.kind in "fc" and type(output) is not ReadoutArray:
                # A readout array's values were checked by probability_readout.
                _check_finite_blocks(array)
            reservation = None if self._persistence is None else self._persistence[0](array.nbytes, provenance)
            dtype = np.dtype(output.storage_dtype)
            if (private and not isinstance(array.base, np.ndarray) and array.flags.c_contiguous
                    and array.dtype == dtype):
                array.flags.writeable = False
                owned = array.view()
            else:
                owned = _freeze_array(np.asarray(array, dtype=dtype))
            digest = "sha256:" + sha256(memoryview(owned).cast("B")).hexdigest()
            manifest = ArtifactManifest(output=output, digest=digest, data_bytes=owned.nbytes,
                                        encoding=output.encoding, **provenance)
            prior = self._handles.get(manifest.content_id)
            if self._persistence is not None:
                self._persistence[1](reservation, manifest, owned, duplicate=prior is not None)
            if prior is not None:
                if prior._array is None:
                    object.__setattr__(prior, "_array", owned)
                return prior
            handle = self._handle(manifest, owned)
            self._handles[manifest.content_id] = handle
            self._data_bytes += owned.nbytes
            return handle
