"""Small immutable records shared by the whole library.

They are the Record base, units, scopes, sources, scalars and limits. They
give no access to numerical data and run no scientific computation.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import json
import math
import operator
import sys
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, StrictFloat, StrictInt,
    StringConstraints, computed_field, model_serializer, model_validator,
)
from pydantic_core import core_schema

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
"""A nonempty string.

Surrounding whitespace is removed, and a string that is then empty is
rejected. Only a `str` is accepted, not a number or bytes.
"""
ContentID = Annotated[str, StringConstraints(strict=True)]
"""A content hash, as text.

NWQLib generates `"sha256:"` followed by 64 lowercase hexadecimal digits, and
the type checks only that the value is a `str`.

Every record has one in its `content_id`, computed from its type and fields,
and a record refers to another, such as a Result to its Plan, by this hash.
See [`Record`][nwqlib.core.records.Record].
"""


def _finite_zero(value: float) -> float:
    """Reject nonfinite reals and map -0.0 to 0.0.

    Both zeros compare equal but serialize differently, so without this one
    physical value could carry two content hashes.
    """
    if not math.isfinite(value):
        raise ValueError("numbers must be finite")
    return 0.0 if value == 0 else value


Real = Annotated[StrictFloat, AfterValidator(_finite_zero)]
"""A finite real number, stored as a binary64 `float`.

A Python `float` or `int` is accepted, and an `int` becomes a `float`.
Infinity, NaN, `bool` and strings are rejected. `-0.0` becomes `0.0`, so equal
values give one content hash.
"""
Nonnegative = Annotated[Real, Field(ge=0)]
"""A finite real number at least 0, accepted as for [`Real`][nwqlib.core.records.Real]."""


def _exact_integer(value):
    """Return a NumPy or other exact integer scalar as the Python `int` of the same value.

    Python numbers, strings and bytes are returned unchanged, so StrictInt
    still rejects a `bool`, a `float` and a string. NumPy booleans have no
    `__index__` and stay rejected too.
    """
    if isinstance(value, (bool, int, float, complex, str, bytes)) or not hasattr(type(value), "__index__"):
        return value
    return operator.index(value)


StrictInteger = Annotated[StrictInt, BeforeValidator(_exact_integer)]
"""An exact integer: a Python `int` or a NumPy integer, stored as `int`. A `float` or a `bool` is rejected."""
PositiveInt = Annotated[StrictInteger, Field(gt=0)]
"""A positive exact integer.

A `float` or a `bool` is rejected. A NumPy integer such as `np.int64(1000)` is
accepted and stored as a Python `int`.
"""


def _bytes_backed(array):
    """Whether the memory of ``array`` belongs to an immutable ``bytes`` object."""
    import numpy as np

    base = array
    while isinstance(base, np.ndarray):
        base = base.base
    return type(base) is bytes


# Elements read per tile by _scan_float64, with one Boolean buffer of this length.
_SCAN_TILE = 1 << 16


def _scan_float64(array):
    """Return whether C-order float64 ``array`` holds negative zero, raising if an element is not finite.

    Negative zero is the one float64 whose bits read as the int64 ``-2**63``.
    The scan reads tiles of ``_SCAN_TILE`` elements and reuses one Boolean
    buffer of at most that length, with no full-size temporary.
    """
    import numpy as np

    flat = array.reshape(-1)
    bits = flat.view(np.int64)
    mask = np.empty(min(flat.size, _SCAN_TILE), dtype=bool)
    negative_zero = False
    for start in range(0, flat.size, _SCAN_TILE):
        stop = min(start + _SCAN_TILE, flat.size)
        out = mask[:stop - start]
        if not np.isfinite(flat[start:stop], out=out).all():
            raise ValueError("FrozenArray float64 elements must be finite")
        negative_zero = negative_zero or bool(np.equal(bits[start:stop], -(1 << 63), out=out).any())
    return negative_zero


def _immutable_elements(values, dtype):
    """Return a read-only C-order array of ``values`` in ``dtype`` whose memory is an immutable ``bytes`` object.

    Canonical input, already in ``dtype`` and without float64 negative zero,
    takes one copy, the ``tobytes`` snapshot, which the scan then reads, so a
    concurrent change to the input cannot reach the stored elements. A
    C-order view of a ``bytes`` object in that form is shared with no copy,
    and its own shape and offset are kept. Negative zero or a byte swap
    needs one writable canonical intermediate, made after the snapshot is
    released, and its snapshot, so the input plus two copies at most.
    """
    import numpy as np

    def stored_array(data, shape):
        # A private header above the frombuffer root. NumPy gives every later view that root as its
        # base, so a change to the root's dtype or shape cannot reinterpret the stored array.
        array = np.frombuffer(data, dtype=dtype)
        return array.view() if len(shape) == 1 else array.reshape(shape)

    float64 = dtype.str == "<f8"
    if values.dtype.str == dtype.str:
        if values.flags.c_contiguous and _bytes_backed(values):
            stored = values.view()
        else:
            stored = stored_array(values.tobytes(order="C"), values.shape)
        if not (float64 and _scan_float64(stored)):
            return stored
        # Adding +0.0 maps -0.0 to +0.0 and leaves every other finite value unchanged.
        canonical = stored + 0.0
        del stored
    else:
        canonical = np.array(values, dtype=dtype, order="C", copy=True)
        if float64:
            _scan_float64(canonical)
            canonical += 0.0
    return stored_array(canonical.tobytes(), canonical.shape)


class _Identity:
    """The serialization context of one ``Record.content_id`` computation.

    Under it a record other than ``root`` serializes as its reference
    ``{type, content_id}`` and a FrozenArray in its digest form.
    """

    __slots__ = ("root",)

    def __init__(self, root):
        self.root = root


class FrozenArray:
    """A read-only float64 or int64 array as a Record field value.

    Public construction from a NumPy array stores the elements, in C order
    and little-endian byte order, in an immutable ``bytes`` object, so no
    NumPy or NWQLib API can make them writable, ``setflags`` included.
    Canonical input, little-endian and without float64 negative zero, takes
    the input plus one copy, and canonical input that is already a C-order
    view of a ``bytes`` object, such as decoded JSON data, takes no copy.
    A float64 input that holds negative zero, or an input in big-endian byte
    order, holds the input plus two copies at the same time while it is
    converted. Construction from a FrozenArray shares its storage, and
    unpickling goes through public construction. The one exception is
    storage handed over by library code through
    ``_from_owned_canonical_array``, which that method's docstring
    describes. Float64 elements must be finite and negative zero is
    canonicalized to positive zero, so equal values have equal bytes and
    one content hash. Equality and hashing compare dtype, shape and
    elements.

    The JSON form is an object with three keys: ``dtype`` (``"<f8"`` or
    ``"<i8"``), ``shape`` (a list of nonnegative integers) and ``data``
    (standard base64 of the little-endian C-order element bytes).
    Validation accepts that object, a FrozenArray or a NumPy array of one
    of the two dtypes. Other dtypes are rejected, not cast. The JSON text
    grows with the array, so a record that holds a problem-sized array
    names the size check that bounds it, as other records whose JSON grows
    with the problem do. The content hash of a Record covers the array by
    its digest form, the keys ``dtype`` and ``shape`` with ``sha256``, the
    hexadecimal SHA-256 digest of the same element bytes, read in place
    without encoding or copying them.

    Attributes:
        array: A new read-only view of the stored elements at each access.
        shape: The shape of ``array``.

    Indexing gives what NumPy indexing of ``array`` gives. Basic indexing
    (integers, slices, ``...`` and ``None``) gives a NumPy scalar for a full
    integer index and otherwise a read-only view of the stored elements.
    Integer-array and Boolean indexing give a new writable copy that shares
    no memory with them. Iteration runs over the first axis and yields what
    indexing with each index yields, and ``tolist()`` gives
    ``array.tolist()``. Writing into a returned view raises ValueError.
    """

    __slots__ = ("_array", "_hash")
    _DTYPES = ("<f8", "<i8")

    def __init__(self, values):
        import numpy as np

        self._hash = None
        if isinstance(values, FrozenArray):
            self._array = values._array
            return
        if not isinstance(values, np.ndarray):
            raise ValueError("FrozenArray requires a NumPy float64 or int64 array")
        dtype = values.dtype.newbyteorder("<")
        if dtype.str not in self._DTYPES:
            raise ValueError(f"FrozenArray supports float64 and int64 arrays, not {values.dtype}")
        self._array = _immutable_elements(values, dtype)

    @classmethod
    def _from_owned_canonical_array(cls, values):
        """Take over a private array whose creator has validated its values, without copying it.

        The creator must hold the only writable reference, have filled every
        element, checked float64 finiteness, and changed negative zero to
        positive zero. The result then holds the array alone. The caller must
        discard its array reference and must not have handed out any
        writable view. This path allocates no element buffer or full-array
        mask.

        The elements stay in that NumPy array, which owns its memory, not in
        a ``bytes`` object. It is the ``.base`` of every view of the result,
        ``setflags(write=True)`` on it succeeds, and a later write changes
        the elements while a ``hash()`` or ``content_id`` computed earlier
        stays. Use this route only where the copy into a ``bytes`` object
        would raise the admitted peak of its stage, and never for Result
        data. Its one caller is the angle table of an
        ``LCHSProductFormulaSelectPlan`` built in-process
        (``lchs.time_independent_terms``), whose admission holds 8 bytes
        per angle where protected storage would hold 16 at the handoff. A
        FrozenArray built from that table shares the same storage, and a
        Plan loaded from JSON stores its table through public construction.
        """
        import numpy as np

        if (
            type(values) is not np.ndarray
            or values.dtype.str not in cls._DTYPES
            or not values.flags.c_contiguous
            or not values.flags.owndata
            or not values.flags.writeable
        ):
            raise ValueError(
                "owned canonical storage requires a private C-order "
                "float64 or int64 array"
            )
        result = object.__new__(cls)
        result._hash = None
        values.flags.writeable = False
        result._array = values.view()
        return result

    @property
    def array(self):
        # A new view each time, so a change to its shape or dtype cannot
        # reinterpret the stored elements.
        return self._array.view()

    @property
    def shape(self):
        return self._array.shape

    def tolist(self):
        return self._array.tolist()

    @classmethod
    def from_json(cls, value):
        """Decode the JSON form described in the class docstring."""
        import numpy as np

        if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data"}:
            raise ValueError("FrozenArray JSON requires exactly dtype, shape and data")
        dtype, shape, data = value["dtype"], value["shape"], value["data"]
        if dtype not in cls._DTYPES:
            raise ValueError(f"FrozenArray dtype must be one of {', '.join(cls._DTYPES)}")
        if (not isinstance(shape, (list, tuple))
                or any(type(n) is not int or n < 0 for n in shape) or not isinstance(data, str)):
            raise ValueError("FrozenArray shape must list nonnegative integers and data must be base64 text")
        try:
            raw = base64.b64decode(data.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError) as error:
            raise ValueError("FrozenArray data is not standard base64") from error
        if len(raw) != 8 * math.prod(shape):
            raise ValueError("FrozenArray data length differs from its shape")
        return cls(np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(shape)))

    def _element_bytes(self):
        """Return the stored C-order element bytes as a read-only buffer, without copying them."""
        # A memoryview cast refuses a shape with a zero, so an empty array gives b"".
        return memoryview(self._array).cast("B") if self._array.size else b""

    def to_json(self):
        """Return the JSON form described in the class docstring."""
        return {"dtype": self._array.dtype.str, "shape": list(self._array.shape),
                "data": base64.b64encode(self._element_bytes()).decode("ascii")}

    def _digest(self):
        """Return the digest form that a content hash covers, described in the class docstring."""
        return {"dtype": self._array.dtype.str, "shape": list(self._array.shape),
                "sha256": hashlib.sha256(self._element_bytes()).hexdigest()}

    def _serialize(self, info):
        return self._digest() if type(info.context) is _Identity else self.to_json()

    @classmethod
    def _validate(cls, value):
        return cls.from_json(value) if isinstance(value, Mapping) else cls(value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=True, when_used="json"),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        return {"type": "object", "additionalProperties": False, "required": ["dtype", "shape", "data"],
                "properties": {"dtype": {"enum": list(cls._DTYPES)},
                               "shape": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                               "data": {"type": "string", "contentEncoding": "base64"}}}

    def __array__(self, dtype=None, copy=None):
        if dtype is not None and self._array.dtype != dtype:
            if copy is False:
                raise ValueError("FrozenArray cannot change dtype without a copy")
            return self._array.astype(dtype)
        # Without a copy, a new view, as for `array`, so its dtype or shape can change without the stored one.
        return self._array.copy() if copy else self._array.view()

    def __len__(self):
        return len(self._array)

    def __getitem__(self, key):
        # A view of the read-only array is read-only, and advanced indexing returns a copy.
        return self._array[key]

    def __iter__(self):
        return iter(self._array)

    def __eq__(self, other):
        import numpy as np

        if not isinstance(other, FrozenArray):
            return NotImplemented
        return (self._array.dtype == other._array.dtype and self._array.shape == other._array.shape
                and bool(np.array_equal(self._array, other._array)))

    def __hash__(self):
        if self._hash is None:
            # A digest of the element buffer, which equal arrays share, read without a bytes copy.
            digest = hashlib.blake2b(self._element_bytes()).digest()
            self._hash = hash((self._array.dtype.str, self._array.shape, digest))
        return self._hash

    def __reduce__(self):
        # A new view, so the returned array's header is not the stored one. Unpickling goes through
        # __init__, so the restored elements are stored in a bytes object again.
        return FrozenArray, (self._array.view(),)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f"FrozenArray({self._array!r})"


class Record(BaseModel):
    """The base of every NWQLib record: immutable, built with keyword arguments and validated when built.

    Problems, outputs, Methods, Plans, Results and the other records derive
    from it. Build a record with keyword arguments. Unknown fields are
    rejected, and a record cannot be changed after it is built.
    `record.revise(**changes)` returns a validated new record with the
    changes and leaves the old one unchanged. Constructors, `model_validate`
    and `model_validate_json` validate field data. A field value that is
    already an instance of the field's exact declared record type is kept as
    it is, without validating it again. `model_dump(mode="json")` returns a
    detached JSON description.

    Each record has a content hash, `content_id`. A Result names its Plan,
    an observation its circuit and a resource quantity its evidence by this
    hash, so data stays attached to the record it came from. The hash covers
    the record's concrete type, so two record classes with equal fields stay
    distinct, its schema version, so a format change cannot collide with
    old data, and its `parent_id`, so a revision stays distinguishable from
    a separately built record with the same fields. The content hash is
    computed from the validated fields when first requested.

    Attributes:
        schema_version: Version of the record's format, fixed by its type.
        parent_id: Content hash of the record that this one revises, or
            `None`. `revise` sets it.
        content_id: The record's content hash, read-only: `"sha256:"`
            followed by the hexadecimal SHA-256 digest of compact, sorted-key
            UTF-8 JSON of an object with the keys `"identity_encoding"`
            (`"nwqlib.record/2"`), `"type"` (the record's fully qualified
            type) and `"fields"` (all its declared fields in
            `model_dump(mode="json")` form, with two substitutions). A
            nested record enters as its reference, an object with its fully
            qualified `"type"` and its `"content_id"`, and a `FrozenArray`
            enters with the SHA-256 digest of its elements in place of
            their base64 text. Tuples keep their order, and numbers use
            Python's round-trip binary64 JSON form.
    """

    # Reuse of admitted records. When a field, ``model_validate`` or a revision
    # receives an instance of exactly the declared record type, validation
    # returns that instance unchanged: its validators, including heavy
    # admission such as Program admission, do not run again, and it keeps its
    # stored identity and any other stored admission result. For an instance
    # of the exact type, the wrap validator ``_admit_record`` returns it before
    # any other validator runs.
    # ``revise`` and ``model_copy`` still validate a new record from field
    # values, and so does a field or ``model_validate`` that receives an
    # instance of a subclass of the declared type. An instance of any other
    # record type is rejected, even when its fields fit.
    #
    # Identity cache. The first ``content_id`` read of a record stores the
    # identity in the slot ``_identity_cache``, and later reads return it. An
    # after-validator may read ``content_id`` before a later validator
    # normalizes a declared field, so each validation that builds a new record
    # clears the stored identity after all its validators have run. An instance returned
    # unchanged keeps it. The slot is neither a field nor a Pydantic private
    # attribute, so equality and hashing never see it, and reading
    # ``content_id`` on one of two equal records keeps them equal. Copies and
    # pickles start without it.
    #
    # Identity JSON memory. Computing ``content_id`` holds the JSON text of the
    # identity encoding and its UTF-8 bytes beside the record's own data, since
    # ``model_dump`` shares the scalar objects. Nested records enter by
    # reference, each computing and storing its own identity first, and arrays
    # by a digest of their element buffer, so this JSON grows with the
    # record's own non-array fields only. A record does not know its caller's
    # byte limit, so no check applies here. For each record whose JSON grows
    # with the problem size, ENGINEERING_CONSTANTS.md, "Record identity JSON",
    # names the admission that bounds its data and whether that bound also
    # covers these two JSON copies.
    #
    # Ordinary JSON dumps. ``_serialize_record`` wraps every record's
    # serialization, so Pydantic infers the type of its result once more and
    # copies the containers of each nested record once. For a q = 6 Pauli
    # Expectation Plan this raised the peak per term from 200 to 854 bytes in
    # ``to_record`` and from 654 to 1,684 bytes in ``Result.save``. The copy is
    # transient, and saving counts written file bytes, not Python objects.
    # Revisit when a workflow's save becomes memory bound, or when Pydantic
    # can choose a serializer per call without inferring the result again.

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, revalidate_instances="never",
    )
    __slots__ = ("_identity_cache",)
    schema_version: Literal[1] = 1
    parent_id: ContentID | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        """Wrap each Record type's validator with ``_admit_record``.

        The wrapper reuses exact-type instances and clears the cached content
        hash after validating new field data. Recursive references reach the same
        wrapper, which is installed once for each concrete type.
        """
        schema = handler(source)
        if schema["type"] == "definition-ref" or schema.get("metadata", {}).get("nwqlib.record_admission") is cls:
            return schema
        schema = dict(schema)
        reference = schema.pop("ref", None)
        return core_schema.no_info_wrap_validator_function(
            cls._admit_record, schema, ref=reference, metadata={"nwqlib.record_admission": cls},
        )

    @classmethod
    def _admit_record(cls, value, handler):
        """Validate field data and reuse instances of the exact declared type.

        An instance of exactly this type is returned unchanged and no
        validator runs again, as the comment "Reuse of admitted records" in
        the class body describes. Every Record type's validator is wrapped
        here, so the path covers each type without a per-class guard.
        """
        if type(value) is cls:
            return value
        if isinstance(value, cls):
            # A subclass instance given for a base-typed field is validated
            # from its field values, so its undeclared fields reject as they
            # do for a mapping. Any other record type reaches the model
            # validator unchanged and is rejected as the wrong type.
            value = {name: getattr(value, name) for name in type(value).model_fields}
        if isinstance(value, Mapping):
            value = dict(value)
            value.pop("content_id", None)
        record = handler(value)
        if record is not value:
            # A validator may have read content_id before a later validator
            # normalized a field. An instance returned unchanged was admitted
            # before, and its stored identity stays.
            object.__setattr__(record, "_identity_cache", None)
        return record

    def _reference(self):
        """Return ``{type, content_id}``, the form in which another record's content hash covers this record."""
        return {"type": _type_name(type(self)), "content_id": self.content_id}

    def _identity_reference(self, info):
        """Return the reference while ``info`` serializes another record's content hash, else None.

        Field serializers that write a nested record themselves call it too.
        """
        context = info.context
        return self._reference() if type(context) is _Identity and context.root is not self else None

    @model_serializer(mode="wrap")
    def _serialize_record(self, handler, info):
        reference = self._identity_reference(info)
        return handler(self) if reference is None else reference

    @computed_field
    @property
    def content_id(self) -> str:
        # SHA-256 of the identity encoding in the class docstring, stored on
        # first read. This is a comment because Pydantic copies a computed
        # field's docstring into every record's serialization schema.
        try:
            identity = object.__getattribute__(self, "_identity_cache")
        except AttributeError:
            identity = None
        if identity is None:
            payload = {
                "identity_encoding": "nwqlib.record/2",
                "type": _type_name(type(self)),
                "fields": self.model_dump(mode="json", exclude_computed_fields=True, context=_Identity(self),
                                          fallback=_identity_fallback),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
            identity = "sha256:" + hashlib.sha256(encoded).hexdigest()
            object.__setattr__(self, "_identity_cache", identity)
        return identity

    def revise(self, **changes) -> Self:
        """Return a validated copy with `changes`, recording this record as its parent.

        The new record's `parent_id` is this record's content hash, and this
        record is unchanged. Problems, outputs and `Accuracy` return a copy
        without a parent link, because a changed input is a new input with
        its own content hash.

        Args:
            **changes (object): New field values, by field name. The parent
                link is derived from this record.

        Returns:
            record (Record): The new record, of the same type.

        Raises:
            ValueError: If the new values fail validation.

        Examples:
            >>> from nwqlib.core import Unit
            >>> hartree = Unit(symbol="Ha", dimension="energy")
            >>> revised = hartree.revise(symbol="Hartree")
            >>> print(revised.symbol, revised.parent_id == hartree.content_id)
            Hartree True
        """
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(changes)
        values["parent_id"] = self.content_id
        return type(self).model_validate(values)

    def model_copy(self, *, update=None, deep=False) -> Self:
        """Copy immutable fields, or validate updates as an explicit revision."""
        if update:
            return self.revise(**update)
        return type(self).model_validate({name: getattr(self, name) for name in type(self).model_fields})

    def __dir__(self):
        # Tab completion reads dir(): it lists the fields and NWQLib's own
        # methods, not the rest of pydantic's BaseModel API (_PYDANTIC_NAMES).
        return [name for name in object.__dir__(self)
                if name not in _PYDANTIC_NAMES or name in type(self).model_fields]


# Names of pydantic's BaseModel API, which tab completion on a record leaves
# out so that the scientific fields stay visible. The Record docstring points
# users to model_dump and model_dump_json, so those two stay.
_PYDANTIC_NAMES = frozenset(name for name in dir(BaseModel) if not name.startswith("__")) - {
    "model_dump", "model_dump_json"}


class _ConstructorDisplay:
    """Display a configuration record as the constructor call that a user writes for it.

    Methods and the option records that their fields take inherit it. The
    display names the type and the fields whose values differ from their
    defaults, such as `LCHS(ham_evo=Trotter(steps=4))`. It omits revision
    ancestry (`parent_id`) from every displayed record. State and operator
    inputs can appear as summaries in angle brackets. A display can be
    evaluated directly only when all its arguments are complete Python
    expressions. Reconstruction from the display need not preserve revision
    ancestry or content identity. `repr`, `str` and IPython show the same
    text, which `_constructor_text` describes. `model_dump()` returns every
    field in full.
    """

    __slots__ = ()

    def __repr__(self):
        return _constructor_text(self)

    def __str__(self):
        return repr(self)

    def _repr_pretty_(self, printer, cycle):
        printer.text(f"{type(self).__name__}(...)" if cycle else repr(self))

    def _argument(self):
        """Return the text of this value as an argument inside another record's constructor display."""
        return repr(self)


# The record's format version and revision link, which a user does not write.
_NOT_DISPLAYED = frozenset({"schema_version", "parent_id"})


@functools.cache
def _display_defaults(cls):
    """Return each optional field's default as a built record stores it.

    A default given as a name that validation replaces by its record, such as
    LCHS's `"coherent"` by `Coherent()`, is compared in the stored form. A type
    with required fields, or whose construction from its defaults raises any
    exception, has no default record, and its declared defaults are compared
    as written. A Method written outside the library can raise a TypeError
    there, and repr and print must still work.
    """
    try:
        built = cls()
    except Exception:
        return {name: info.get_default(call_default_factory=True)
                for name, info in cls.model_fields.items() if not info.is_required()}
    return {name: getattr(built, name) for name in cls.model_fields}


def _constructor_text(record):
    """Return ``Type(name=value, ...)`` with the fields of ``record`` that differ from their defaults.

    Required fields always appear, `schema_version` and `parent_id` never, and
    a fixed `kind` equals its default. Each value appears as `_argument_text`
    gives it.
    """
    defaults = _display_defaults(type(record))
    arguments = []
    for name in type(record).model_fields:
        value = getattr(record, name)
        if name in _NOT_DISPLAYED or (
                name in defaults and type(value) is type(defaults[name]) and value == defaults[name]):
            continue
        arguments.append(f"{name}={_argument_text(value)}")
    return f"{type(record).__name__}({', '.join(arguments)})"


def _argument_text(value):
    """Return the text of ``value`` as a constructor argument, as a user would write it.

    Scalars, strings and tuples of them appear as Python expressions.
    Configuration records use their own displays, which can contain data
    summaries and omit revision ancestry. A record may shorten its argument
    form (`_ConstructorDisplay._argument`), as `ProviderConfig` does to its
    name. Any other value, such as an operator, a state or a sequence of
    them, appears as a summary in angle brackets (`_data_summary`), without
    its numbers.
    """
    if isinstance(value, _ConstructorDisplay):
        return value._argument()
    return repr(value) if _expression(value) else f"<{_data_summary(value)}>"


def _expression(value):
    """Whether ``value`` uses ``repr`` instead of a data summary in a constructor display."""
    return (value is None or type(value) in (str, int, float, bool) or isinstance(value, _ConstructorDisplay)
            or (type(value) is tuple and all(_expression(item) for item in value)))


def _data_summary(value):
    """Return the short text of a data value in a constructor display, reading only its metadata.

    An operator or state input appears as in a Problem's display, such as
    `dense 2x2`, an array by its shape, a tuple by its length and its first
    three entries, and any other object by its type name.
    """
    from nwqlib.operators.inputs import OperatorInput
    from nwqlib.problems.inputs import StateInput
    from nwqlib.problems.records import _field_summary
    if isinstance(value, (OperatorInput, StateInput)):
        return _field_summary(value)
    if isinstance(value, FrozenArray):
        return f"array of shape {value.array.shape}"
    if type(value) is tuple:
        shown = [repr(item) if _expression(item) else _data_summary(item) for item in value[:3]]
        return (f"{len(value)} {'entry' if len(value) == 1 else 'entries'}: {'; '.join(shown)}"
                f"{'; ...' if len(value) > 3 else ''}")
    return type(value).__name__


@functools.cache
def _type_name(cls):
    # One string per record type, shared by all references to its records.
    return f"{cls.__module__}.{cls.__qualname__}"


def _identity_fallback(value):
    # An array inside an Any value, which type inference does not serialize, by its digest.
    if isinstance(value, FrozenArray):
        return value._digest()
    raise TypeError(f"a content hash cannot encode a value of type {type(value).__name__}")


def _declared_fields_equal(record, other):
    """Compare two records of one exact type by their declared fields only.

    A record class whose private attributes cache derived values assigns this
    function as its ``__eq__``, so a cache filled on one of two equal records
    keeps them equal. For an operand of another type it returns
    ``NotImplemented``, so Python can try the other operand's comparison.
    """
    if type(other) is not type(record):
        return NotImplemented
    return all(getattr(record, name) == getattr(other, name) for name in type(record).model_fields)


class Unit(Record):
    """A unit label with its dimension. NWQLib converts no units and does no unit algebra.

    Build it with keyword arguments, for example
    `Unit(symbol="Hartree", dimension="energy")`. Both arguments are
    required. A Problem's `unit` also accepts a string, which becomes a
    `Unit` of dimension `"custom"`.

    Attributes:
        symbol: Required. The unit's symbol, a nonempty string.
        dimension: Required. `"dimensionless"`, `"energy"`, `"time"`,
            `"inverse_time"`, `"angle"`, `"bytes"`, `"count"`, `"currency"`
            or `"custom"`.
    """

    symbol: Text
    dimension: Literal["dimensionless", "energy", "time", "inverse_time", "angle", "bytes", "count", "currency", "custom"]

    def same_unit(self, other: Unit) -> bool:
        """Return whether `other` has the same symbol and dimension, whatever the revision history of either.

        Args:
            other (Unit): The unit to compare with.

        Returns:
            same (bool): `True` when symbol and dimension are equal.
        """
        return (self.symbol, self.dimension) == (other.symbol, other.dimension)


class Scope(Record):
    """The algebraic object, or the declared physical or model domain, that a value refers to.

    Build it with keyword arguments, for example
    `Scope(kind="model", domain="1D heat equation on 64 grid points")`.
    `domain` is required.

    Attributes:
        kind: Default `"algebraic"`. `"algebraic"`, `"physical"` or `"model"`.
        domain: Required. Description of the object or domain.
    """

    kind: Literal["algebraic", "physical", "model"] = "algebraic"
    domain: Text


class Source(Record):
    """The name, version, domain and reference of a piece of code, data or method.

    It declares where something came from. It is not evidence that the
    source was verified. Build it with keyword arguments. All four arguments
    are required.

    Attributes:
        name: Required. Name of the source.
        version: Required. Its version.
        domain: Required. What it applies to.
        reference: Required. Where to find it, such as a paper identifier, a
            URL or a function name.
    """

    name: Text
    version: Text
    domain: Text
    reference: Text


class Float64(Record):
    """A finite binary64 number, as a record.

    Build it with keyword arguments, `Float64(value=0.5)`. Its decimal form
    is the float's value and does not claim that the number is exact.

    Attributes:
        kind: Fixed `"float64"`.
        value: Required. The number, accepted as for
            [`Real`][nwqlib.core.records.Real].
    """

    kind: Literal["float64"] = "float64"
    value: Real


class Complex128(Record):
    """A complex number with finite binary64 parts, as a record.

    A part given as `-0.0` is stored as `0.0`.

    Attributes:
        kind: Fixed `"complex128"`.
        real: Required. Real part.
        imag: Required. Imaginary part.
    """

    kind: Literal["complex128"] = "complex128"
    real: Real
    imag: Real


class Rational(Record):
    """An exact rational number `numerator / denominator`.

    Build it with keyword arguments, for example
    `Rational(numerator=2, denominator=6)`. Construction reduces it to
    lowest terms with a positive denominator, here `1/3`, so each rational
    number has one representation and one content hash.

    Attributes:
        kind: Fixed `"rational"`.
        numerator: Required. Integer numerator. After reduction it carries
            the sign of the number.
        denominator: Required. Nonzero integer denominator, positive after
            reduction.

    Raises:
        ValueError: If `denominator` is zero, or if a saved record cannot
            hold the reduced numerator or denominator: the numerator must lie
            above -10**4299 and below 10**4300, and the positive denominator
            below 10**4300.
    """

    kind: Literal["rational"] = "rational"
    numerator: StrictInteger
    denominator: StrictInteger

    @model_validator(mode="after")
    def _reduce(self):
        if self.denominator == 0:
            raise ValueError("rational denominator cannot be zero")
        # gcd is nonnegative and gcd(0, d) = |d|, so zero becomes 0/1, and the
        # sign moves to the numerator. Both divisions are exact.
        divisor = math.gcd(self.numerator, self.denominator)
        sign = -1 if self.denominator < 0 else 1
        object.__setattr__(self, "numerator", sign * self.numerator // divisor)
        object.__setattr__(self, "denominator", sign * self.denominator // divisor)
        if not (storable_integer(self.numerator) and storable_integer(self.denominator)):
            raise ValueError("a saved record holds rational numerators above -10**4299 and below 10**4300, "
                             "and denominators below 10**4300")
        return self


# A record writes an integer as a decimal JSON number. CPython's int() reads at
# most sys.int_info.default_max_str_digits (4300) digits, sign excluded, and the
# pydantic-core reader behind model_validate_json reads a number of at most 4300
# characters, sign included. A record stores the integers that both read: at
# most 4300 characters in decimal, -10**4299 < value < 10**4300. The default
# rather than this process's setting keeps every record readable in a process
# with default settings. The positive bound has 14285 bits.
STORABLE_INTEGER_BOUND = 10 ** sys.int_info.default_max_str_digits
_NEGATIVE_STORABLE_BOUND = STORABLE_INTEGER_BOUND // 10


def storable_integer(value):
    """Return whether a record can store the integer ``value``, ``-10**4299 < value < 10**4300``."""
    return -_NEGATIVE_STORABLE_BOUND < value < STORABLE_INTEGER_BOUND


def storable_power(coefficient, exponent):
    """Return ``coefficient * 2**exponent`` when a record can store it, else None.

    A value with more bits than ``STORABLE_INTEGER_BOUND`` is never formed.
    """
    if coefficient.bit_length() + exponent > STORABLE_INTEGER_BOUND.bit_length():
        return None
    value = coefficient << exponent
    return value if storable_integer(value) else None


Scalar = Annotated[Float64 | Complex128 | Rational, Field(discriminator="kind")]


class Symbol(Record):
    """A named mathematical symbol and the source that defines it.

    It is a name only and is never evaluated as an expression.

    Attributes:
        name: Required. A letter followed by letters, digits or underscores.
        reference: Required. The [`Source`][nwqlib.core.records.Source] that
            defines the symbol.
    """

    name: Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
    reference: Source


class InputRef(Record):
    """A declaration of an input and of how it is accessed, without its data.

    It names the input and its representation. It does not build an oracle
    or a circuit for the input. `Eigenproblem.subspace` takes one to name an
    explicitly defined subspace.

    Attributes:
        identity: Required. Name of the input.
        representation: Required. How the input is represented, for example
            `"circuit"`.
        source: Required. The [`Source`][nwqlib.core.records.Source] of the
            input.
        schema_version: Fixed `2`. Version of the saved record format.
    """

    schema_version: Literal[2] = 2
    identity: Text
    representation: Text
    source: Source


class Basis(Record):
    """The coordinate basis of an input: its name, its dimension and the order of its entries and bits.

    A Problem reports the basis of its inputs in `basis`, and all inputs of
    one Problem must have the same basis.

    Attributes:
        identity: Required. Name of the basis, such as `"computational"`.
        dimension: Required. Number of coordinates, a positive integer.
        ordering: Required. Order of the entries and bits, for example
            `"coordinate index increasing; qubit 0 is rightmost tensor/bit position"`.
    """

    identity: Text
    dimension: PositiveInt
    ordering: Text


Stage = Literal["planning", "preparation", "execution", "analysis", "verification"]


class Limit(Record):
    """A declared limit on one quantity at one stage of the workflow.

    Declaring a limit neither approves nor runs any work. `kind` separates a
    budget that is used up (`"consumption"`, such as CPU seconds, shots or
    currency), a capacity that is reused (`"capacity_stock"`, such as memory
    or stored bytes) and a `"deadline"`, because they combine differently.
    A budget adds up across work, a capacity is compared with a peak, and a
    deadline bounds the elapsed time. Each metric accepts one unit and one
    kind, so a limit cannot be read in the wrong sense:

    - `cpu_time` is a budget and `wall_time` a deadline, both in
      `Unit(symbol="s", dimension="time")`.
    - `memory` and `stored` are capacities, in
      `Unit(symbol="byte", dimension="bytes")`.
    - `materialization` and `transfer` are budgets, in the same byte unit.
    - `evaluations`, `shots`, `jobs`, `host_invocations` and `host_work`
      are budgets, in `Unit(symbol="count", dimension="count")`.
    - `currency` is a budget, in any unit of dimension `"currency"`.

    Attributes:
        stage: Required. Workflow stage the limit applies to:
            `"planning"`, `"preparation"`, `"execution"`, `"analysis"` or
            `"verification"`.
        metric: Required. Limited quantity: `"cpu_time"`, `"wall_time"`,
            `"memory"`, `"materialization"`, `"stored"`, `"transfer"`,
            `"evaluations"`, `"shots"`, `"jobs"`, `"currency"`,
            `"host_invocations"` or `"host_work"`.
        unit: Required. Unit, which must match the metric.
        kind: Required. `"consumption"`, `"capacity_stock"` or `"deadline"`,
            which must match the metric.
        value: Required. Nonnegative limit value. Byte and count metrics
            require an exact integer.
        scope: Required. What the limit applies to.

    Raises:
        ValueError: If the unit or kind does not match the metric, or a byte
            or count limit is not an integer.
    """

    stage: Stage
    metric: Literal["cpu_time", "wall_time", "memory", "materialization", "stored", "transfer", "evaluations", "shots", "jobs", "currency", "host_invocations", "host_work"]
    unit: Unit
    kind: Literal["consumption", "capacity_stock", "deadline"]
    value: Annotated[StrictInteger, Field(ge=0)] | Nonnegative
    scope: Text

    @model_validator(mode="after")
    def _metric_unit(self):
        """Require the one (unit dimension, unit symbol, kind) triple that each metric accepts, as listed in the class docstring.

        A currency limit accepts any currency symbol, and byte and count
        values must be exact integers.
        """
        expected = {
            "cpu_time": ("time", "s", "consumption"),
            "wall_time": ("time", "s", "deadline"),
            "memory": ("bytes", "byte", "capacity_stock"),
            "materialization": ("bytes", "byte", "consumption"),
            "stored": ("bytes", "byte", "capacity_stock"),
            "transfer": ("bytes", "byte", "consumption"),
            "evaluations": ("count", "count", "consumption"),
            "shots": ("count", "count", "consumption"),
            "jobs": ("count", "count", "consumption"),
            "host_invocations": ("count", "count", "consumption"),
            "host_work": ("count", "count", "consumption"),
            "currency": ("currency", self.unit.symbol, "consumption"),
        }[self.metric]
        if (self.unit.dimension, self.unit.symbol, self.kind) != expected:
            raise ValueError("metric/unit/limit kind mismatch")
        if self.unit.dimension in {"bytes", "count"} and not isinstance(self.value, int):
            raise ValueError("byte/count limits require exact integers")
        return self
