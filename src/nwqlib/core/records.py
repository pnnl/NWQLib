"""Small immutable interchange records; no payload access or scientific execution."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from itertools import islice
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StrictFloat, StrictInt,
    StringConstraints, computed_field, model_validator,
)
from pydantic_core import core_schema

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
"""A nonempty string.

Surrounding whitespace is removed, and a string that is then empty is
rejected. Only a `str` is accepted, not a number or bytes.
"""
ContentID = Annotated[str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
"""A content hash: `"sha256:"` followed by 64 lowercase hexadecimal digits.

Every record has one in its `content_id`, computed from its type and fields,
and a record refers to another, such as a Result to its Plan, by this hash.
See [`Record`][nwqlib.core.records.Record].
"""


def _finite_zero(value: float) -> float:
    """Reject nonfinite reals and map -0.0 to 0.0.

    Both zeros compare equal but serialize differently, so without this one
    physical value could carry two content identities.
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
PositiveInt = Annotated[StrictInt, Field(gt=0)]
"""A positive Python `int`.

A `float`, a `bool` or a NumPy integer is rejected. Convert it with `int(...)`.
"""


class FrozenArray:
    """A read-only float64 or int64 array as a Record field value.

    Public construction from a NumPy array copies it once, in C order and
    little-endian byte order. Construction from a FrozenArray shares its
    read-only storage. Internal producers can transfer a private canonical
    buffer through ``_from_owned_canonical_array`` under that method's
    ownership and validation preconditions. Float64 elements must be finite
    and negative zero is canonicalized to positive zero, so equal values
    have equal bytes and one content identity. Equality and hashing compare
    dtype, shape and elements.

    The JSON form, which the identity encoding of a Record hashes, is an
    object with three keys: ``dtype`` (``"<f8"`` or ``"<i8"``), ``shape`` (a
    list of nonnegative integers) and ``data`` (standard base64 of the
    little-endian C-order element bytes). Validation accepts that object, a
    FrozenArray or a NumPy array of one of the two dtypes; other dtypes are
    rejected rather than cast. The JSON text grows with the array, so a
    record that stores a problem-sized array names the admission that bounds
    it, as for other growing record JSON.

    Attributes:
        array: The read-only NumPy array.
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
        array = np.array(values, dtype=dtype, order="C", copy=True)
        if dtype.str == "<f8":
            if not np.isfinite(array).all():
                raise ValueError("FrozenArray float64 elements must be finite")
            # Adding +0.0 maps -0.0 to +0.0 and leaves every other finite value unchanged.
            array += 0.0
        # A view of a read-only array cannot be made writeable again.
        array.flags.writeable = False
        self._array = array.view()

    @classmethod
    def _from_owned_canonical_array(cls, values):
        """Adopt a private array whose producer has validated its values.

        The producer must own the only writable reference, have filled every
        element, checked float64 finiteness, and changed negative zero to
        positive zero. Ownership passes to the result. The caller must
        discard its array reference and must not have published any writable
        view. This path allocates no element buffer or full-array mask.
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
        return self._array

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

    def to_json(self):
        """Return the JSON form described in the class docstring."""
        return {"dtype": self._array.dtype.str, "shape": list(self._array.shape),
                "data": base64.b64encode(self._array.tobytes()).decode("ascii")}

    @classmethod
    def _validate(cls, value):
        return cls.from_json(value) if isinstance(value, Mapping) else cls(value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.to_json(), when_used="json"),
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
        return self._array.copy() if copy else self._array

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
            self._hash = hash((self._array.dtype.str, self._array.shape, self._array.tobytes()))
        return self._hash

    def __reduce__(self):
        # Unpickling goes through __init__, so the restored array is read-only again.
        return FrozenArray, (self._array,)

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
    changes and leaves the old one unchanged. Every construction, including
    `model_validate` and `model_validate_json` from saved JSON, runs the
    full validation, and `model_dump(mode="json")` returns a detached JSON
    description. Unchecked `model_construct` and Pydantic's legacy `copy`
    are disabled.

    Each record has a content hash, `content_id`. A Result names its Plan,
    an observation its circuit and a resource quantity its evidence by this
    hash, so data stays attached to the record it came from. The hash covers
    the record's concrete type, so two record classes with equal fields stay
    distinct, its schema version, so a format change cannot collide with
    old data, and its `parent_id`, so a revision stays distinguishable from
    a separately built record with the same fields. A `content_id` given in
    saved JSON is checked when the record is loaded, so edited JSON cannot
    keep a stale hash.

    Attributes:
        schema_version: Version of the record's format, fixed by its type.
        parent_id: Content hash of the record that this one revises, or
            `None`. `revise` sets it.
        content_id: The record's content hash, read-only: the SHA-256 digest
            of compact, sorted-key UTF-8 JSON of the record's fully qualified
            type and all its declared fields. Tuples keep their order, and
            numbers use Python's round-trip binary64 JSON form.
    """

    # Reuse of admitted records. When a field, ``model_validate`` or a revision
    # receives an instance of exactly the declared record type, validation
    # returns that instance unchanged: its validators, including heavy
    # admission such as Program admission, do not run again, and it keeps its
    # stored identity and any other stored admission result. Records are frozen
    # and every construction path runs the full validation, so an instance of
    # the exact type is an admitted one, and the wrap validator
    # ``_check_identity`` returns it before any other validator runs.
    # ``revise`` and ``model_copy`` still validate a new record from field
    # values, and so does a field or ``model_validate`` that receives an
    # instance of a subclass of the declared type. An instance of any other
    # record type is rejected, even when its fields fit.
    #
    # Identity cache. The first ``content_id`` read of a record stores the
    # identity in the slot ``_identity_cache``, and later reads return it. An
    # after-validator may read ``content_id`` before a later validator
    # normalizes a declared field, so each validation that builds a new record
    # clears the stored identity after all its validators have run and before a
    # supplied ``content_id`` is checked; an admitted instance returned
    # unchanged keeps it. The slot is neither a field nor a Pydantic private
    # attribute, so equality and hashing never see it, and reading
    # ``content_id`` on one of two equal records keeps them equal. Copies and
    # pickles start without it.
    #
    # Identity JSON memory. Computing ``content_id`` holds the JSON text of the
    # identity encoding and its UTF-8 bytes beside the record's own data, since
    # ``model_dump`` shares the scalar objects. A record does not know its
    # caller's byte limit, so no check applies here. For each record whose JSON
    # grows with the problem size, ENGINEERING_CONSTANTS.md, "Record identity
    # JSON", names the admission that bounds its data and whether that bound
    # also covers these two JSON copies.

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True, revalidate_instances="never",
    )
    __slots__ = ("_identity_cache",)
    schema_version: Literal[1] = 1
    parent_id: ContentID | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        """Wrap each Record type's validator with ``_check_identity``.

        Pydantic calls this hook when it builds a validator, including for a
        Record nested in another field, so every construction and load path
        checks unknown keys and a supplied ``content_id``. The schema's
        ``ref`` moves to the wrapper, so a recursive reference also reaches
        the check. A bare reference, or a schema already wrapped for this
        class, is returned unchanged, so the check runs once.
        """
        schema = handler(source)
        if schema["type"] == "definition-ref" or schema.get("metadata", {}).get("nwqlib.final_identity") is cls:
            return schema
        schema = dict(schema)
        reference = schema.pop("ref", None)
        return core_schema.no_info_wrap_validator_function(
            cls._check_identity, schema, ref=reference, metadata={"nwqlib.final_identity": cls},
        )

    @classmethod
    def _check_identity(cls, value, handler):
        """Validate supplied data and require any supplied content_id to match the admitted record.

        An instance of exactly this type was admitted when it was built, so it
        is returned unchanged and no validator runs again (the admitted-instance
        path of the class docstring). Every Record type's validator is wrapped
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
        supplied = None
        if isinstance(value, Mapping):
            value = dict(value)
            unknown = value.keys() - cls.model_fields.keys() - {"content_id"}
            if unknown:
                # Keys explain admission; arbitrary supplied values are never
                # coerced or represented to format this error.
                keys = sorted(repr(key[:80]) if type(key) is str else f"<{type(key).__name__} key>"
                              for key in islice(unknown, 8))
                fields = list(islice(cls.model_fields, 16))
                raise ValueError(f"undeclared record fields for {cls.__name__}: {', '.join(keys)}"
                                 + (f" ({len(unknown)-8} more)" if len(unknown)>8 else "")
                                 + f"; declared fields: {', '.join(fields)}"
                                 + (f" ({len(cls.model_fields)-16} more)" if len(cls.model_fields)>16 else ""))
        if isinstance(value, dict) and "schema_version" in value:
            if type(value["schema_version"]) is not int:
                raise ValueError("schema_version must be an integer")
        if isinstance(value, dict) and "content_id" in value:
            value = dict(value)
            supplied = value.pop("content_id")
            if not isinstance(supplied, str):
                raise ValueError("content_id must be a sha256 identity")
        record = handler(value)
        if record is not value:
            # A validator may have read content_id before a later validator
            # normalized a field. An instance returned unchanged was admitted
            # before, and its stored identity stays.
            object.__setattr__(record, "_identity_cache", None)
        if supplied is not None and supplied != record.content_id:
            raise ValueError("content_id does not match admitted content")
        return record

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
                "identity_encoding": "nwqlib.record/1",
                "type": f"{type(self).__module__}.{type(self).__qualname__}",
                "fields": self.model_dump(mode="json", exclude_computed_fields=True),
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
            **changes (object): New field values, by field name. `content_id` and
                `parent_id` cannot be given.

        Returns:
            record (Record): The new record, of the same type.

        Raises:
            ValueError: If `content_id` or `parent_id` is given, or the new
                values fail validation.

        Examples:
            >>> from nwqlib.core import Unit
            >>> hartree = Unit(symbol="Ha", dimension="energy")
            >>> revised = hartree.revise(symbol="Hartree")
            >>> print(revised.symbol, revised.parent_id == hartree.content_id)
            Hartree True
        """
        if "content_id" in changes or "parent_id" in changes:
            raise ValueError("revision identity and parent are derived, not caller updates")
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(changes)
        values["parent_id"] = self.content_id
        return type(self).model_validate(values)

    def model_copy(self, *, update=None, deep=False) -> Self:
        """Copy immutable fields, or validate updates as an explicit revision."""
        if update:
            return self.revise(**update)
        return type(self).model_validate({name: getattr(self, name) for name in type(self).model_fields})

    def copy(self, *, include=None, exclude=None, update=None, deep=False):
        """Reject Pydantic's legacy unchecked copy operation."""
        raise TypeError("legacy copy is unavailable; use revise or model_copy for validated records")

    @classmethod
    def model_construct(cls, _fields_set=None, **values):
        raise TypeError("use model_validate for public records")


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
        ValueError: If `denominator` is zero.
    """

    kind: Literal["rational"] = "rational"
    numerator: StrictInt
    denominator: StrictInt

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
        return self


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
    Use adds up across work, a capacity is compared with a peak, and a
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
    value: Annotated[StrictInt, Field(ge=0)] | Nonnegative
    scope: Text

    @model_validator(mode="after")
    def _metric_unit(self):
        """Require the one (unit dimension, unit symbol, kind) triple each metric admits.

        Both times are in seconds. CPU time is a consumption and wall time a
        deadline. Memory and stored bytes are capacities, and the other byte
        and count metrics are consumptions. A currency limit accepts any
        currency symbol. Byte and count values must be exact integers.
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
