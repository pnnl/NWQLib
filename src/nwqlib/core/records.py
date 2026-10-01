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
ContentID = Annotated[str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]


def _finite_zero(value: float) -> float:
    """Reject nonfinite reals and map -0.0 to 0.0.

    Both zeros compare equal but serialize differently, so without this one
    physical value could carry two content identities.
    """
    if not math.isfinite(value):
        raise ValueError("numbers must be finite")
    return 0.0 if value == 0 else value


Real = Annotated[StrictFloat, AfterValidator(_finite_zero)]
Nonnegative = Annotated[Real, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


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
    """Validated fields determine identity; JSON exports are detached snapshots.

    ``revise`` and ``model_copy(update=...)`` validate changes and link the old
    identity. Pydantic's unsafe ``model_construct`` and legacy ``copy`` are
    deliberately unavailable.
    Identity encoding v1 is UTF-8, sorted-key compact JSON with the fully
    qualified record type and all declared fields (including schema/parent).
    Tuples keep their order. Numbers use Python's round-trippable binary64 JSON.

    Content identity is how NWQLib associates data with the selection that
    produced it. A Result names its Plan, an observation its realization, a
    resource quantity its evidence. It therefore covers everything that
    changes meaning. The concrete type is included, so two record classes
    with equal fields stay distinct. The schema version is included, so a
    format change cannot collide with old data. The parent revision is
    included, so a revision stays distinguishable from an independently
    built record with the same fields. A supplied ``content_id`` is checked
    on load, so edited JSON cannot keep a stale identity. Records are frozen,
    and every construction or load from field values or JSON runs the full
    validation.

    An admitted record is reused, not validated again. When a field,
    ``model_validate`` or a revision receives an instance of exactly the
    declared record type, validation returns that instance unchanged: its
    validators, including heavy admission such as Program admission, do not
    run again, and it keeps its stored identity and any other stored
    admission result. Records are frozen and every construction path runs
    the full validation, so an instance of the exact type is an admitted one,
    and the wrap validator ``_check_identity`` returns it before any other
    validator runs. ``revise`` and ``model_copy`` still validate a new record
    from field values, and so does a field or ``model_validate`` that
    receives an instance of a subclass of the declared type. An instance of
    any other record type is rejected, even when its fields fit.

    The first ``content_id`` read of a record stores the identity in the
    slot ``_identity_cache``, and later reads return it. An after-validator
    may read ``content_id`` before a later validator normalizes a declared
    field, so each validation that builds a new record clears the stored
    identity after all its validators have run and before a supplied
    ``content_id`` is checked; an admitted instance returned unchanged keeps
    it. The slot is neither a field nor a Pydantic private attribute, so
    equality and hashing never see it, and reading ``content_id`` on one of
    two equal records keeps them equal. Copies and pickles start without it.

    Computing ``content_id`` holds the JSON text of the identity encoding and
    its UTF-8 bytes beside the record's own data, since ``model_dump`` shares
    the scalar objects. A record does not know its caller's byte limit, so
    no check applies here. For each record whose JSON grows with the problem
    size, ENGINEERING_CONSTANTS.md, "Record identity JSON", names the
    admission that bounds its data and whether that bound also covers these
    two JSON copies.
    """

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
        """Validate a new revision while keeping the original record."""
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
    """A declared unit label, without conversion or unit algebra."""

    symbol: Text
    dimension: Literal["dimensionless", "energy", "time", "inverse_time", "angle", "bytes", "count", "currency", "custom"]

    def same_unit(self, other: Unit) -> bool:
        """Compare declared unit meaning independently of record ancestry."""
        return (self.symbol, self.dimension) == (other.symbol, other.dimension)


class Scope(Record):
    """The algebraic object or explicitly declared physical/model domain."""

    kind: Literal["algebraic", "physical", "model"] = "algebraic"
    domain: Text


class Source(Record):
    """A source/version/domain declaration, not a verification receipt."""

    name: Text
    version: Text
    domain: Text
    reference: Text


class Float64(Record):
    """Finite binary64 value; decimal spelling is not an exact-number claim."""

    kind: Literal["float64"] = "float64"
    value: Real


class Complex128(Record):
    """Two finite binary64 components, with canonical positive zeros."""

    kind: Literal["complex128"] = "complex128"
    real: Real
    imag: Real


class Rational(Record):
    """Exact integer ratio in lowest terms with a positive denominator.

    Reduction gives each rational number one representation, and hence one
    content identity.
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
    """An inert named mathematical symbol; no expression evaluation."""

    name: Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
    reference: Source


class InputRef(Record):
    """An immutable input/access declaration; it does not prepare an oracle."""

    schema_version: Literal[2] = 2
    identity: Text
    representation: Text
    source: Source


class Basis(Record):
    """Declared coordinate basis and entry/bit ordering."""

    identity: Text
    dimension: PositiveInt
    ordering: Text


Stage = Literal["planning", "preparation", "execution", "analysis", "verification"]


class Limit(Record):
    """A stage-specific limit declaration. Admission neither approves nor runs work.

    ``kind`` separates a consumable budget (CPU seconds, shots, currency), a
    capacity that is reused (memory, stored bytes) and a deadline, because
    they combine differently. Consumption adds across work, capacity is
    compared with a peak, and a deadline bounds elapsed time. The validator
    fixes one unit and kind per metric so a limit cannot be read in the
    wrong sense.

    Attributes:
        stage: Workflow phase the limit governs.
        metric: Limited quantity.
        unit: Declared unit, which must match the metric.
        kind: ``consumption``, ``capacity_stock`` or ``deadline``, which must match the metric.
        value: Limit value. Byte and count metrics require an exact integer.
        scope: What the limit applies to.
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
