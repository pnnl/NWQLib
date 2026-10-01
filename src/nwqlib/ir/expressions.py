"""Finite typed expression definitions; exact integer work, never source evaluation."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from nwqlib.core.records import Float64, PositiveInt, Record, Text

NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
Value = StrictInt | Float64


def walk_kept(value, reserve):
    """Yield every Record, tuple and scalar reachable from value, charging each collection first.

    ``reserve(n)`` is called with a Record's field count or a tuple's length
    before its children are pushed on the stack, so a ``reserve`` that
    raises rejects an over-budget collection by its length before it is
    iterated or copied. Program admission (``ir/validation.py::_Admission``)
    instead only counts in ``reserve`` and compares the total after the walk,
    so its refusal reports the complete stored field inventory. A local ID
    string is yielded as data and never resolved, so a shared definition is
    not expanded.
    """
    reserve(1)
    stack = [value]
    while stack:
        value = stack.pop()
        yield value
        if isinstance(value, Record):
            fields = type(value).model_fields
            reserve(len(fields))
            stack.extend(getattr(value, field) for field in fields)
        elif isinstance(value, tuple):
            reserve(len(value))
            stack.extend(value)


class AdmissionLimits(Record):
    """Caps on kept definitions, nesting, context work and integer bit length.

    Admission and the resource fold run on metadata a caller can make
    arbitrarily large or deep. These caps make both operations reject before
    their work or memory grows past a fixed envelope. They bound kept
    structure and admission work, never the dynamic size of the workload the
    Program describes. Repeat counts and range lengths are not expanded.

    Attributes:
        max_definitions: Total kept definitions, expressions, parameters, registers, classical values and signatures. The default, 16,384, is the smallest power of two with at least twofold margin over the 5,951 of a sampled FixedGCIM Plan with a 12-qubit, 200-term observable and four basis states (ENGINEERING_CONSTANTS.md, the Program inventory row).
        max_depth: Longest reference chain, also the recursion ceiling of lifecycle admission.
        max_steps: Admission work units per check, and separately the count of kept field slots.
        max_integer_bits: Largest bit length of any admitted or computed integer.
    """

    # See ENGINEERING_CONSTANTS.md: finite planning inventory, not dynamic work limits.
    max_definitions: PositiveInt = 16384
    max_depth: Annotated[PositiveInt, Field(le=128)] = 128
    max_steps: PositiveInt = 100000
    max_integer_bits: PositiveInt = 4096


class Parameter(Record):
    """Named integer or finite binary64 domain, with optional inclusive bounds."""

    name: Text
    domain: Literal["integer", "real"]
    lower: Value | None = None
    upper: Value | None = None

    @model_validator(mode="after")
    def _domain(self):
        for value in (self.lower, self.upper):
            if value is not None and self.domain == "integer" and type(value) is not int:
                raise ValueError("integer domain bounds must be exact integers")
        if self.lower is not None and self.upper is not None:
            if number(self.lower) > number(self.upper):
                raise ValueError("parameter domain bounds are reversed")
        return self

    def admit(self, value):
        """Reject concrete values outside the declared domain without rounding."""
        if self.domain == "integer" and type(value) is not int:
            raise ValueError(f"parameter {self.name} requires an exact integer")
        if self.domain == "real" and not isinstance(value, Float64):
            raise ValueError(f"parameter {self.name} requires Float64")
        result = number(value)
        if ((self.lower is not None and result < number(self.lower))
                or (self.upper is not None and result > number(self.upper))):
            raise ValueError(f"parameter {self.name} outside declared domain")
        return result


def number(value):
    """Read an already admitted scalar without evaluating any external reference."""
    return value.value if isinstance(value, Float64) else value


class Binding(Record):
    """One concrete parameter assignment, kept in the enclosing identity."""

    parameter: Text
    value: Value


class ExprRef(Record):
    """A local reference into Program.expressions; never a source-code string."""

    expression: Text


Integer = NonnegativeInt | ExprRef


class Constant(Record):
    """An exact integer or finite Float64 literal."""

    kind: Literal["constant"] = "constant"
    value: Value


class ParameterRef(Record):
    """The value bound to a declared Parameter.

    While the parameter is unbound, admission treats the value as unknown and
    the resource fold keeps it symbolic. It is never read as zero.
    """

    kind: Literal["parameter"] = "parameter"
    parameter: Text


class Binary(Record):
    """One operation on two local operands. Its ceildiv is exact and requires a positive denominator.

    ``left`` and ``right`` are expression IDs of the same numeric domain.
    ``eq``, ``lt`` and ``le`` return bool, and ``ceildiv`` requires integers.
    """

    kind: Literal["binary"] = "binary"
    op: Literal["add", "multiply", "ceildiv", "min", "max", "eq", "lt", "le"]
    left: Text
    right: Text


class Expression(Record):
    """A kept expression definition with a unique local ID."""

    id: Text
    value: Annotated[Constant | ParameterRef | Binary, Field(discriminator="kind")]


def topological(table, children, limits):
    """Validate a DAG and return each definition once in dependency order.

    This is an iterative depth-first search with three colors: absent means
    unvisited, 1 means on the current path and 2 means finished. Meeting a
    color-1 node again is a cycle. A node is finished after all its
    children, so the result lists children before parents, and its depth is
    one more than its deepest child, checked against ``limits.max_depth``.
    """
    colors, depths, result = {}, {}, []
    for root in table:
        stack = [(root, False)]
        while stack:
            name, finished = stack.pop()
            if name not in table:
                raise ValueError(f"unresolved local reference: {name}")
            if finished:
                refs = children(table[name])
                depth = 1 + max((depths[ref] for ref in refs), default=0)
                if depth > limits.max_depth:
                    raise ValueError("definition depth exceeds admission limit")
                depths[name] = depth
                colors[name] = 2
                result.append(name)
            elif colors.get(name) != 2:
                if colors.get(name) == 1:
                    raise ValueError("cyclic local references")
                colors[name] = 1
                stack.append((name, True))
                stack.extend((ref, False) for ref in reversed(children(table[name])))
    return result
