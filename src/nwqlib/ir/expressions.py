"""Finite typed expression definitions with exact integer work.

No source is ever evaluated.
"""

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
    """Limits on the size of a Program's stored structure and on the work of checking it.

    Pass it as `limits=` to [`Program`][nwqlib.ir.records.Program]. Every
    argument is optional. A caller can make a Program's metadata arbitrarily
    large or deep, and these limits make the structural check and the resource
    estimate reject it before their work or memory passes a fixed size. They
    limit the stored structure and the checking work, never the size of the
    workload the Program describes, and repeat counts and range lengths are not
    expanded. A Method with a `max_admission_steps` setting uses it as
    `max_steps` of its Programs, and [Program checks](../development/program_checks.md) explains how planning
    work is counted.

    Attributes:
        max_definitions: Default `1_000_000`. Positive limit on the stored
            definitions, expressions, parameters, registers, classical values and
            signatures together. The declaration ceiling covers a sampled FixedGCIM
            plan with three basis states and a 20-qubit, 8,083-term molecular
            Hamiltonian in 1,285 measurement groups, with 11,565 circuits and 25,731
            declarations. It has over sixfold margin over the 161,691 declarations
            of singleton grouping (the Program inventory row of
            [Engineering constants](../ENGINEERING_CONSTANTS.md)). Stored-field and
            checking-work limits apply independently, and no space is reserved from
            this ceiling alone.
        max_depth: Default `128`, also the largest accepted value. Limit on the
            longest chain of references, which is also the recursion limit of the
            lifecycle check.
        max_steps: Default `100000`. Positive limit on the checking work units of
            one check, and separately on the number of stored fields.
        max_integer_bits: Default `4096`. Positive limit on the bit length of any
            integer the check accepts or computes.
    """

    # Reasons and revisit conditions: ENGINEERING_CONSTANTS.md, "Shared Program
    # admission limits". They bound the stored planning inventory, not dynamic work.
    max_definitions: PositiveInt = 1_000_000
    max_depth: Annotated[PositiveInt, Field(le=128)] = 128
    max_steps: PositiveInt = 100000
    max_integer_bits: PositiveInt = 4096


class Parameter(Record):
    """A named integer or real parameter of a Program, with optional inclusive bounds.

    Build it with keyword arguments, for example
    `Parameter(name="steps", domain="integer", lower=1)`, and pass it in
    `parameters=` of a [`Program`][nwqlib.ir.records.Program] or a
    [`BlockSignature`][nwqlib.ir.records.BlockSignature]. `name` and `domain` are
    required. A real value is a finite binary64 number given as `Float64`.

    Attributes:
        name: Required. Parameter name.
        domain: Required. `"integer"` or `"real"`.
        lower: Default `None`. Inclusive lower bound, an exact integer for an
            integer domain.
        upper: Default `None`. Inclusive upper bound, not below `lower`.

    Raises:
        ValueError: If an integer domain has a non-integer bound, or `lower`
            exceeds `upper`.
    """

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
        """Check a value against the parameter's domain and bounds, without rounding, and return it as a number.

        Args:
            value (int | Float64): An exact integer for an integer domain, or a
                `Float64` for a real domain.

        Returns:
            value (int | float): The value as a number.

        Raises:
            ValueError: If the value has the wrong type or lies outside the bounds.
        """
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
    """A concrete value for one parameter, part of the content hash of the record that holds it.

    Build it as `Binding(parameter="steps", value=4)` and pass it in
    `bindings=` of a Program, a `Setting` or a selected definition. Both
    arguments are required.

    Attributes:
        parameter: Required. Parameter name.
        value: Required. An exact integer or a `Float64`.
    """

    parameter: Text
    value: Value


class ExprRef(Record):
    """A reference to an expression of the Program by its ID, used wherever a width, count or argument may be computed.

    Build it as `ExprRef(expression="width")`. The ID names an entry of
    `Program.expressions`. It is never source code.

    Attributes:
        expression: Required. ID of the expression.
    """

    expression: Text


Integer = NonnegativeInt | ExprRef


class Constant(Record):
    """An exact integer or finite `Float64` literal, the value of an expression.

    Build it as `Constant(value=3)` and wrap it in an
    [`Expression`][nwqlib.ir.expressions.Expression].

    Attributes:
        value: Required. An exact integer or a `Float64`.
        kind: Fixed `"constant"`. Names the expression type in the saved
            record.
    """

    kind: Literal["constant"] = "constant"
    value: Value


class ParameterRef(Record):
    """The value of a declared parameter, as an expression.

    Build it as `ParameterRef(parameter="steps")` and wrap it in an
    [`Expression`][nwqlib.ir.expressions.Expression]. While the parameter is
    unbound, the structural check treats the value as unknown and the resource
    estimate keeps it symbolic. It is never read as zero.

    Attributes:
        parameter: Required. Name of a declared parameter.
        kind: Fixed `"parameter"`. Names the expression type in the saved
            record.
    """

    kind: Literal["parameter"] = "parameter"
    parameter: Text


class Binary(Record):
    """An operation on two expressions of the Program, such as a sum or an exact ceiling division.

    Build it as `Binary(op="multiply", left="n", right="two")` and wrap it in an
    [`Expression`][nwqlib.ir.expressions.Expression]. Both operands must have the
    same numeric domain. `"eq"`, `"lt"` and `"le"` return a bool, and
    `"ceildiv"` is exact, requires integers and a positive denominator.

    Attributes:
        op: Required. `"add"`, `"multiply"`, `"ceildiv"`, `"min"`, `"max"`, `"eq"`,
            `"lt"` or `"le"`.
        left: Required. ID of the left operand's expression.
        right: Required. ID of the right operand's expression.
        kind: Fixed `"binary"`. Names the expression type in the saved record.
    """

    kind: Literal["binary"] = "binary"
    op: Literal["add", "multiply", "ceildiv", "min", "max", "eq", "lt", "le"]
    left: Text
    right: Text


class Expression(Record):
    """A named expression of a Program: a constant, a parameter value or an operation on two expressions.

    Build it as `Expression(id="width", value=Constant(value=3))` and pass it in
    `expressions=` of a [`Program`][nwqlib.ir.records.Program]. Both arguments are
    required, and the ID is unique within the Program.

    Attributes:
        id: Required. Unique ID within the Program.
        value: Required. A [`Constant`][nwqlib.ir.expressions.Constant], a
            [`ParameterRef`][nwqlib.ir.expressions.ParameterRef] or a
            [`Binary`][nwqlib.ir.expressions.Binary].
    """

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
