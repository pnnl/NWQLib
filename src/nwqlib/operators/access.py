"""Metadata records of operator and state inputs, and the checks made before work.

The byte and product checks run without Qiskit or SciPy.
"""

from nwqlib._limits import ResourceLimitError, check_memory, exceeds, memory_limit
from nwqlib._validation import integer

from typing import Annotated, Literal

from pydantic import Field

from nwqlib.core.records import Basis, InputRef, Record, StrictInteger, Text

Count = Annotated[StrictInteger, Field(ge=0)]
"""Nonnegative integer field: a Python `int` or NumPy integer that is 0 or larger, stored as `int`.

A `bool` or a float such as `2.0` is rejected, not converted. Convert a
whole-number float with `int(...)`.
"""


def _check_bytes(size: int, max_bytes: int | None, operation: str = "input") -> None:
    """Reject an operation whose dominant arrays would exceed the memory limit, before it allocates.

    ``size`` is the caller's coarse byte estimate of the arrays alive at the
    same time at the peak of one operation. It excludes workspace inside
    SciPy, Qiskit and LAPACK, Python object overhead and the resident memory
    of the process. ``max_bytes=None`` uses the physical memory of this
    computer (`nwqlib._limits.check_memory`).
    """
    check_memory(size, max_bytes, operation)


def _check_products(count: int, max_products: int | None, limit_name: str = "max_products") -> None:
    """Bound scalar multiplications in one explicitly requested action, where ``max_products=None`` sets no limit.

    ``count`` is a derived count of the scalar products the action will
    perform, fixed by its representation before any product runs. It is a
    work limit, not measured CPU time. ``limit_name`` is the caller's option
    that supplies ``max_products``, and the error message names it.
    """
    if max_products is None:
        return
    try:
        max_products = integer(max_products, limit_name, 1)
    except ValueError:
        raise ValueError(f"{limit_name} must be a positive integer or None") from None
    if count > max_products:
        raise ResourceLimitError(f"action needs {count} scalar products, exceeding {limit_name}={max_products}")


def refuse_known_need(family, field, cap, known_need, later):
    """Raise when the largest byte requirement already known exceeds ``cap``, where None means the physical memory.

    At each planning boundary a Method passes the maximum of the complete
    requirements of every phase whose occurrence and dimensions are already
    known. Cumulative work formulas stay sums where the Method counts a sum.
    Successive peak-memory phases combine by maximum, with data held across
    them added inside each phase. A branch that the Method might not take
    gives a sufficient upper bound, not a necessary requirement, and is not
    passed here. ``later`` names the phases whose sizes depend on
    coefficients, chosen ranks, spectra or observations, which can need
    more. No eigensolve, symbolic expansion, grouping or pair scan is run to
    discover a remedy.
    """
    limit = memory_limit(cap)
    if not exceeds(known_need, limit):
        return
    if cap is None:
        raise ResourceLimitError(
            f"the largest known planning requirement of {family}, {known_need} bytes, exceeds the "
            f"physical memory ({limit} bytes) that {family}.{field}=None allows. A computer with at "
            f"least {known_need} bytes of physical memory admits it. Later phases can need more: {later}."
        )
    raise ResourceLimitError(
        f"{family}.{field}={cap} is below the largest known "
        f"planning requirement {known_need}. Raise {family}.{field} "
        f"to at least {known_need}. Later phases can need more: {later}."
    )


class InputManifest(Record):
    """Metadata of an operator or state input: content hash, representation, basis, size and checks.

    `OperatorInput.manifest` and `StateInput.manifest` hold it. Reading or
    saving it never reads the numerical data. `access` lists only the
    classical operations of the input object. How a state can be prepared on qubits is
    recorded separately, in
    [`StatePreparationSpec`][nwqlib.problems.inputs.StatePreparationSpec].
    The fields below are read-only.

    Attributes:
        reference: Reference of the input: identifier, representation (for
            example `"dense"`, `"csr"`, `"pauli"` or `"vector"`) and source.
        basis: Computational basis, dimension and coordinate order.
        identity_status: `"ingested"` when the identifier is a SHA-256 hash of
            the stored data with its representation, dtype, shape and order,
            `"declared"` when the caller supplied it, as for a preparation
            circuit.
        dtype: Stored numerical dtype, or `None` when the input has none.
        shape: Stored array or table shape, or `None`.
        byte_order: `"little"` or `"big"` for stored numbers,
            `"not_applicable"` for one-byte data, or `None`.
        payload_bytes: Bytes of the stored data, or `None` when unknown.
        access: Classical operations of the input object, among `"matvec"`,
            `"entries"`, `"pauli_terms"` and `"fermion_terms"`.
        checked: Properties checked when the input object was created, such
            as finite entries or exact equality with the conjugate transpose.
        work_law: Text form of the cost of converting and applying this
            representation.
    """

    reference: InputRef
    basis: Basis
    identity_status: Literal["ingested", "declared"]
    dtype: Text | None
    shape: tuple[Count, ...] | None
    byte_order: Literal["little", "big", "not_applicable"] | None
    payload_bytes: Count | None
    access: tuple[Literal["matvec", "entries", "pauli_terms", "fermion_terms"], ...]
    checked: tuple[Text, ...] = ()
    work_law: Text


class ProductManifest(Record):
    """Metadata of a [`FactorizedOperatorProduct`][nwqlib.operators._factorized.FactorizedOperatorProduct]: its ordered factors, basis and sizes.

    Reading it never reads the factors' numerical data. The fields below are
    read-only.

    Attributes:
        factors: References of the factors, in product order.
        basis: The basis that every factor shares.
        structure: Always `"general"`.
        payload_bytes: Bytes of the factor-reference tuple only.
        referenced_payload_bytes: Sum of the factors' stored bytes, counted
            once per occurrence, so a repeated factor counts each time. This
            is not a measurement of deduplicated memory. `None` when the size
            of any factor is unknown.
    """

    factors: tuple[InputRef, ...]
    basis: Basis
    structure: Literal["general"] = "general"
    payload_bytes: Count
    referenced_payload_bytes: Count | None
