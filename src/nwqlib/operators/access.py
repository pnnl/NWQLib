"""Small descriptive input manifests and byte admission, without SDKs."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from typing import Annotated, Literal

from pydantic import Field, StrictInt

from nwqlib.core.records import Basis, InputRef, Record, Text

Count = Annotated[StrictInt, Field(ge=0)]
"""Nonnegative integer field: a Python `int` that is 0 or larger.

A `bool` or a float such as `2.0` is rejected, not converted.
"""
DEFAULT_INPUT_BYTES = DEFAULT_MAX_BYTES


def _check_bytes(size: int, max_bytes: int, operation: str = "input") -> None:
    """Reject an operation whose peak live arrays would exceed max_bytes, before it allocates.

    ``size`` is the caller's byte law, an upper bound on the arrays alive at
    the same time at the peak of one operation, as far as they are known
    before allocation. It excludes SDK and LAPACK workspace, Python object overhead
    and process RSS. Each caller states which arrays its terms count. A
    dimension integer such as ``2**q`` is charged by its own byte length.
    docs/CODE_TOUR.md, "Byte and work budgets", explains why budgets are
    checked before work.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if size > max_bytes:
        raise ValueError(f"{operation} needs {size} data bytes, exceeding max_bytes={max_bytes}")


def _check_products(count: int, max_products: int, limit_name: str = "max_products") -> None:
    """Bound scalar multiplications in one explicitly requested action.

    ``count`` is a derived count of the scalar products the action will
    perform, fixed by its representation before any product runs. It is a
    work limit, not measured CPU time. ``limit_name`` is the caller's option
    that supplies ``max_products``; the refusal names it.
    """
    if type(max_products) is not int or max_products < 1:
        raise ValueError(f"{limit_name} must be a positive integer")
    if count > max_products:
        raise ValueError(f"action needs {count} scalar products, exceeding {limit_name}={max_products}")


def refuse_known_need(family, field, cap, known_need, later):
    """Refuse when the largest planning requirement already known exceeds ``cap``.

    At each planning boundary a Method passes the maximum of the complete
    requirements of every phase whose occurrence and dimensions are already
    known. Cumulative work laws stay sums where the owner charges a sum;
    successive peak-memory phases combine by maximum, with data held across
    them added inside each phase. A hypothetical branch that might not be
    selected is a sufficient envelope, not a necessary requirement, and is
    not passed here. ``later`` names the phases whose populations depend on
    coefficients, selected ranks, spectra or observations, which can need
    more. No eigensolve, symbolic expansion, grouping or pair scan is run to
    discover a remedy.
    """
    if known_need > cap:
        raise ValueError(
            f"{family}.{field}={cap} is below the largest known "
            f"planning requirement {known_need}. Raise {family}.{field} "
            f"to at least {known_need}. Later phases can need more: {later}."
        )


class InputManifest(Record):
    """Metadata of an accepted operator or state: content hash, representation, basis, size and checks.

    `OperatorInput.manifest` and `StateInput.manifest` hold it. Reading or
    saving it never reads the numerical data. `access` lists the classical
    operations of the handle only. How a state can be prepared on qubits is
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
        access: Classical operations of the handle, among `"matvec"`,
            `"entries"`, `"pauli_terms"` and `"fermion_terms"`.
        checked: Properties checked when the input was accepted, such as
            finite entries or exact equality with the conjugate transpose.
        work_law: Text form of the cost of accepting and applying this
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
