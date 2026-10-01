"""Small descriptive input manifests and byte admission, without SDKs."""

from nwqlib._limits import DEFAULT_MAX_BYTES

from typing import Annotated, Literal

from pydantic import Field, StrictInt

from nwqlib.core.records import Basis, InputRef, Record, Text

Count = Annotated[StrictInt, Field(ge=0)]
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
    """Immutable metadata; exports never read native payloads.

    Identity status distinguishes an ingestion digest from a caller assertion.
    Access lists executable classical operations only. Quantum access is
    declared separately by a selected preparation specification.

    Attributes:
        reference: Input identity, representation and ingestion source.
        basis: Computational basis, dimension and coordinate ordering.
        identity_status: ``ingested`` for a digest of the admitted bytes,
            ``declared`` for a caller-supplied identity.
        dtype: Stored numerical dtype, when the input has one.
        shape: Stored array or table shape.
        byte_order: Byte order of the stored numbers.
        payload_bytes: Stored native bytes.
        access: Classical operations this handle can perform.
        checked: Structural facts established at admission, such as finite
            entries or exact Hermitian equality.
        work_law: Admission and action cost law of this representation.
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
    """Preserved ordered factors; inspection never reads their native payloads.

    payload_bytes describes only the tuple's logical metadata. Referenced
    factor payload bytes are summed separately, or unknown if any is unknown;
    repeated references are counted per occurrence, not deduplicated RSS.
    """

    factors: tuple[InputRef, ...]
    basis: Basis
    structure: Literal["general"] = "general"
    payload_bytes: Count
    referenced_payload_bytes: Count | None
