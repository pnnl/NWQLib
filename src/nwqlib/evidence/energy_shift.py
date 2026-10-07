"""Explicit H+cI comparison of two saved energy results and their inputs.

With the trial space held fixed, H->H+cI leaves S unchanged and adds cS to
the projected Hamiltonian, so in exact arithmetic every Ritz value shifts by
exactly c. For a Chebyshev trial space the span of T_k(K)|psi>, k<m, equals
the span of H^k|psi>, k<m (Kirby, Motta and Mezzacapo, arXiv 2208.00567v4,
Eq. (14)), and a shift does not change that span. The check reports
abs(E_target-E_baseline-c), which is zero in exact arithmetic. It measures
floating-point departure for exact or classical moments and also contains
the sampling difference between two sampled runs. Both results must share
frame and sector. A table relation also requires matched basis, unit,
preparations and subspace rule and order, so that the trial spaces are the
same, and establishes H_target-H_baseline=cI exactly from the stored
operators before the comparison. An asserted relation stays an open check
prerequisite. This comparison is NWQLib's own design.
"""

import math
from itertools import repeat
from typing import ClassVar, Literal

from pydantic import model_validator

from nwqlib.core.records import (
    Basis,
    ContentID,
    Nonnegative,
    PositiveInt,
    Real,
    Record,
    Source,
    Text,
    Unit,
)
from nwqlib.operators.access import Count, _check_bytes
from ._work import BOOKKEEPING_BYTES, DEFAULT_MAX_INTEGER_BITS, ExactArithmetic
from .error_model import CheckDomain, CheckSpec, ErrorFrame
from .verification import _publish

# Longest JSON text of a finite binary64 value. json writes float repr, the
# shortest round-trip form, with at most 17 significant digits, a sign, a
# decimal point and an exponent such as e-308, as in -1.2345678901234567e-308.
_FLOAT_JSON_CHARACTERS = 24


class EnergyEndpoint(Record):
    """Actual selected scalar, operator table and normalized preparation recipes.

    It stores exactly one operator payload, either the Pauli terms or the
    distinct nonzero (row, column, real, imag) matrix entries in original
    coordinates. subspace_rule and order name the trial-space rule and its
    size, for Lanczos the Chebyshev rule and the Krylov dimension.
    """

    plan_id: ContentID
    result_id: ContentID
    construction_id: ContentID
    operator_id: ContentID
    basis: Basis
    unit: Unit
    preparations: tuple[ContentID, ...]
    subspace_rule: Literal["fixed_preparations", "chebyshev"]
    order: PositiveInt
    sector: Text | None
    frame: ErrorFrame
    terms: tuple[tuple[Text, Real], ...] | None = None
    matrix_entries: tuple[tuple[Count, Count, Real, Real], ...] | None = None
    value: Real | None
    analyzer: Source | None

    @model_validator(mode="after")
    def _coordinates(self):
        """Require preparations and exactly one operator table in the endpoint's own coordinates.

        Pauli labels must have one IXYZ character per qubit of a power-of-two
        basis. Matrix entries must be distinct, nonzero and inside the
        dimension, so each coordinate appears once in the exact comparison.
        """
        dimension = self.basis.dimension
        width = dimension.bit_length() - 1
        if not self.preparations or (self.terms is None) == (self.matrix_entries is None):
            raise ValueError(
                "energy endpoint needs preparations and exactly one actual operator payload"
            )
        if self.terms is not None:
            if (
                dimension < 2
                or dimension.bit_count() != 1
                or any(len(label) != width or set(label) - set("IXYZ") for label, _ in self.terms)
            ):
                raise ValueError("Pauli energy endpoint requires actual qubit coordinates")
        else:
            seen = set()
            for row, column, real, imag in self.matrix_entries:
                if (
                    row >= dimension
                    or column >= dimension
                    or (row, column) in seen
                    or real == imag == 0
                ):
                    raise ValueError(
                        "matrix endpoint entries must be distinct nonzero original coordinates"
                    )
                seen.add((row, column))
        return self


def _operator_payload(operator, *, max_bytes):
    """Return the endpoint's operator table, ``terms`` for Pauli input or ``matrix_entries`` for a dense or compressed matrix, after a byte check.

    It is explicit endpoint evidence, never run or cached by ordinary solve
    or analysis.

    Dense scans visit stored entries, and CSR/CSC scans visit only nnz. The record stores
    one nonzero table under a known scalar/index envelope, not a Python RSS bound.
    No operator realization, Pauli transform or reference solve is performed.

    The byte check covers the returned table, the EnergyEndpoint record that
    both callers (the Lanczos and FixedGCIM ``energy_endpoint``) build from it
    immediately, and the table's share of the identity of any record that
    holds the endpoint. ``Record.content_id`` keeps the table's scalars with
    the JSON text of the table and its UTF-8 bytes, because ``model_dump``
    shares the scalar objects. The check counts the borrowed operator
    storage, the scalars and label characters of the table and the table's
    part of those two JSON copies. The record's fixed fields, which do not
    grow with the operator, are not counted. This endpoint-capture check
    excludes tuple, list and set slots from its logical table and JSON
    payload charges. ENGINEERING_CONSTANTS.md, "Eigen input conversion and
    classical preparation", registers both laws.

    Endpoint capture admits coordinate expansion, zero filtering and
    gathered values in addition to its tuple and JSON payloads. Extraction
    preserves the input representation's established entry order. For a
    matrix, let n be the scanned count, including explicit compressed zeros,
    or D^2 for dense input. The capture charge with F = 24 is

        B_operator + n {32 + 2[6 + 2 digits(D-1) + 2F]} + 4 + 41n + 24D + H0.

    The added ``B_extraction = 41n + 24D + H0`` prices the new coordinate
    arrays. For CSR/CSC, arange(D), diff(indptr) and a possible intp
    repeat-count conversion need at most 24D bytes. The repeated
    major-coordinate vector is 8n, its keep mask n, and gathered row,
    column and values at most 32n. Their sum gives 41n+24D without assuming
    early collection. Values are gathered once in their existing float64 or
    complex128 dtype, and each scalar's real and imaginary parts are read
    when its tuple is built. Dense input needs at most D^2 mask bytes plus
    32 times its surviving count, covered by 41D^2, and omits 24D. Empty
    compressed rows still contribute index preparation, explicit zeros are
    priced before filtering, and dense zero matrices scan D^2 entries. No
    flattened row*D+column key is formed. Dense nonzero coordinates keep
    row-major order. Compressed extraction keeps its compressed-major order,
    including CSC. The extraction arrays are released before the table's
    JSON is serialized. The 32 per tuple is a logical field charge, not
    Python object storage. H0 is ``BOOKKEEPING_BYTES``.
    """
    if "pauli_terms" in operator.manifest.access:
        m, q = operator.manifest.shape[0], operator.basis.dimension.bit_length() - 1
        # Per term, the largest of three peaks. PauliTerms.labels materializes
        # q characters and a 16-byte complex coefficient, and the table adds an
        # 8-byte binary64 real part, q + 24 bytes. EnergyEndpoint validation
        # copies each label while the table is kept, 2q + 8 bytes. An identity
        # hash keeps the record's label and real part (q + 8) with two copies
        # of the term's JSON text ["label",real], of at most
        # q + 6 + _FLOAT_JSON_CHARACTERS = q + 30 bytes each counting its
        # separator, 3q + 68 bytes, the largest of the three for every q. The
        # table's enclosing brackets add at most 2 bytes to each JSON copy.
        _check_bytes(
            operator.manifest.payload_bytes + m * (3 * q + 68) + 4,
            max_bytes,
            "explicit energy endpoint terms",
        )
        return dict(
            terms=tuple(
                (p, float(c.real)) for p, c in operator.pauli_terms().labels(max_bytes=max_bytes)
            )
        )
    sparse = operator.reference.representation in ("csr", "csc")
    if not sparse and operator.reference.representation != "dense":
        raise ValueError("energy endpoint requires existing Pauli or matrix entry access")
    d = operator.basis.dimension
    count = operator._data.nnz if sparse else d * d
    # Per scanned entry, 32 bytes for the two integer indices and two binary64
    # parts of the (row, column, real, imag) tuple, which EnergyEndpoint
    # validation and model_dump share, plus two copies of its identity JSON
    # text [row,column,real,imag]. That text has 6 punctuation characters,
    # counting the separator, two indices of at most len(str(d - 1)) digits
    # and two binary64 reprs of at most _FLOAT_JSON_CHARACTERS characters. The
    # table's enclosing brackets add at most 2 bytes to each JSON copy.
    digits = len(str(d - 1))
    # Tuple and JSON payload (first line), then B_extraction of the docstring.
    _check_bytes(
        operator.manifest.payload_bytes + count * (32 + 2 * (6 + 2 * digits + 2 * _FLOAT_JSON_CHARACTERS)) + 4
        + 41 * count + (24 * d if sparse else 0) + BOOKKEEPING_BYTES,
        max_bytes,
        "explicit energy endpoint entries",
    )
    import numpy as np

    if sparse:
        data, indices, indptr = operator.sparse_entries(max_bytes=max_bytes)
        major = np.repeat(np.arange(d, dtype=np.intp), np.subtract(indptr[1:], indptr[:-1], dtype=np.intp))
        keep = data != 0
        major, minor, values = major[keep], indices[keep], data[keep]
        rows, columns = (major, minor) if operator.reference.representation == "csr" else (minor, major)
    else:
        dense = operator.dense_array()
        rows, columns = np.nonzero(dense)
        values = dense[rows, columns]
    # A real array's .imag would allocate a zero copy. Its imaginary parts are 0.0.
    imag = values.imag.tolist() if values.dtype.kind == "c" else repeat(0.0, values.size)
    entries = tuple(zip(rows.tolist(), columns.tolist(), values.real.tolist(), imag))
    return dict(matrix_entries=entries)


_SOURCE = Source(
    name="saved_energy_shift",
    version="1",
    domain="comparison of two recorded energies under an explicit H+cI relation",
    reference="nwqlib.evidence.energy_shift.verify_energy_shift",
)


class EnergyShiftOptions(Record):
    """Options of the energy-shift check, which compares an energy Result with a saved baseline for `H_target = H_baseline + c I`.

    Build it with [`for_result`][nwqlib.evidence.energy_shift.EnergyShiftOptions.for_result]
    from the baseline Result, for example
    `EnergyShiftOptions.for_result(baseline, name="shift", shift=0.5,
    tolerance=1e-10)`, and pass it to the target's
    `result.verify(checks=...)` or
    [`verify_energy_shift`][nwqlib.evidence.energy_shift.verify_energy_shift].
    It applies to Lanczos and FixedGCIM Results.

    The check reports `abs(E_target - E_baseline - c)` in the output unit,
    on `[0, inf)`, against `tolerance`. With the trial space held fixed,
    every Ritz value shifts by exactly c in exact arithmetic, so the value
    is zero in exact arithmetic. The [verification
    guide](../verification.md#two-result-energy-shift) derives this
    (Kirby, Motta and Mezzacapo, arXiv:2208.00567v4, Eq. (14)) and states
    what the value measures for exact, classical and sampled moments. This
    comparison is NWQLib's own design.

    Both Results must share the output frame and sector. With
    `relation="pauli_table"` or `"matrix_entries"`, the check first
    establishes `H_target - H_baseline = c I` exactly from the stored
    binary64 operator tables, with absent entries meaning zero. It also
    requires the same basis, unit, preparations and subspace rule and order,
    so that the two trial spaces are the same. A valid operator relation
    does not imply a zero discrepancy between independently rounded or
    sampled Ritz values. With `relation="asserted"` the stated assumption stays
    an open prerequisite, which keeps the check INCONCLUSIVE, and numerical
    agreement does not prove it. Neither workload runs again.

    Attributes:
        name: Required. Prefix of the check name, which is
            `name + ".energy_shift"`.
        baseline: Required. Energy endpoint of the saved baseline Result.
            `for_result` fills it.
        shift: Required. The constant c in `H_target = H_baseline + c I`, in
            the output unit.
        tolerance: Required. Nonnegative threshold on
            `abs(E_target - E_baseline - c)`.
        relation: Default `"pauli_table"`, and `for_result` picks the table
            kind of the baseline. How the operator relation is established:
            `"pauli_table"` or `"matrix_entries"` for an exact comparison,
            `"asserted"` for a stated assumption.
        assertion: Default `None`. The stated assumption, set exactly when
            `relation="asserted"`.

    Raises:
        ValueError: If `assertion` is set without `relation="asserted"` or
            missing with it.
    """

    # The CheckSpec of the last target Result is kept in the slot
    # _target_checks with that Result's identity, so verify_energy_shift and
    # a later Certificate.with_verification with the same options object
    # build the target endpoint once. The slot is not a field, so equality
    # and identity never see it.
    __slots__ = ("_target_checks",)
    name: Text
    baseline: EnergyEndpoint
    shift: Real
    tolerance: Nonnegative
    relation: Literal["pauli_table", "matrix_entries", "asserted"] = "pauli_table"
    assertion: Text | None = None
    source: ClassVar[Source] = _SOURCE

    @model_validator(mode="after")
    def _relation(self):
        if (self.relation == "asserted") != (self.assertion is not None):
            raise ValueError("an asserted energy relation needs its explicit premise")
        return self

    @classmethod
    def for_result(cls, result, **choices):
        """Build options with `result` as the baseline and its operator table kind as the default relation.

        Args:
            result (Result): The baseline Lanczos or FixedGCIM Result, with
                its Plan.
            **choices (object): The other fields, `name`, `shift` and
                `tolerance`, and optionally `relation` and `assertion`.

        Returns:
            options (EnergyShiftOptions): Options with the baseline's energy
                endpoint.

        Raises:
            TypeError: If the Result has no saved energy endpoint.
        """
        endpoint = _endpoint(result)
        choices.setdefault(
            "relation", "pauli_table" if endpoint.terms is not None else "matrix_entries"
        )
        return cls(baseline=endpoint, **choices)

    def verification_checks(self, result):
        """Return a one-element tuple with the CheckSpec this comparison answers for the target ``result``.

        The target must keep the baseline's error frame and sector. The check
        is ``abs(E_target - E_baseline - shift)`` in the output unit, on the
        domain [0, inf), against ``tolerance``. An asserted relation enters
        as an open prerequisite, which keeps the check INCONCLUSIVE.
        """
        try:
            stored = object.__getattribute__(self, "_target_checks")
        except AttributeError:
            stored = None
        if stored is not None and stored[0] == result.content_id:
            return stored[1]
        return self._checks(result, _endpoint(result))

    def _checks(self, result, target):
        """Build and keep the CheckSpec of ``result`` from its already read endpoint ``target``."""
        if (
            not target.frame.compatible(self.baseline.frame)
            or target.sector != self.baseline.sector
        ):
            raise ValueError(
                "energy comparison must preserve quantity, unit, scope, conditioning and sector"
            )
        checks = (
            CheckSpec(
                name=self.name + ".energy_shift",
                claim_id=result.plan.output.content_id,
                frame=target.frame.revise(metric="absolute_shift_discrepancy"),
                domain=CheckDomain(lower=0.0),
                source=self.source,
                options_id=self.content_id,
                threshold=self.tolerance,
                prerequisites=() if self.assertion is None else (self.assertion,),
                access=("two saved energy results and their actual operator/preparation metadata",),
                experiments=0,
                classical_work="exact selected operator relation by a sorted entry merge and one absolute shifted difference",
                reference_work="none; neither workload is executed again",
                data_description="both endpoint identities, baseline metadata and scalar discrepancy",
            ),
        )
        object.__setattr__(self, "_target_checks", (result.content_id, checks))
        return checks


def _endpoint(result):
    """Read the Result's own EnergyEndpoint and check that it names this Plan, Result and construction."""
    result.validate_plan(result.plan)
    factory = getattr(result, "energy_endpoint", None)
    if not callable(factory):
        raise TypeError("result has no supported saved energy endpoint")
    endpoint = factory()
    if (
        type(endpoint) is not EnergyEndpoint
        or endpoint.plan_id != result.plan_id
        or endpoint.result_id != result.content_id
        or endpoint.construction_id != result.construction_id
    ):
        raise ValueError("energy endpoint differs from its actual Plan/result")
    return endpoint


def _matched_coordinates(baseline, target):
    """Require the same basis, unit, preparations and subspace rule and order, hence the same trial space."""
    if (
        baseline.basis != target.basis
        or not baseline.unit.same_unit(target.unit)
        or baseline.preparations != target.preparations
        or (baseline.subspace_rule, baseline.order) != (target.subspace_rule, target.order)
    ):
        raise ValueError(
            "H+cI comparison requires matched actual basis and normalized preparation/subspace rule"
        )


# Exact-zero predicate and relation workspace of the stored-table H+cI check.
#
# For two finite binary64 values, float equality is equality of their
# represented real values, and signed zeros represent the same real number.
# Every finite binary64 number is an integer multiple of eta = 2**-1074 with
# magnitude below 2**1024. For three such numbers t, b and c, the exact
# S = t-b-c is an integer multiple of eta, so if S is nonzero then |S| >= eta.
# The round-to-nearest basin of zero extends only to eta/2, so correctly
# rounding S cannot give zero, and if S is zero its rounded value is zero.
# This holds under gradual underflow, including subnormal operands and
# residuals, and fails under flush-to-zero. CPython 3.12's fsum keeps exact
# nonoverlapping partials and rounds their total under IEEE
# round-to-nearest-even
# (https://github.com/python/cpython/blob/v3.12.14/Modules/mathmodule.c#L1165-L1362,
# https://docs.python.org/3.12/library/math.html#math.fsum). These
# implementation premises, not a small test, support the predicate.
# Rounding each side first loses it: t == b+c accepts t = b = 1, c = eta,
# whose exact residual is -eta.


def _triple_zero(t, b, c):
    """Exact ``t - b - c == 0`` for finite binary64 under CPython fsum's IEEE premises.

    ``fsum((t, -b, -c))`` can raise OverflowError for finite inputs whose exact
    total is nonzero (t = DBL_MAX, b = -DBL_MAX, c = DBL_MAX). For exactly
    three finite operands whose exact sum is zero, any same-sign pair has
    magnitude equal to the third operand and cannot overflow, an
    opposite-sign pair cannot overflow either, and the exact-partial
    reconstruction cannot overflow in this zero case. An overflow therefore
    means the relation is false.
    """
    try:
        return math.fsum((float(t), -float(b), -float(c))) == 0.0
    except OverflowError:
        return False  # Three finite operands with zero exact sum cannot overflow.


def _relation_bytes(base_count, target_count, *, pauli_width=None):
    """Return the byte admission of the exact relation workspace, checked before allocation.

    The exact relation check admits linear coordinate, value, grouping and
    NumPy sort workspace before allocation. Missing matrix diagonals are
    counted without materializing them. Existing endpoint
    objects belong to B_held and are not charged here.

    Matrix relations, ``U = N_base + N_target``, where ``N_base`` and
    ``N_target`` count the stored entries of the two endpoints:
    ``B_matrix = 192U + H0 + B_coord_objects``. Here H0 is
    ``BOOKKEEPING_BYTES`` and ``B_coord_objects`` charges newly boxed coordinate
    integers. The 160U logical envelope covers converted coordinates and
    values (32U), concatenated coordinates (16U), sorted coordinates (16U),
    the returned permutation (8U), starts and group maps (17U), base and
    target union arrays (at most 32U) and masks and gathered comparison
    operands (at most 39U). 32U more covers sorting scratch and buffer
    growth. NumPy 2.5.2 PyArray_LexSort sorts its two keys sequentially by
    stable indirect sort. Its copy branch uses 8U value bytes and 8U index
    bytes, and the int64/intp and object indirect Timsort keep a merge
    buffer of at most U/2 indices whose realloc can overlap old and new
    buffers (8U), so 16U copy buffers plus 8U merge buffers fit the extra
    32U. Fixed run stacks and headers belong to H0, and there is no
    ``U*log(U)`` auxiliary array
    (https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/multiarray/item_selection.c,
    https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/npysort/npysort_methods.cpp,
    https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/npysort/timsort.hpp,
    https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/npysort/timsort_generic.cpp).
    Coordinates beyond int64 use eight-byte object pointers to the
    endpoint's existing Python integers, which are borrowed, so this
    implementation boxes no coordinate objects and ``B_coord_objects = 0``.
    A conversion that boxed them would add ``2U*L(bit_length(D-1))``, where
    D is the matrix dimension and L is ``integer_object_bytes``.

    Pauli relations, N = N_base + N_target + 1 including the identity-shift
    contribution and q the label width: ``B_Pauli = [16 max(1,q) + 256]N +
    H0``. Four Unicode populations (original labels, unique's flattened
    copy, the permuted copy and the returned unique labels) take at most
    4*max(1,q)*N bytes each. 256N covers values, inverse, count, group and
    index arrays, temporary list references, grouped values, masks, gathers
    and sort workspace
    (https://github.com/numpy/numpy/blob/v2.5.2/numpy/lib/_arraysetops_impl.py).
    H0 is ``BOOKKEEPING_BYTES``.
    """
    count = base_count + target_count
    if pauli_width is not None:
        return (16 * max(1, pauli_width) + 256) * (count + 1) + BOOKKEEPING_BYTES
    return 192 * count + BOOKKEEPING_BYTES


def _pauli_relation(baseline, target, shift, *, max_bytes):
    """Establish ``H_target - H_baseline = shift * I`` exactly from the two Pauli tables.

    Missing entries denote zero.

    Every public Pauli ``OperatorInput`` is coalesced by
    ``operators/inputs.py::_coalesced_input``, which merges identical
    words. The Lanczos and FixedGCIM endpoint producers read that table
    through ``_operator_payload``, and a reopened Plan restores the stored
    coalesced tables. Thus each endpoint has at most one coefficient per
    word.

    The relation groups the baseline coefficient with a minus sign, the
    target coefficient and the negative identity shift. A nonidentity
    group has at most two contributions and the identity group at most
    three. Singletons are compared with zero, pairs with exact negation,
    and triples with ``_triple_zero`` under its stated CPython and IEEE
    premises. These tests decide equality of the represented binary64
    operators without a tolerance. Pauli words are linearly independent,
    so these conditions are equivalent to
    ``H_target - H_baseline = shift * I``, where ``H_target`` and ``H_baseline``
    are the two operators and I is the identity. With q the label width
    and ``N_base`` and ``N_target`` the two stored term counts, the Pauli
    workspace is
    ``[16*max(1,q)+256]*(N_base+N_target+1)+BOOKKEEPING_BYTES`` bytes.

    The comment above ``_triple_zero`` gives the exact-zero argument under
    the stated CPython and IEEE premises. Its docstring explains why an
    intermediate overflow implies a nonzero exact three-term sum. The
    contributions are grouped by label with
    ``np.unique(..., return_inverse=True, return_counts=True)`` and a
    stable argsort. The exact relation concerns the coalesced binary64
    operators, not exact sums of the user's original decimal or duplicate
    coefficient stream before ingestion rounded it.
    """
    _matched_coordinates(baseline, target)
    if baseline.terms is None or target.terms is None:
        raise ValueError("Pauli relation requires two actual Pauli endpoint tables")
    width = target.basis.dimension.bit_length() - 1
    _check_bytes(_relation_bytes(len(baseline.terms), len(target.terms), pauli_width=width),
                 max_bytes, "explicit Pauli relation entries")
    import numpy as np

    label_type = f"<U{max(1, width)}"
    labels = np.concatenate((
        np.fromiter((label for label, _ in baseline.terms), dtype=label_type, count=len(baseline.terms)),
        np.fromiter((label for label, _ in target.terms), dtype=label_type, count=len(target.terms)),
        np.array(["I" * width], dtype=label_type)))
    values = np.concatenate((
        -np.fromiter((value for _, value in baseline.terms), dtype=np.float64, count=len(baseline.terms)),
        np.fromiter((value for _, value in target.terms), dtype=np.float64, count=len(target.terms)),
        np.array([-float(shift)])))
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    ordered = values[np.argsort(inverse, kind="stable")]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    one = starts[counts == 1]
    two = starts[counts == 2]
    valid = not (np.any(ordered[one] != 0) or np.any(ordered[two] != -ordered[two + 1]))
    for index in np.flatnonzero(counts == 3) if valid else ():
        entries = ordered[starts[index]:starts[index] + 3]
        valid = _triple_zero(entries[0], -entries[1], -entries[2])
        if not valid:
            break
    if not valid:
        raise ValueError("recorded Pauli tables do not establish the selected exact H+cI relation")


def _coordinates(entries, position, dimension):
    """Gather one coordinate column as int64, or as the endpoint's own Python integers beyond int64."""
    import numpy as np

    dtype = np.int64 if dimension - 1 <= np.iinfo(np.int64).max else object
    return np.fromiter((entry[position] for entry in entries), dtype=dtype, count=len(entries))


def _matrix_relation(baseline, target, shift, *, max_bytes):
    """Establish ``H_target - H_baseline = shift * I`` exactly from the two entry tables.

    Missing entries denote zero. Equal nonidentity and
    off-diagonal entries are compared directly, while each diagonal or
    identity shift is checked by an exact-zero sum under the supported IEEE
    arithmetic model. Matrix coordinates are merged without materializing
    missing diagonal entries. The comparison uses no tolerance.

    The union of stored coordinates is merged with ``np.lexsort((column,
    row))``, where column and row are the coordinate vectors. No flattened
    ``row*D+column`` key is formed, since it can overflow for a very large
    sparse dimension D. The absent side of a union key is zero. Imaginary
    parts must be equal on every union key and real parts equal off the
    diagonal. Each diagonal key must satisfy
    ``_triple_zero(target_real, base_real, shift)``, where ``target_real`` and
    ``base_real`` are the real entries at that key. If shift is zero these
    real tests reduce to vector equality. If shift is nonzero and the union
    holds fewer than D distinct diagonal coordinates, some diagonal is
    absent on both sides with difference zero, so the relation fails. This
    counts missing diagonals without allocating D entries. Empty tables
    establish only shift zero. For ``U = N_base + N_target``, where ``N_base``
    and ``N_target`` count the two endpoints' stored entries, the merge uses
    O(U log U) coordinate comparisons and O(U) array entries of working
    storage. These counts do not grow with D. Python-integer coordinate
    comparisons can depend on coordinate bit length. The workspace law is
    ``_relation_bytes``.
    Under the supported IEEE arithmetic model, every decision equals that of
    exact rational arithmetic on the recorded binary64 values, and its only
    resource refusal is the workspace law.
    """
    _matched_coordinates(baseline, target)
    if baseline.matrix_entries is None or target.matrix_entries is None:
        raise ValueError("matrix relation requires two actual matrix endpoint tables")
    base, other = baseline.matrix_entries, target.matrix_entries
    _check_bytes(_relation_bytes(len(base), len(other)), max_bytes, "explicit matrix relation entries")
    if not base and not other:
        valid = shift == 0
    else:
        import numpy as np

        dimension = target.basis.dimension
        rows = np.concatenate((_coordinates(base, 0, dimension), _coordinates(other, 0, dimension)))
        columns = np.concatenate((_coordinates(base, 1, dimension), _coordinates(other, 1, dimension)))
        order = np.lexsort((columns, rows))
        sorted_rows, sorted_columns = rows[order], columns[order]
        starts = np.concatenate(([True], (sorted_rows[1:] != sorted_rows[:-1])
                                 | (sorted_columns[1:] != sorted_columns[:-1])))
        group_sorted = np.cumsum(starts) - 1
        group = np.empty(order.size, dtype=np.intp)
        group[order] = group_sorted
        size = int(group_sorted[-1]) + 1
        base_values, target_values = np.zeros(size, dtype=np.complex128), np.zeros(size, dtype=np.complex128)
        base_values[group[:len(base)]] = np.fromiter(
            (complex(entry[2], entry[3]) for entry in base), dtype=np.complex128, count=len(base))
        target_values[group[len(base):]] = np.fromiter(
            (complex(entry[2], entry[3]) for entry in other), dtype=np.complex128, count=len(other))
        diagonal = sorted_rows[starts] == sorted_columns[starts]
        if np.any(base_values.imag != target_values.imag) or np.any(
                base_values.real[~diagonal] != target_values.real[~diagonal]):
            valid = False
        elif shift == 0:
            valid = bool(np.all(base_values.real == target_values.real))
        elif np.count_nonzero(diagonal) != dimension:
            valid = False
        else:
            valid = all(_triple_zero(t, b, shift) for t, b in zip(
                target_values.real[diagonal].tolist(), base_values.real[diagonal].tolist()))
    if not valid:
        raise ValueError("recorded matrix entries do not establish the selected exact H+cI relation")


def verify_energy_shift(
    result, *, options: EnergyShiftOptions, max_integer_bits=DEFAULT_MAX_INTEGER_BITS
):
    """Run the energy-shift check of a target Result against the baseline in `options`, running neither method again.

    `result.verify(checks=options)` calls this for Lanczos and FixedGCIM.
    The target's stored energy, operator table and preparations are read
    once. The exact operator relation is checked first, and the discrepancy
    `abs(E_target - E_baseline - c)` is computed in exact arithmetic.

    Args:
        result (Result): The target Lanczos or FixedGCIM Result, with its
            Plan.
        options (EnergyShiftOptions): The baseline, shift and tolerance.
        max_integer_bits (int): Bit limit of the exact arithmetic. Default
            4096.

    Returns:
        verification (tuple): `(receipt, facts)`: the `VerificationReceipt`
            of this comparison, and a one-element tuple with the
            discrepancy as a [`FramedFact`][nwqlib.evidence.error_model.FramedFact]
            that cites it. The fact is unknown when either energy is
            missing.

    Raises:
        TypeError: If `options` is not an `EnergyShiftOptions`, or the
            Result has no saved energy endpoint.
        ValueError: If the Results differ in frame or sector, basis, unit,
            preparations or subspace rule, the stored tables do not
            establish the relation exactly, or an exact intermediate value
            needs more than `max_integer_bits` bits.
    """
    if type(options) is not EnergyShiftOptions:
        raise TypeError("energy comparison requires concrete selected options")
    target = _endpoint(result)
    checks = options._checks(result, target)
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    if options.relation == "pauli_table":
        _pauli_relation(options.baseline, target, options.shift, max_bytes=result.plan.method.max_bytes)
    elif options.relation == "matrix_entries":
        _matrix_relation(options.baseline, target, options.shift, max_bytes=result.plan.method.max_bytes)
    value = None
    if target.value is not None and options.baseline.value is not None:
        value = arithmetic.absolute(
            arithmetic.subtract(
                arithmetic.subtract(
                    arithmetic.fraction(target.value), arithmetic.fraction(options.baseline.value)
                ),
                arithmetic.fraction(options.shift),
            )
        )
    return _publish(result, options, checks, (value,), max_integer_bits=max_integer_bits)
