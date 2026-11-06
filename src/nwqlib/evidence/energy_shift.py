"""Explicit H+cI comparison of two saved energy results and their inputs.

With the trial space held fixed, H->H+cI leaves S unchanged and adds cS to
the projected Hamiltonian, so in exact arithmetic every Ritz value shifts by
exactly c. For a Chebyshev trial space the span of T_k(K)|psi>, k<m, equals
the span of H^k|psi>, k<m (Kirby, Motta and Mezzacapo, arXiv 2208.00567v4,
Eq. (14)), and a shift does not change that span. The check reports
abs(E_target-E_baseline-c), which is zero in exact arithmetic. It measures
floating-point departure for exact or classical moments and also contains
the sampling difference between two sampled runs. Both results must share
error frame and sector. A relation established from the operator tables also
requires matched basis, unit, preparations and subspace rule and order, so
that the trial spaces are the same, and establishes H_target-H_baseline=cI
exactly from the saved operators before the comparison. An asserted relation
stays an open assumption of the check. This comparison is NWQLib's own design.
"""

import math
from fractions import Fraction
from typing import ClassVar, Literal

from pydantic import model_validator

from nwqlib.core.records import (
    Basis,
    ContentID,
    FrozenArray,
    Nonnegative,
    PositiveInt,
    Real,
    Record,
    Source,
    Text,
    Unit,
)
from nwqlib.operators.access import _check_bytes
from .error_model import CheckDomain, CheckSpec, ErrorFrame
from .verification import _publish


class EnergyEndpoint(Record):
    """Energy of one Result with the operator table and preparations it was computed from.

    The operator table holds the Result's operator in its original
    coordinates. With ``table="pauli_terms"`` or ``"matrix_entries"`` it has
    one row for each stored Pauli word or nonzero matrix entry. A row of the
    int64 array ``table_keys`` holds the word's X and then Z mask words, or
    the entry's row and column, and the corresponding row of the float64
    array ``table_values`` holds the real and imaginary parts of its
    coefficient or entry. With ``table="dense_matrix"``, ``table_keys`` is
    None and ``table_values`` has one row for every entry of the D-by-D
    matrix in row-major order. subspace_rule and order name the trial-space
    rule and its size, for Lanczos the Chebyshev rule and the Krylov
    dimension.
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
    table: Literal["pauli_terms", "matrix_entries", "dense_matrix"]
    table_keys: FrozenArray | None
    table_values: FrozenArray
    value: Real | None
    analyzer: Source | None

    @model_validator(mode="after")
    def _table(self):
        """Require preparations and an operator table in the endpoint's own coordinates.

        A Pauli key has the ``2*ceil(q/64)`` mask words of a ``2**q``
        basis, a matrix key a row and a column inside the dimension, and a
        dense table one row per entry of the matrix.
        """
        values = self.table_values.array
        dimension = self.basis.dimension
        if self.table == "dense_matrix":
            rows, width, keys = dimension * dimension, None, None
        else:
            keys = None if self.table_keys is None else self.table_keys.array
            if self.table == "pauli_terms" and (dimension < 2 or dimension.bit_count() != 1):
                raise ValueError("Pauli energy endpoint requires actual qubit coordinates")
            width = 2 * -(-(dimension.bit_length() - 1) // 64) if self.table == "pauli_terms" else 2
            rows = None if keys is None or keys.ndim != 2 else keys.shape[0]
        if (
            not self.preparations
            or (keys is None) != (self.table == "dense_matrix")
            or keys is not None and (keys.dtype.kind != "i" or keys.ndim != 2 or keys.shape[1] != width)
            or values.dtype.kind != "f"
            or values.shape != (rows, 2)
            or self.table == "matrix_entries" and keys.size and (keys.min() < 0 or keys.max() >= dimension)
        ):
            raise ValueError(
                "energy endpoint needs preparations and one (real, imag) row per operator entry "
                "in its own coordinates"
            )
        return self


def _operator_table(operator, *, max_bytes):
    """Return the endpoint's operator table fields, read from the operator's stored arrays after a byte check.

    It is computed only for the explicit energy-shift check, never by an
    ordinary solve or analysis, and is not cached. A Pauli operator gives
    the rows of its packed table, and a CSR or CSC matrix its stored nonzero
    entries in compressed-major order. A dense matrix gives all of its
    entries when at least half of them are nonzero, the smaller table, and
    otherwise its nonzero entries in row-major order. No operator is
    converted, and no Pauli transform or reference solve is performed.

    The byte check is a coarse estimate of the dominant arrays, 6T bytes
    for a table of T bytes plus one mask byte per entry of a dense matrix.
    A dense matrix table has at most 16 bytes per entry. 6T covers the
    gathered arrays, the record's copies and the base64 JSON text that the
    identity of a record holding the endpoint writes.
    ENGINEERING_CONSTANTS.md, "Eigen input conversion and classical
    preparation", records it.
    """
    import numpy as np

    keys = None
    if "pauli_terms" in operator.manifest.access:
        terms = operator.pauli_terms()
        rows, words = terms.x.shape
        _check_bytes(6 * rows * (16 * words + 16), max_bytes, "explicit energy endpoint terms")
        keys = np.hstack((terms.x, terms.z)).view(np.int64)
        values = terms.coefficients
        table = "pauli_terms"
    elif operator.reference.representation == "dense":
        d = operator.basis.dimension
        _check_bytes(6 * 16 * d * d + d * d, max_bytes, "explicit energy endpoint entries")
        dense = operator.dense_array()
        if 2 * np.count_nonzero(dense) >= d * d:
            values, table = dense.ravel(), "dense_matrix"
        else:
            rows, columns = np.nonzero(dense)
            keys, values, table = np.column_stack((rows, columns)), dense[rows, columns], "matrix_entries"
    elif operator.reference.representation in ("csr", "csc"):
        _check_bytes(6 * 32 * operator._data.nnz, max_bytes, "explicit energy endpoint entries")
        data, indices, indptr = operator.sparse_entries(max_bytes=max_bytes)
        keep = data != 0
        major = np.repeat(np.arange(operator.basis.dimension), np.diff(indptr))[keep]
        minor, values = indices[keep], data[keep]
        keys = np.column_stack((major, minor) if operator.reference.representation == "csr" else (minor, major))
        table = "matrix_entries"
    else:
        raise ValueError("energy endpoint requires existing Pauli or matrix entry access")
    parts = np.column_stack((values.real, values.imag)).astype(np.float64, copy=False)
    return dict(table=table, table_keys=None if keys is None else FrozenArray(keys.astype(np.int64, copy=False)),
                table_values=FrozenArray(parts))


def _rows(endpoint):
    """Return the keys and (real, imag) values of an endpoint's stored rows, the nonzero entries of a dense table."""
    import numpy as np

    values = endpoint.table_values.array
    if endpoint.table != "dense_matrix":
        return endpoint.table_keys.array, values
    index = np.flatnonzero(values.any(axis=1))
    return np.column_stack(np.divmod(index, endpoint.basis.dimension)), values[index]


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
    establishes `H_target - H_baseline = c I` exactly from the saved
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
            "relation", "pauli_table" if endpoint.table == "pauli_terms" else "matrix_entries"
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


def _relation(baseline, target, shift, table, *, max_bytes):
    """Establish ``H_target - H_baseline = shift * I`` exactly from two endpoint tables, Pauli or matrix by ``table``.

    Missing entries denote zero, and a dense table contributes its nonzero
    entries. The rows of both tables are merged by a lexicographic sort of
    their keys, and each table may hold a key once. Imaginary parts must be
    equal on every key, and real parts on every key that the identity does
    not touch. The identity touches the Pauli word with zero masks, or every
    diagonal matrix key. On each such key the real parts must satisfy
    ``_triple_zero(target_real, baseline_real, shift)``. With a nonzero shift
    the union must therefore hold the identity word or all D diagonal keys,
    because a key absent from both tables has difference zero. Missing
    diagonals are counted without materializing them. Under the supported
    IEEE arithmetic model every decision equals that of exact rational
    arithmetic on the recorded binary64 values, and the comparison uses no
    tolerance.

    The byte check is a coarse estimate of the merge arrays, ``(16k + 96)``
    bytes for each of the U rows of both tables, whose keys have k int64
    words, counting every entry of a dense table.
    """
    _matched_coordinates(baseline, target)
    pauli = table == "pauli_terms"
    kinds = {"pauli_terms"} if pauli else {"matrix_entries", "dense_matrix"}
    if baseline.table not in kinds or target.table not in kinds:
        raise ValueError(f"the {table} relation requires two endpoint tables of that kind")
    import numpy as np

    # Matched bases give both tables the same key width.
    width = target.table_keys.shape[1] if pauli else 2
    count = baseline.table_values.shape[0] + target.table_values.shape[0]
    _check_bytes((16 * width + 96) * count, max_bytes, "explicit H+cI relation")
    (base_keys, base_values), (target_keys, target_values) = _rows(baseline), _rows(target)
    keys = np.concatenate((base_keys, target_keys))
    count, split = keys.shape[0], base_keys.shape[0]
    order = np.lexsort(keys.T[::-1])
    ordered = keys[order]
    starts = np.ones(count, dtype=bool)
    starts[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    group = np.empty(count, dtype=np.intp)
    group[order] = np.cumsum(starts) - 1
    size = int(np.count_nonzero(starts))
    for part in (group[:split], group[split:]):
        if part.size and np.bincount(part, minlength=size).max() > 1:
            raise ValueError("an energy endpoint table repeats a key")
    base, other = np.zeros((size, 2)), np.zeros((size, 2))
    base[group[:split]] = base_values
    other[group[split:]] = target_values
    unique = ordered[starts]
    shifted = ~unique.any(axis=1) if pauli else unique[:, 0] == unique[:, 1]
    if np.any(base[:, 1] != other[:, 1]) or np.any(base[~shifted, 0] != other[~shifted, 0]):
        valid = False
    elif shift == 0:
        valid = bool(np.all(base[:, 0] == other[:, 0]))
    elif np.count_nonzero(shifted) != (1 if pauli else target.basis.dimension):
        valid = False
    else:
        valid = all(_triple_zero(t, b, shift) for t, b in zip(other[shifted, 0].tolist(), base[shifted, 0].tolist()))
    if not valid:
        recorded = "Pauli tables" if pauli else "matrix entries"
        raise ValueError(f"recorded {recorded} do not establish the selected exact H+cI relation")


def verify_energy_shift(result, *, options: EnergyShiftOptions):
    """Run the energy-shift check of a target Result against the baseline in `options`, running neither method again.

    `result.verify(checks=options)` calls this for Lanczos and FixedGCIM.
    The target's saved energy, operator table and preparations are read
    once. The exact operator relation is checked first, and the discrepancy
    `abs(E_target - E_baseline - c)` is computed in exact rational
    arithmetic.

    Args:
        result (Result): The target Lanczos or FixedGCIM Result, with its
            Plan.
        options (EnergyShiftOptions): The baseline, shift and tolerance.

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
            preparations or subspace rule, or the saved tables do not
            establish the relation exactly.
    """
    if type(options) is not EnergyShiftOptions:
        raise TypeError("energy comparison requires concrete selected options")
    target = _endpoint(result)
    checks = options._checks(result, target)
    if options.relation != "asserted":
        table = "pauli_terms" if options.relation == "pauli_table" else "matrix_entries"
        _relation(options.baseline, target, options.shift, table, max_bytes=result.plan.method.max_bytes)
    value = None
    if target.value is not None and options.baseline.value is not None:
        value = abs(Fraction(target.value) - Fraction(options.baseline.value) - Fraction(options.shift))
    return _publish(result, options, checks, (value,))
