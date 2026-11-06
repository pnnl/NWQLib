"""Cartesian access of a generator `A = L + iH` and the structural scalar facts of its parts, without dense work.

Linear dynamics `du/dt = -A u + b` uses `L = (A + A†)/2` and
`H = (A - A†)/(2i)`, both Hermitian, with `A = L + iH`. This module gives
a solver the two parts of a compact A as operator inputs, and upper and
lower bounds on their spectra that come from the stored data alone.
Nothing here forms a vector or matrix of the system dimension, runs an
eigensolver or simulates a circuit. The work and storage grow with the
number of Pauli terms or bands. A lower bound of L that the caller declares
to `ingest_cartesian` replaces the structural one when it is larger, and the
facts then name it as an assertion (`_with_declared_lower`).
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from math import inf, isfinite, nextafter
from typing import Literal

import numpy as np

from nwqlib._numerics import complex_magnitude_upper, complex_magnitudes_upper, outward_sum
from .access import _check_bytes


def _outward(value: Fraction, *, upward: bool) -> float:
    """Return the binary64 number next to the exact rational ``value`` in the requested direction.

    The result is at least ``value`` when ``upward`` and at most ``value``
    otherwise, so a bound computed exactly stays a bound after rounding.
    A value beyond the binary64 range raises, because an infinite bound
    would carry no information.
    """
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("operator bound exceeds the binary64 range") from None
    exact = Fraction(result)
    if upward and exact < value:
        result = nextafter(result, inf)
    elif not upward and exact > value:
        result = nextafter(result, -inf)
    if not isfinite(result):
        raise ValueError("operator bound exceeds the binary64 range")
    return result


@dataclass(frozen=True, kw_only=True)
class PartFacts:
    """Scalar facts of one Hermitian part X of `A = L + iH`, in the units of A.

    Attributes:
        access: How X is available: `"zero"` (X is exactly zero), `"pauli"`,
            `"band"`, `"block_encoding"` (a supplied encoding of X) or
            `"single_a"` (only an encoding of A exists, so X is not a
            separate operator and only its bounds are known).
        identity: The coefficient `c_X` of the identity, read exactly from a
            Pauli sum (the identity word) or a band (offset 0), which is the
            mean eigenvalue `tr(X)/D` of X, and 0 when X gives no identity
            coefficient to extract, which says nothing about its mean.
        centered_alpha: The LCU normalization of `X - c_X I`: the sum of the
            non-identity coefficient magnitudes of a Pauli sum or band,
            rounded up, which is positive exactly when X is not scalar, or the
            `alpha` of a supplied encoding, which does not show that. `None`
            for `"single_a"`.
        lower: Lower bound on the smallest eigenvalue of X, or `None`.
        upper: Upper bound on the largest eigenvalue of X, or `None`.
        norm: Upper bound on the spectral norm of X, or `None`.
        evidence: `"certified_bound"` (exact rational arithmetic on the
            stored binary64 values, rounded outward; the bounds of a zero
            part are exact) or `"user_assertion"` (from the asserted
            `alpha`, error bound and lower bound of a supplied encoding, or
            from a `dissipative_lower_bound` given to `ingest_cartesian` that
            improves on the structural lower bound; `upper`, `norm` and
            `centered_alpha` of that part stay structural).
    """

    access: Literal["zero", "pauli", "band", "block_encoding", "single_a"]
    identity: float
    centered_alpha: float | None
    lower: float | None
    upper: float | None
    norm: float | None
    evidence: Literal["certified_bound", "user_assertion"]


ZERO_PART = PartFacts(access="zero", identity=0.0, centered_alpha=0.0, lower=0.0, upper=0.0, norm=0.0,
                      evidence="certified_bound")


@dataclass(frozen=True, kw_only=True)
class CartesianAccess:
    """The parts of a compact generator A that a solver reads, with their facts.

    Attributes:
        dissipative: The operator input of L, or `None` when L is exactly
            zero or is available only through `single_a`.
        hamiltonian: The operator input of H, with the same convention.
        single_a: The supplied encoding of a general A when no separate parts
            exist, otherwise `None`.
        dissipative_facts: [`PartFacts`][nwqlib.operators.cartesian.PartFacts] of L.
        hamiltonian_facts: `PartFacts` of H.
        generator_norm: Upper bound on `||A||` of the original generator,
            the smaller of `||L|| + ||H||` from the part bounds and a direct
            bound when the representation gives one (`sum_P |c_P|` for a
            Pauli A, `alpha + delta` for a supplied A), or `None` when
            neither is known.
        parts_commute: Where an exact `[L, H] = 0` comes from, when the
            representation itself gives it: `"periodic_bands"` when both
            parts are periodic band operators on one grid, which are
            polynomials in the same cyclic shift `S` and so commute exactly.
            `None` when the representation gives no such fact; no
            commutator is computed.
    """

    dissipative: object
    hamiltonian: object
    single_a: object
    dissipative_facts: PartFacts
    hamiltonian_facts: PartFacts
    generator_norm: float | None
    parts_commute: str | None = None


def _pauli_facts(part, *, max_bytes):
    """Return the facts of an exactly Hermitian Pauli part, or ZERO_PART for an empty table.

    For `X = c_I I + sum_{P != I} c_P P`, each Pauli word has norm one, so
    `(c_I - R) I <= X <= (c_I + R) I` and `||X|| <= |c_I| + R` with
    `R = sum_{P != I} |c_P|`. The identity is the row with no X or Z bit,
    and each sum over the stored coefficients is the outward rounding of its
    exact value (``_numerics.outward_sum``). The masks, the magnitudes and
    the summands with the identity term take an untuned 25 bytes per term.
    """
    terms = part.pauli_terms()
    if not len(terms):
        return ZERO_PART
    _check_bytes(25 * len(terms), max_bytes, "Pauli part bounds")
    if np.any(terms.coefficients.imag != 0):
        raise ValueError("a Hermitian Pauli part requires real coefficients")
    values = terms.coefficients.real
    identity = ~(terms.x.any(axis=1) | terms.z.any(axis=1))
    # Coalesced words are distinct, so at most one row is the identity.
    center = float(values[identity][0]) if identity.any() else 0.0
    radius = np.abs(values[~identity])
    return PartFacts(access="pauli", identity=center, centered_alpha=outward_sum(radius, upward=True),
        lower=outward_sum(np.append(-radius, center), upward=False),
        upper=outward_sum(np.append(radius, center), upward=True),
        norm=outward_sum(np.append(radius, abs(center)), upward=True), evidence="certified_bound")


def _band_facts(part):
    """Return the facts of a Hermitian periodic band part, or ZERO_PART when it has no bands.

    For `X = sum_b beta_b S^b` with the unitary cyclic shift S, Hermitian X
    has a real `beta_0`, and `||X - beta_0 I|| <= R = sum_{b != 0} |beta_b|`
    gives `(beta_0 - R) I <= X <= (beta_0 + R) I` and `||X|| <= |beta_0| + R`.
    Each magnitude enters as the exact rational upper bound
    of ``_numerics.complex_magnitude_upper``.
    """
    bands = part.band()
    if not bands.offsets:
        return ZERO_PART
    center, radius = Fraction(0), Fraction(0)
    for offset, coefficient in zip(bands.offsets, bands.coefficients, strict=True):
        if offset == 0:
            center = Fraction(coefficient.real)
        else:
            radius += complex_magnitude_upper(coefficient)
    return PartFacts(access="band", identity=float(center), centered_alpha=_outward(radius, upward=True),
        lower=_outward(center - radius, upward=False), upper=_outward(center + radius, upward=True),
        norm=_outward(abs(center) + radius, upward=True), evidence="certified_bound")


def _supplied_bound(encoding):
    """Return the exact rational `alpha + delta` of a supplied encoding, or None when delta is unknown."""
    if encoding.error_bound is None:
        return None
    return Fraction(encoding.alpha) + Fraction(encoding.error_bound)


def _supplied_facts(part):
    """Return the facts of a supplied encoding of a Hermitian part X.

    A known error bound delta gives `||X|| <= alpha + delta`, so
    `-(alpha + delta)` and `alpha + delta` bound the spectrum. An asserted
    lower bound tightens the lower end when it is larger. With an unknown
    delta, only an asserted lower bound remains, and no norm bound exists.
    """
    supplied = part.supplied_block_encoding()
    bound = _supplied_bound(supplied.encoding)
    lowers = [Fraction(supplied.lower_bound)] if supplied.lower_bound is not None else []
    if bound is not None:
        lowers.append(-bound)
    return PartFacts(access="block_encoding", identity=0.0, centered_alpha=float(supplied.encoding.alpha),
        lower=_outward(max(lowers), upward=False) if lowers else None,
        upper=None if bound is None else _outward(bound, upward=True),
        norm=None if bound is None else _outward(bound, upward=True), evidence="user_assertion")


def _with_declared_lower(facts, declared):
    """Return the facts of L under the lower bound ``declared`` given to `ingest_cartesian`.

    Under the premise that ``declared`` is correct, `ell = max(ell_struct,
    declared)` is a lower bound of `lambda_min(L)`. A declaration that does not
    improve on the structural bound leaves the facts with their structural
    evidence. A larger one replaces the bound and makes the evidence
    `"user_assertion"`, which every fact depending on the shift inherits.
    `upper`, `norm` and `centered_alpha` read the stored data alone, so they
    stay as they are.

    The eigenvalues of a Hermitian L have the mean `c = tr(L)/D`, so
    `lambda_min(L) <= c`, and equality makes every eigenvalue equal to c, that
    is `L = c I`. In a coalesced Pauli sum or band, `facts.identity` is c, because
    every other word or shift has trace zero, and `facts.centered_alpha > 0` proves
    that L is not scalar. So a declaration above c, or equal to c for a non-scalar
    L, contradicts the premise and is refused. This also refuses a declaration at
    or above the structural upper bound, which is at least c and, for a non-scalar
    L, above it. Both are exact comparisons of the stored coefficients, so no
    tolerance applies.

    A false declaration strictly below c stays undetected: on `.5 I + .3 X`, whose
    smallest eigenvalue is .2, the value .25 passes. Accepting a declaration
    therefore does not verify it, and no spectrum computation is performed, so an
    accepted value stays `"user_assertion"`. Only Pauli and band facts reach this
    function, because `ingest_cartesian` refuses a declaration for a supplied
    encoding of L. A supplied encoding has no trace to compare: its `identity = 0`
    is a placeholder, and `.5 I` encoded with `alpha = .5` is a valid example.
    """
    if declared > facts.identity or (declared == facts.identity and facts.centered_alpha > 0):
        raise ValueError(f"dissipative_lower_bound {declared!r} exceeds the mean eigenvalue {facts.identity!r} of L, or "
                         "equals it for a non-scalar L, so it cannot bound the smallest eigenvalue from below. "
                         "State a smaller value that you know bounds the smallest eigenvalue, or leave the bound out")
    return replace(facts, lower=declared, evidence="user_assertion") if declared > facts.lower else facts


def _part_facts(part, *, max_bytes):
    representation = part.manifest.reference.representation
    if representation == "pauli":
        return _pauli_facts(part, max_bytes=max_bytes)
    if representation == "band":
        return _band_facts(part)
    return _supplied_facts(part)


def _pauli_cartesian_parts(operator, *, max_bytes):
    """Split a coalesced Pauli A into Hermitian Pauli parts L and H.

    With `A = sum_P c_P P` and `P† = P`, `L = sum_P Re(c_P) P` and
    `H = sum_P Im(c_P) P`. Taking real and imaginary parts of the stored
    coefficients is exact, and only parts whose coefficients are exactly
    zero are dropped, so no nonzero term is pruned. A Hermitian A is its own
    L and is not copied. The words of A are distinct, so each part keeps its
    rows in order without summing again and has the content hash of
    ingesting the same terms. Both parts, the selected rows and the
    transient byte copy take at most four L-byte copies of the table and
    two identity JSON shares per term (``inputs.ingest_pauli_masks``).
    """
    from .inputs import _distinct_pauli_input, _pauli_identity_bytes
    if operator.structure == "hermitian":
        return operator, None
    terms = operator.pauli_terms()
    q, w = terms.num_qubits, terms.x.shape[1]
    _check_bytes(len(terms) * (4 * (16 * w + 16) + 2 * _pauli_identity_bytes(q)), max_bytes, "Pauli Cartesian parts")
    parts = []
    for values in (terms.coefficients.real, terms.coefficients.imag):
        kept = values != 0
        parts.append(_distinct_pauli_input(q, terms.x[kept], terms.z[kept], values[kept].astype(np.complex128))
                     if kept.any() else None)
    return tuple(parts)


def cartesian_access(operator, *, max_bytes=None) -> CartesianAccess:
    """Return the Cartesian parts of a compact generator A and their structural facts.

    Supported inputs are a Pauli sum (split into Re and Im coefficient
    parts), a Cartesian
    input from `ingest_cartesian`, a Hermitian periodic band operator, and a
    supplied block encoding. A supplied encoding of a general A gives no
    separate parts (`single_a`), and then
    `||L||, ||H|| <= ||A|| <= alpha + delta` bound both parts, because
    `(A ± A†)/2` has norm at most `||A||`. No dense matrix is formed. A
    declared lower bound of a Cartesian input enters the facts of L through
    `_with_declared_lower`.

    Args:
        operator (OperatorInput): The generator A, holding its data.
        max_bytes (int | None): Byte limit of the Pauli split and label reads.
            Default `None`, the physical memory of this computer.

    Returns:
        access (CartesianAccess): The parts and their facts.

    Raises:
        ValueError: If A has another representation, such as a dense or
            sparse matrix, which has no compact Cartesian access here, or is
            a non-Hermitian band operator, whose parts must be given
            through `ingest_cartesian`, or if a declared lower bound of L
            exceeds the mean eigenvalue of L or equals it for a non-scalar L.
    """
    representation = operator.manifest.reference.representation
    direct = declared = None
    if representation == "pauli":
        parts = _pauli_cartesian_parts(operator, max_bytes=max_bytes)
        # ||A|| <= sum_P |c_P| for unitary Pauli words, which is never larger
        # than ||L|| + ||H|| from the Re/Im split, since |c| <= |Re c| + |Im c|.
        # A sum beyond the binary64 range gives no direct bound.
        try:
            direct = Fraction(outward_sum(complex_magnitudes_upper(operator.pauli_terms().coefficients), upward=True))
        except ValueError:
            direct = None
    elif representation == "cartesian":
        given = operator.cartesian_parts()
        parts, declared = (given.dissipative, given.hamiltonian), given.dissipative_lower_bound
    elif representation == "band":
        if operator.structure != "hermitian":
            raise ValueError("a non-Hermitian band generator needs its Hermitian parts through ingest_cartesian")
        parts = (operator, None)
    elif representation == "block_encoding":
        supplied = operator.supplied_block_encoding()
        if supplied.hermitian:
            parts = (operator, None)
        else:
            bound = _supplied_bound(supplied.encoding)
            facts = PartFacts(access="single_a", identity=0.0, centered_alpha=None,
                lower=None if bound is None else _outward(-bound, upward=False),
                upper=None if bound is None else _outward(bound, upward=True),
                norm=None if bound is None else _outward(bound, upward=True), evidence="user_assertion")
            return CartesianAccess(dissipative=None, hamiltonian=None, single_a=operator,
                                   dissipative_facts=facts, hamiltonian_facts=facts, generator_norm=facts.norm)
    else:
        raise ValueError(f"a {representation} generator has no compact Cartesian access; "
                         "give a Pauli sum, a Hermitian band, ingest_cartesian parts or a supplied block encoding")
    facts = tuple(ZERO_PART if part is None else _part_facts(part, max_bytes=max_bytes) for part in parts)
    if declared is not None:
        facts = (_with_declared_lower(facts[0], declared), facts[1])
    parts = tuple(None if part is None or fact.access == "zero" else part for part, fact in zip(parts, facts))
    bounds = [Fraction(facts[0].norm) + Fraction(facts[1].norm)] if None not in (facts[0].norm, facts[1].norm) else []
    if direct is not None:
        bounds.append(direct)
    return CartesianAccess(dissipative=parts[0], hamiltonian=parts[1], single_a=None,
                           dissipative_facts=facts[0], hamiltonian_facts=facts[1],
                           generator_norm=_outward(min(bounds), upward=True) if bounds else None,
                           parts_commute="periodic_bands" if facts[0].access == facts[1].access == "band" else None)


@dataclass(frozen=True, kw_only=True)
class ScalarCompensation:
    """The scalar shift and known decay for a lower bound `ell <= lambda_min(L)`.

    With `s_+ = max(0, -ell)` and `c = max(0, ell)`, the signed shift is
    `s = s_+ - c = -ell`, and `L_s = L + s I` is positive semidefinite.
    Because the identity commutes with A, `exp(-A t) = exp(s t) exp(-(A + s I) t)`,
    so a positive `c` is known physical decay `exp(-c t)` taken out of every
    application, and a positive `s_+` is the shift that makes the kernel's
    premise hold. A negative `ell` is an enclosure, not a proof that L has a
    negative eigenvalue. These relations need a valid `ell`, so a declared
    lower bound (`_with_declared_lower`) makes them conditional on that
    declaration.

    Attributes:
        psd_shift: The nonnegative shift `s_+`.
        scalar_decay: The nonnegative known decay rate `c`.
        processed_norm: Upper bound `B_{L_s}` on `||L + s I||`: `upper - ell`
            when an upper eigenvalue bound exists, else `norm + |s|`,
            rounded up.
    """

    psd_shift: float
    scalar_decay: float
    processed_norm: float

    @property
    def signed_shift(self) -> float:
        """The signed shift `s = psd_shift - scalar_decay`."""
        return self.psd_shift - self.scalar_decay


def scalar_compensation(facts: PartFacts) -> ScalarCompensation:
    """Return the P1 compensation of L from its facts.

    Args:
        facts (PartFacts): Facts of L, with a lower bound and an upper or
            norm bound.

    Returns:
        compensation (ScalarCompensation): The shift, decay and processed
            norm bound.

    Raises:
        ValueError: If the facts have no lower bound, or neither an upper
            nor a norm bound, so neither the PSD premise nor the kernel's
            norm bound is available.
    """
    if facts.lower is None:
        raise ValueError("the PSD shift of L needs a lower eigenvalue bound, which these facts lack")
    lower = Fraction(facts.lower)
    if facts.upper is not None:
        processed = Fraction(facts.upper) - lower
    elif facts.norm is not None:
        processed = Fraction(facts.norm) + abs(lower)
    else:
        raise ValueError("the kernel needs an upper bound on ||L + s I||, which these facts lack")
    return ScalarCompensation(psd_shift=max(0.0, -facts.lower), scalar_decay=max(0.0, facts.lower),
                              processed_norm=_outward(processed, upward=True))
