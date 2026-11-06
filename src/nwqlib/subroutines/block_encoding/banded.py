"""Banded (periodic Toeplitz) block encoding as an exact LCU of cyclic shifts.

A periodic banded Toeplitz operator is a circulant,
``A = sum_b beta_b S^b`` with ``S |x> = |x + 1 mod N>``, so it is exactly an
LCU of shift unitaries, the PREP-SELECT-PREP^dagger construction that
Childs and Wiebe, arXiv:1202.5822v1, give for two terms in Lemma 2 and
Fig. 1 (``subroutines/lcu/core.py`` states its general form), with
``alpha = sum_b |beta_b|`` and
``ceil(log2 B)`` address ancillas for ``B`` bands. Each shift is a
Draper-style modular adder (Draper, arXiv:quant-ph/0008033v1). With Qiskit's
``QFT |x> = N**-0.5 sum_k exp(2 pi i x k / N) |k>``,
``S^b = QFT^dagger D_b QFT`` with ``D_b = diag_k exp(2 pi i k b / N)``, and
``D_b`` factors into the phase ``P(2 pi b 2**q / N)`` on each system qubit
``q``. SELECT therefore shares one QFT/IQFT pair across all bands, applies
one UCRZ address table per system qubit, and combines the ``P(phi)``
half-phase corrections with coefficient phases in one address diagonal.
Building from a band specification needs no eigendecomposition and no
dense matrix.

PREP and SELECT follow the conventions of ``subroutines/lcu/core.py``.
Coefficient phases are absorbed into the unitaries that SELECT applies
(here, as controlled phases on the coefficient register), the control
register precedes the system register, and the encoded block sits at
statevector stride ``2**num_ancillas``.

A Hermitian specification, ``beta_{-b mod N} = conj(beta_b)`` for the
stored values after reducing offsets mod N, is built instead as a full
involution with paired labels (``_band_layout``). Each pair ``b, -b`` gets
adjacent labels ``2p`` and ``2p + 1`` with
``Q_{2p} = exp(i arg beta_b) S^b`` and ``Q_{2p+1} = Q_{2p}^dagger``, whose
angles and phase are the exact negations of those of ``Q_{2p}``. The
self-inverse offsets 0 and N/2 have real coefficients and take the labels
after the pairs. The permutation J swaps the labels of each pair and fixes
the others, so with r pairs and ``y = label >> 1``,
``J |y, d> = |y, d xor [y < r]>``. Since a pair's two PREP amplitudes are
equal, ``J |p> = |p>`` for ``|p> = PREP |0>``. On the labels of one pair
``Q J`` is ``[[0, V], [V^dagger, 0]]`` with ``V = Q_{2p}``, and on a
self-inverse label it is ``sign(beta) S^b``, a Hermitian involution. Hence

    U = PREP^dagger Q J PREP,  U^dagger = U,  U**2 = I,  <0|U|0> = A/alpha,

on the whole space. The conjugate shell around a structured Hermitian
SELECT follows Sünderhauf, Campbell and Camps, arXiv:2302.10949v2,
Sec. 2.4. J is a product of at most
``a - 1`` disjoint multi-controlled X gates on label bit 0, one for each set
bit of r, with no workspace (``_pair_swap_cubes``). The shift by N/2 is an X
on the most significant system qubit, and the sign of a negative
self-inverse coefficient is a phase flip on its label (an H-conjugated
multi-controlled X), so neither uses a rounded angle such as ``exp(i pi)``.
A non-Hermitian specification keeps the unpaired shift LCU above, whose
unitary is not an involution because its block is not Hermitian.

This implementation is periodic-only. For broader structured banded
block-encoding constructions, see Camps, Lin, Van Beeumen, and Yang,
arXiv:2203.10236v4. Their banded-circulant circuit (Sec. 4.2, built on
Theorem 4.1) uses sparse-access oracles and encodes ``A / s`` when every
entry has magnitude at most one, with ``s`` the padded band count. Scaling
``A`` into that range gives ``alpha = s * max_b |beta_b|``. The shift LCU
here has ``alpha = sum_b |beta_b|``, which is never larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Any
import numpy as np

from nwqlib._preparation_laws import direct_preparation_cx_bound

from nwqlib.subroutines._multiplexors import (
    append_control_diagonal_phases,
    append_uniformly_controlled_rz,
    multiplexor_resource_law,
)
from nwqlib.subroutines._serialization import complex_to_dict as _complex_to_dict
from nwqlib.subroutines.block_encoding.core import (
    BlockEncoding,
    _append_mcx,
    _append_phase_flip,
    _base_metadata,
    _mcx_cx,
    _norm_bound_from_sums,
)


@dataclass(frozen=True, kw_only=True)
class BandSpecification:
    """A periodic banded Toeplitz (circulant) matrix given by its bands, without forming the matrix.

    The matrix is `A = sum_b beta_b S^b` with the cyclic shift
    `S |x> = |x + 1 mod 2**num_qubits>`. Build it with keyword arguments,
    for example
    `BandSpecification(offsets=(-1, 0, 1), coefficients=(-1, 2, -1), num_qubits=3)`
    for `A = 2 I - S - S^-1` on 8 points, and pass it to
    [`build_block_encoding`][nwqlib.subroutines.block_encoding.build_block_encoding].
    All three arguments are required. The builder checks that `num_qubits`
    is positive, that there is one coefficient per offset and at least one
    band, that the offsets are distinct modulo `2**num_qubits` and that no
    coefficient is zero.

    This specification is periodic-only. For broader structured banded
    block encodings, see Camps et al., arXiv:2203.10236v4, Sec. 4.

    Args:
        offsets: Signed band offsets `b`. Band `b`
            couples `|x>` to `|x + b mod 2**num_qubits>`.
        coefficients: One nonzero complex coefficient
            `beta_b` per offset.
        num_qubits: Number of system qubits (dimension
            `2**num_qubits`).
    """

    offsets: tuple[int, ...]
    coefficients: tuple[complex, ...]
    num_qubits: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like band-specification record."""

        return {
            "offsets": [int(offset) for offset in self.offsets],
            "coefficients": [_complex_to_dict(value) for value in self.coefficients],
            "num_qubits": self.num_qubits,
        }


def _detect_banded_structure_with_error(
    matrix: Any, *, atol: float | None = None
) -> tuple[BandSpecification, float]:
    """Detect the bands and, in the same scan, bound the difference between the supplied operator and its circulant.

    Offsets above ``N/2`` are reported as negative shifts, sorted ascending.
    Band ``b`` is the wrapped diagonal ``A[(j + b) mod N, j]``.

    The default ``atol`` is the window ``tau = 1e-12 M`` with
    ``M = max_ij |A_ij|``. For a nonzero finite scalar ``a``, unchanged
    dimension and exact arithmetic, ``M(aA) = |a| M(A)``, every difference
    within one wrapped diagonal and every symbol magnitude is multiplied by
    ``|a|``, and ``tau`` scales likewise. The same offsets are therefore kept,
    their symbols are multiplied by ``a``, ``C(aA) = a C(A)`` and
    ``alpha(aA) = |a| alpha(A)``. This is a statement about the detector. A
    route that needs a Hermitian operator can lose that premise under a complex
    ``a``, and a real ``a`` keeps it. With an explicit absolute ``atol``, the
    relation requires the caller to scale it by ``|a|``. Finite-arithmetic ties,
    nonfinite rescalings and rounded input changes are outside this exact
    equality. ``M = 0`` gives the zero operator, which finds no band and takes
    the zero-matrix refusal.

    Detection succeeding does not mean ``A = C``. For the actual residual
    ``E = A - C`` of the detected circulant ``C``, a kept band contributes its
    diagonal minus its symbol and a dropped band its whole diagonal, so a small
    symbol does not make that residual zero. For a general real or complex
    matrix ``E``,

        ||E||_2**2 = rho(E^dagger E) <= ||E^dagger E||_inf
                  <= ||E^dagger||_inf ||E||_inf = ||E||_1 ||E||_inf,

    so ``||E||_2 <= sqrt(max_j sum_i |E_ij| * max_i sum_j |E_ij|)``. That is an
    identity in real arithmetic. The returned bound on the represented residual
    rests on the outward rounding of the entries and of the sums
    (``_norm_bound_from_sums``).
    """

    array = np.asarray(matrix, dtype=complex)
    dimension = array.shape[0]
    if atol is None:
        # Relative equality for an explicit "banded" request, registered in
        # docs/ENGINEERING_CONSTANTS.md. Automatic routing passes its roundoff
        # window, 64 eps max(1, n) max|A| (core._detect_circulant). Revisit with
        # a changed precision or detector.
        atol = 1.0e-12 * float(np.max(np.abs(array)))

    columns = np.arange(dimension)
    row_errors = np.zeros(dimension)
    column_errors = np.zeros(dimension)
    bands: list[tuple[int, complex]] = []
    for offset in range(dimension):
        diagonal = array[(columns + offset) % dimension, columns]
        symbol = diagonal[0]
        if np.max(np.abs(diagonal - symbol)) > atol:
            from nwqlib.algorithms.protocol import ApplicabilityError
            raise ApplicabilityError(
                "matrix is not a periodic banded Toeplitz (circulant) operator; "
                "the 'banded' implementation supports only the periodic path — "
                "use 'pauli_lcu' or 'dense_dilation' for unstructured input"
            )
        if abs(symbol) > atol:
            signed = offset - dimension if offset > dimension // 2 else offset
            bands.append((signed, complex(symbol)))
            residual = np.abs(diagonal - symbol)
        else:
            residual = np.abs(diagonal)
        np.nextafter(residual, np.inf, out=residual, where=residual != 0.0)
        row_errors[(columns + offset) % dimension] += residual
        column_errors += residual
    if not bands:
        raise ValueError("banded structure detection found only zero bands (zero matrix)")
    bands.sort(key=lambda item: item[0])
    return (
        BandSpecification(
            offsets=tuple(offset for offset, _ in bands),
            coefficients=tuple(coefficient for _, coefficient in bands),
            num_qubits=int(np.log2(dimension)),
        ),
        _norm_bound_from_sums(row_errors, column_errors),
    )


def _hermitian_band_projection(specification: BandSpecification,
                               detection_error: float) -> tuple[BandSpecification, float]:
    """Return the Hermitian projection of a circulant detected from a proved Hermitian input, with its error bound.

    A proved Hermitian A has ``A[b, 0] = conj(A[0, b])``, but ``A[b, 0]`` and
    ``A[N - b, 0]``, which give the two detected bands, lie on two diagonals
    with independent roundoff, so ``beta_{-b} = conj(beta_b)`` usually fails
    by a few ulps. For ``C = sum_b beta_b S^b`` take

        Pi(C) = (C + C^dagger)/2 = sum_b (beta_b + conj(beta_{-b}))/2 S^b,

    with a missing band counted as zero. Since A is Hermitian,
    ``A - Pi(C) = ((A - C) + (A - C)^dagger)/2``, so
    ``||A - Pi(C)||_2 <= ||A - C||_2`` and ``detection_error``, the detector's
    bound on the right side, still bounds the left. The bound
    ``detection_error + sum_b |beta_b - beta'_b|`` is valid too, but looser,
    since it charges the projection that this relation shows costs nothing.

    Only the rounding of the stored averages adds to it. Each S^b has norm
    one, so that term is at most ``sum_b |Delta beta_b|``, and an average of
    two binary64 values has ``|Delta beta_b| <= 2**-53 |beta'_b| (1 + 2**-53)``.
    Counting ``2**-52 |beta'_b|`` leaves room for the rounding of ``abs`` and
    the sum in the normal range without overflow, and subnormal terms are not
    counted, so this is a numerical record rather than an arithmetic enclosure.

    One rounded average gives both members of a pair, so
    ``beta'_{-b} == conj(beta'_b)`` holds exactly and ``_band_layout`` pairs
    them. The self-conjugate offsets 0 and N/2 keep their real parts, which
    is exact. A pair already conjugate has an exact average and adds nothing,
    so a specification already paired returns unchanged.

    A band can project to exact zero, for example ``Re(beta)`` of an imaginary
    roundoff band at ``N/2`` in ``I + 1e-15 Y``, or a pair with
    ``beta_b = -conj(beta_{-b})``. It leaves the specification, which forbids
    zero coefficients, and ``Pi(C)`` and its bound stay the same. Every nonzero
    coefficient stays, however small. The caller establishes
    the Hermitian input (``core._detect_circulant``); this function cannot,
    and a non-Hermitian input must not reach it.
    """
    dimension = 1 << specification.num_qubits
    bands = {int(offset) % dimension: complex(value)
             for offset, value in zip(specification.offsets, specification.coefficients, strict=True)}
    projected, rounded = {}, []
    for offset in sorted(bands.keys() | {(-offset) % dimension for offset in bands}):
        partner = (-offset) % dimension
        value, conjugate = bands.get(offset, 0j), bands.get(partner, 0j).conjugate()
        if offset == partner:
            projected[offset] = complex(value.real)
        elif offset < partner:
            average = complex(.5 * (value.real + conjugate.real), .5 * (value.imag + conjugate.imag))
            projected[offset], projected[partner] = average, average.conjugate()
            if value != conjugate:
                rounded += [abs(average), abs(average)]
    if projected == bands:
        return specification, detection_error
    signed = sorted((offset - dimension if offset > dimension // 2 else offset, value)
                    for offset, value in projected.items() if value != 0)
    error = detection_error
    if rounded:
        error = float(np.nextafter(detection_error + 2.0**-52 * fsum(rounded), np.inf))
    return (BandSpecification(offsets=tuple(offset for offset, _ in signed),
                              coefficients=tuple(value for _, value in signed),
                              num_qubits=specification.num_qubits), error)


def _validate_band_specification(specification: BandSpecification) -> None:
    """Validate offsets, coefficients, and qubit count of a band specification."""

    if specification.num_qubits <= 0:
        raise ValueError("band specification num_qubits must be positive")
    if len(specification.offsets) != len(specification.coefficients):
        raise ValueError("band specification needs one coefficient per offset")
    if not specification.offsets:
        raise ValueError("band specification requires at least one band")
    dimension = 2**specification.num_qubits
    reduced = [int(offset) % dimension for offset in specification.offsets]
    if len(set(reduced)) != len(reduced):
        raise ValueError(
            f"band offsets must be distinct modulo 2**num_qubits = {dimension}"
        )
    if any(abs(complex(value)) == 0.0 for value in specification.coefficients):
        raise ValueError("band coefficients must be nonzero; drop zero bands")


@dataclass(frozen=True)
class _BandLayout:
    """Address labels of the shift LCU of one band specification.

    Label ``l`` carries the PREP magnitude ``magnitudes[l]``, the shift
    ``offsets[l]`` and phase ``phases[l]`` that the QFT tables apply, and,
    for a paired layout, the half shift or sign recorded below. A paired
    label ``2p + 1`` stores the exact negations of label ``2p``. A
    self-inverse label has offset 0 and phase 0 in the QFT tables.
    """

    num_system_qubits: int
    address_qubits: int
    paired: bool
    magnitudes: tuple[float, ...]
    offsets: tuple[int, ...]
    phases: tuple[float, ...]
    pair_count: int
    half_shift_label: int | None
    negative_labels: tuple[int, ...]


def _band_layout(specification: BandSpecification) -> _BandLayout:
    """Validate the specification and return its paired layout when it is Hermitian, else its unpaired layout.

    Hermiticity is tested on the stored values with offsets reduced mod N:
    ``beta_{-b mod N} == conj(beta_b)`` exactly, in O(B) time and space at
    any register width. It makes the self-inverse coefficients at offsets
    0 and N/2 real. The pairs are ordered by their offset in (0, N/2), then
    the self-inverse labels follow, offset 0 first. Each band keeps one
    label, so the address width ``ceil(log2 B)`` is unchanged. The unpaired
    layout keeps the specification's order, offsets and ``arg beta`` phases.
    """
    _validate_band_specification(specification)
    n = specification.num_qubits
    dimension = 1 << n
    band_map = {int(offset) % dimension: complex(value)
                for offset, value in zip(specification.offsets, specification.coefficients, strict=True)}
    address_qubits = (len(band_map) - 1).bit_length()
    if not all(value == band_map.get((-offset) % dimension, 0).conjugate() for offset, value in band_map.items()):
        coefficients = np.asarray(specification.coefficients, dtype=complex)
        return _BandLayout(
            num_system_qubits=n, address_qubits=address_qubits, paired=False,
            magnitudes=tuple(float(value) for value in np.abs(coefficients)),
            offsets=tuple(int(offset) for offset in specification.offsets),
            phases=tuple(float(value) for value in np.angle(coefficients)),
            pair_count=0, half_shift_label=None, negative_labels=())
    magnitudes: list[float] = []
    offsets: list[int] = []
    phases: list[float] = []
    for offset in sorted(offset for offset in band_map if 0 < offset < dimension - offset):
        magnitude, phase = abs(band_map[offset]), float(np.angle(band_map[offset]))
        magnitudes += [magnitude, magnitude]
        offsets += [offset, -offset]
        phases += [phase, -phase]
    pair_count = len(magnitudes) // 2
    half_shift_label = None
    negative_labels = []
    for offset in (0, dimension // 2):
        if offset in band_map and (offset == 0 or 2 * offset == dimension):
            value = band_map[offset].real
            if offset:
                half_shift_label = len(magnitudes)
            if value < 0.0:
                negative_labels.append(len(magnitudes))
            magnitudes.append(abs(value))
            offsets.append(0)
            phases.append(0.0)
    return _BandLayout(num_system_qubits=n, address_qubits=address_qubits, paired=True,
                       magnitudes=tuple(magnitudes), offsets=tuple(offsets), phases=tuple(phases),
                       pair_count=pair_count, half_shift_label=half_shift_label,
                       negative_labels=tuple(negative_labels))


def _pair_swap_cubes(pair_count: int, address_qubits: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return the label-bit controls and control state of each X on label bit 0 whose product is J.

    With ``y = label >> 1`` on ``a - 1`` bits and r pairs, ``[y < r]`` is the
    disjoint union over the set bits i of r of the prefix cubes
    ``y_j = r_j (j > i), y_i = 0``. Each cube is one X on label bit 0
    controlled by label bits ``i + 1, ..., a - 1`` (``y_i, ..., y_{a-2}``)
    with state ``(r >> i) xor 1``. The cubes are disjoint, so the X gates
    commute and their product flips bit 0 exactly when ``y < r``. When
    every label is paired, ``2r = 2**a``, J is one X without controls.
    """
    if not pair_count:
        return ()
    if 2 * pair_count == 1 << address_qubits:
        return (((), 0),)
    return tuple((tuple(range(bit + 1, address_qubits)), (pair_count >> bit) ^ 1)
                 for bit in range(address_qubits - 1) if pair_count >> bit & 1)


def _band_shifts(layout: _BandLayout) -> bool:
    """Whether the layout needs the QFT tables, that is, it has a shift other than 0 and N/2.

    The address diagonal does not depend on this: a band ``beta I`` with
    ``beta`` not real has no shift but needs the phase ``arg beta``.
    """
    return any(layout.offsets)


def _band_phase_diagonal(layout: _BandLayout) -> list[float] | None:
    """Return the address-diagonal phases when any is nonzero, else None (no diagonal gate)."""
    phases = _band_diagonal_phases(layout, _band_angle_tables(layout))
    return phases if any(phase != 0.0 for phase in phases) else None


def _band_angle_tables(layout: _BandLayout) -> list[list[float]]:
    """Return, per system qubit q, the RZ angle of each label's offset b, ``2 pi 2**q b / N`` reduced modulo 2 pi.

    The reduction ``(2**q |b|) mod N`` is exact in Python integers, so the
    angle ``2 pi r / N`` has an error of a few units in the last place for
    every stored offset, and the sign of b is applied last, so a negated
    offset gives the exactly negated angle. ``P(phi) = exp(i phi / 2) RZ(phi)``
    has period 2 pi, and ``_band_diagonal_phases`` takes its half angles from
    these same reduced angles, so the reduction leaves the operator unchanged.
    """
    dimension = 1 << layout.num_system_qubits
    return [[(-1.0 if offset < 0 else 1.0) * 2.0 * np.pi * (((abs(offset) << qubit_index) % dimension) / dimension)
             for offset in layout.offsets]
            for qubit_index in range(layout.num_system_qubits)]


def _band_diagonal_phases(layout: _BandLayout, angle_tables: list[list[float]]) -> list[float]:
    """Return the address-diagonal phase of each label: its coefficient phase plus half of its RZ angles.

    ``P(phi) = exp(i phi / 2) RZ(phi)``, so the RZ tables of SELECT need the
    half-angle correction. A paired label ``2p + 1`` gets the exact negation
    of label ``2p``'s phase, since its phase and angles are exact negations
    and rounded sums and products are odd functions. A layout without
    shifts has zero angles, so its phases are the coefficient phases.
    """
    return [float(phase) + 0.5 * sum(table[label] for table in angle_tables)
            for label, phase in enumerate(layout.phases)]



def _band_core_cx(layout: _BandLayout, num_controls: int) -> int:
    """Return the CX of the SELECT core with k external controls, without building it.

    With m = a + k selecting qubits: one RZ multiplexor of ``2**m`` CX per
    system qubit when the layout has shifts; the address diagonal of
    ``2**m - 2`` CX (``multiplexor_resource_law``) when a phase is nonzero,
    with or without shifts; the X gates of J with k more controls each; the
    half shift, an X with m controls; and each sign, a phase flip on one
    basis state of the m qubits, an X with m - 1 controls between two H.
    The shared QFT pair and PREP pair are the shell (``_band_shell_cx``).
    """
    m = layout.address_qubits + num_controls
    cx = _paired_label_cx(layout, num_controls)
    if _band_shifts(layout):
        cx += layout.num_system_qubits * multiplexor_resource_law("ucrz", control_qubits=m)["basis_cx_gates"]
    if _band_phase_diagonal(layout) is not None:
        cx += multiplexor_resource_law("diagonal", control_qubits=m)["basis_cx_gates"]
    return cx


def _paired_label_cx(layout: _BandLayout, num_controls: int) -> int:
    """Return the CX of J, the half shift and the sign flips with k external controls (zero for an unpaired layout)."""
    m = layout.address_qubits + num_controls
    cx = sum(_mcx_cx(len(controls) + num_controls)
             for controls, _ in _pair_swap_cubes(layout.pair_count, layout.address_qubits))
    if layout.half_shift_label is not None:
        cx += _mcx_cx(m)
    return cx + (len(layout.negative_labels) * _mcx_cx(m - 1) if layout.negative_labels and m else 0)


def _band_shell_cx(layout: _BandLayout) -> int:
    """Return the CX of the never-controlled shell: the PREP pair and, when shifts need it, the QFT pair.

    Each QFT on n qubits has ``n(n-1)/2`` controlled phases of two CX and
    ``n // 2`` swaps of three CX. The PREP pair is two positive-amplitude
    trees, counted by their slot bound (``direct_preparation_cx_bound``).
    """
    n = layout.num_system_qubits
    qft_pair = 2 * n * (n - 1) + 6 * (n // 2) if _band_shifts(layout) else 0
    return qft_pair + 2 * direct_preparation_cx_bound(layout.address_qubits, complex_phases=False)


def _band_query_cx(layout: _BandLayout, num_controls: int) -> int:
    """Return the CX of one query with k external controls (``core.block_encoding_query_cx``)."""
    return _band_shell_cx(layout) + _band_core_cx(layout, num_controls)


def _band_gate_counts(layout: _BandLayout) -> dict[str, Any]:
    """Return the uncontrolled gate counts that the per-query CX law and the LCHS census read.

    ``backends.resources.block_encoding_per_query_cx`` adds the QFT pairs, the
    RZ multiplexors, the address diagonal, the PREP pair and
    ``paired_label_cx``, the CX of J, the half shift and the signs, which
    sum to ``_band_query_cx(layout, 0)``. ``pair_swap_controls`` lists the
    control count of each X of J.
    """
    a = layout.address_qubits
    shifts = _band_shifts(layout)
    diagonal_active = _band_phase_diagonal(layout) is not None
    ucrz_law = multiplexor_resource_law("ucrz", control_qubits=a)
    diagonal_law = multiplexor_resource_law("diagonal", control_qubits=a)
    return {
        "qft_pairs": int(shifts),
        "ucrz_multiplexor_gates": layout.num_system_qubits if shifts else 0,
        "ucrz_table_entries_per_gate": ucrz_law["table_entries"],
        "ucrz_basis_cx_per_gate": ucrz_law["basis_cx_gates"],
        "control_diagonal_gates": int(diagonal_active and a > 0),
        "control_diagonal_table_entries": diagonal_law["table_entries"],
        "control_diagonal_basis_cx": diagonal_law["basis_cx_gates"] if diagonal_active else 0,
        "prep_register_qubits": a,
        "prep_pair_cx": 2 * direct_preparation_cx_bound(a, complex_phases=False),
        "paired_labels": layout.paired,
        "pair_count": layout.pair_count,
        "pair_swap_controls": [len(controls) for controls, _ in _pair_swap_cubes(layout.pair_count, a)],
        "half_shift_label": layout.half_shift_label,
        "negative_self_inverse_labels": list(layout.negative_labels),
        "paired_label_cx": _paired_label_cx(layout, 0),
    }


def _band_structure_facts(layout: _BandLayout, detection_error: float,
                          *, hermitian_input: bool = False) -> dict[str, bool | None]:
    """Return the three structural facts (``core.STRUCTURE_FACT_KEYS``) of the circuit of this layout.

    The three have separate proofs. The paired layout is Hermitian in its
    stored values and its circuit is an involution (module docstring). An
    unpaired layout has a non-Hermitian block, so its unitary is no
    involution. The target's Hermiticity comes from the supplied input:
    ``hermitian_input`` states that its stored values proved it
    (``core._detect_circulant``), whatever the detection error. Otherwise a
    circulant detected with a nonzero error only approximates that target,
    whose Hermiticity then stays unknown.
    """
    target = True if hermitian_input else layout.paired if detection_error == 0.0 else None
    return {"target_operator_is_hermitian": target,
            "encoded_operator_is_hermitian": layout.paired,
            "full_unitary_involution": layout.paired}


def _band_query_census(layout: _BandLayout) -> dict[str, Any]:
    """Return the census of the four query sequences with one control (``core._query_census``)."""
    from nwqlib.subroutines.block_encoding.core import _query_census

    return _query_census(shell="PREP and QFT" if _band_shifts(layout) else "PREP",
                         shell_cx=_band_shell_cx(layout), core_cx=_band_core_cx(layout, 0),
                         controlled_core_cx=_band_core_cx(layout, 1),
                         shell_bound_cx=2 * direct_preparation_cx_bound(layout.address_qubits, complex_phases=False),
                         core_bound_cx=_paired_label_cx(layout, 0),
                         controlled_core_bound_cx=_paired_label_cx(layout, 1))




def _band_circuit(layout: _BandLayout, *, num_controls: int = 0, ctrl_state: int = 0, alpha: float | None = None):
    """Return the shift LCU of ``layout`` on [external controls, address, system], its core controlled on ``ctrl_state``.

    The order is PREP, J, QFT, RZ multiplexors, address diagonal, inverse
    QFT, half shift, signs and PREP^dagger. J acts on the address and the
    QFT on the system, so they commute, and the written product is
    ``PREP^dagger Q J PREP`` with ``Q`` the block-diagonal SELECT. PREP and
    the QFT pair form the shell and are never controlled. With k external
    controls each core table runs over the controls and the address and is
    the identity unless the controls hold ``ctrl_state``
    (``core._controlled_table``), and J, the half shift and the signs gain
    the k controls, which gives ``C(U) = shell^dagger C(Q J) shell``. A layout
    whose only shifts are 0 and N/2 skips the QFT pair and its RZ tables,
    which would act as the identity, but keeps the address diagonal of any
    nonzero coefficient phase. That diagonal acts on the address and the
    controls only, so it commutes with the QFT pair.
    """
    from nwqlib.subroutines.block_encoding.core import _controlled_table
    from nwqlib.subroutines.state_preparation import build_qiskit_state_preparation
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import QFTGate

    k, a, n = num_controls, layout.address_qubits, layout.num_system_qubits
    magnitudes = np.asarray(layout.magnitudes, dtype=float)
    alpha = float(np.sum(magnitudes)) if alpha is None else alpha
    registers = [QuantumRegister(k, "control")] if k else []
    if a:
        registers.append(QuantumRegister(a, "lcu_control"))
    system = QuantumRegister(n, "system")
    circuit = QuantumCircuit(*registers, system, name="c_banded_shift_lcu" if k else "banded_shift_lcu")
    external, address = list(circuit.qubits[:k]), list(circuit.qubits[k:k + a])
    selectors = external + address
    if a:
        padded = np.zeros(2**a, dtype=float)
        padded[:magnitudes.size] = magnitudes
        prepare = build_qiskit_state_preparation(np.sqrt(padded / alpha).astype(complex)).circuit.to_gate()
        circuit.append(prepare, address)
    for controls, state in _pair_swap_cubes(layout.pair_count, a):
        _append_mcx(circuit, external + [address[bit] for bit in controls], ctrl_state | (state << k), address[0])
    shifts = _band_shifts(layout)
    diagonal_phases = _band_phase_diagonal(layout)
    if shifts:
        # SELECT shares one QFT/IQFT pair. Each system qubit q receives one UCRZ
        # table with angle 2*pi*b*2**q/N for band b. The P(phi) = exp(i phi/2) RZ(phi)
        # corrections and coefficient phases share one address diagonal.
        circuit.append(QFTGate(n), list(system))
        angle_tables = _band_angle_tables(layout)
        for qubit_index in range(n):
            append_uniformly_controlled_rz(circuit, system[qubit_index], selectors,
                                           _controlled_table(angle_tables[qubit_index], a, k, ctrl_state, 0.0))
    if diagonal_phases is not None:
        append_control_diagonal_phases(circuit, selectors, _controlled_table(diagonal_phases, a, k, ctrl_state, 0.0))
    if shifts:
        circuit.append(QFTGate(n).inverse(), list(system))
    if layout.half_shift_label is not None:
        # x + N/2 mod N flips only the most significant bit of x.
        _append_mcx(circuit, selectors, ctrl_state | (layout.half_shift_label << k), system[n - 1])
    for label in layout.negative_labels:
        _append_phase_flip(circuit, selectors, ctrl_state | (label << k), system[0])
    if a:
        circuit.append(prepare.inverse(), address)
    return circuit


def build_banded_block_encoding(
    operator: Any,
    *,
    requested_implementation: str = "banded",
) -> BlockEncoding:
    """Block-encode a periodic banded Toeplitz operator as a linear combination of cyclic shifts.

    The address PREP is the direct magnitude and phase tree of
    [`build_qiskit_state_preparation`][nwqlib.subroutines.state_preparation.direct.build_qiskit_state_preparation]
    over the ``2**a`` band addresses, where ``a`` is the number of address
    qubits. It is part of this circuit rather than a separate
    state-preparation block of a `Program` (NWQLib's description of a
    circuit as named steps), so the Run's memory check of declared direct
    state preparations does not count it. Planning checks it
    at ``64 * 2**a * (q + 16)`` bytes and ``32 * 2**a * q**2`` work for q
    system qubits. With at most ``2**q`` distinct bands, the work check
    allows at most ``2**13`` addresses at a `max_work` of ``10**8`` and
    ``2**16`` at ``10**9``.

    When the stored coefficients satisfy `beta_{-b} = conj(beta_b)` with
    offsets taken modulo `2**q`, the operator is Hermitian and the circuit
    is built so that its whole unitary `U` satisfies `U = U^dagger` and
    `U @ U = I` (paired labels, see the module docstring). It then adds at
    most `a + 2` multi-controlled X gates: `a - 1` for the label exchange, one
    for the half shift and two for the signs of negative self-inverse
    coefficients. Otherwise the circuit is the
    unpaired shift combination. The metadata records which one under
    `"full_unitary_involution"`.

    Args:
        operator (BandSpecification): The bands. No dense matrix is formed.
            `build_block_encoding` detects bands in a dense matrix.
        requested_implementation (str): Default `"banded"`. Implementation
            name recorded as requested.

    Returns:
        encoding (BlockEncoding): The circuit with `alpha = sum_b |beta_b|`,
            `ceil(log2 num_bands)` ancillas in the low-order `lcu_control`
            register, and zero algebraic error for the given specification.

    Raises:
        TypeError: If `operator` is not a `BandSpecification`.
        ValueError: For invalid band specifications.
    """
    if not isinstance(operator, BandSpecification):
        raise TypeError("build_banded_block_encoding requires a BandSpecification")
    specification = operator
    layout = _band_layout(specification)
    magnitudes = np.asarray(layout.magnitudes, dtype=float)
    alpha = float(np.sum(magnitudes))
    circuit = _band_circuit(layout, alpha=alpha)
    metadata = _base_metadata(
        resolved="banded",
        requested=requested_implementation,
        preprocessing_label="scalable_oracle",
        preprocessing_note="band specification input; no dense matrix materialized anywhere",
        register_order=("lcu_control", "system"),
    )
    metadata["band_specification"] = specification.to_dict()
    metadata["num_bands"] = len(specification.offsets)
    metadata["structure_detection"] = "given_band_specification"
    metadata.update(_band_structure_facts(layout, 0.0))
    # Structural law inputs consumed unchanged by the per-query CX law and the
    # LCHS compiled-SELECT census.
    metadata["periodic_gate_counts"] = _band_gate_counts(layout)
    metadata["query_census"] = _band_query_census(layout)
    return BlockEncoding(
        circuit=circuit,
        alpha=alpha,
        num_ancillas=layout.address_qubits,
        system_qubits=specification.num_qubits,
        error_bound=0.0,
        implementation="banded",
        metadata=metadata,
    )


__all__ = [
    "BandSpecification",
    "build_banded_block_encoding",
]
