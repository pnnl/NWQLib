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
half-phase corrections with coefficient phases in one address diagonal. The
specification path needs no eigendecomposition and no dense matrix.

PREP/SELECT conventions follow ``subroutines/lcu/core.py``: coefficient
phases are absorbed into the unitaries that SELECT applies (here, as
controlled phases on the coefficient register), the control register precedes the system
register, and the encoded block sits at statevector stride
``2**num_ancillas``.

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
    _base_metadata,
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
    """Detect bands and bound the supplied-to-structured operator difference during the same scan.

    The default ``atol`` is ``1e-12 * max(1, max|A|)``. Offsets above ``N/2``
    are reported as negative shifts, sorted ascending. Band ``b`` is the
    wrapped diagonal ``A[(j + b) mod N, j]``. The returned error is
    ``sqrt(max row sum * max column sum)`` of ``|A - C|`` for the detected
    circulant ``C``, an outward-rounded spectral-norm bound.
    """

    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("banded structure detection requires a square matrix")
    dimension = array.shape[0]
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("banded structure detection requires a power-of-two dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("banded structure detection requires finite matrix entries")
    if atol is None:
        # Scaled machine-precision equality for an explicit "banded" request,
        # registered in docs/ENGINEERING_CONSTANTS.md. Automatic routing
        # passes atol=0. Revisit with a changed precision or detector.
        atol = 1.0e-12 * max(1.0, float(np.max(np.abs(array))))
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("banded structure detection atol must be finite and nonnegative")

    columns = np.arange(dimension)
    row_errors = np.zeros(dimension)
    column_errors = np.zeros(dimension)
    bands: list[tuple[int, complex]] = []
    for offset in range(dimension):
        diagonal = array[(columns + offset) % dimension, columns]
        symbol = diagonal[0]
        if np.max(np.abs(diagonal - symbol)) > atol:
            raise ValueError(
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


def build_banded_block_encoding(
    operator: Any,
    *,
    requested_implementation: str = "banded",
) -> BlockEncoding:
    """Block-encode a periodic banded Toeplitz operator as a linear combination of cyclic shifts.

    The address PREP is the direct magnitude and phase tree of
    [`build_qiskit_state_preparation`][nwqlib.subroutines.state_preparation.direct.build_qiskit_state_preparation]
    over the `2**a` band addresses. It is not a declared PREP block, so a
    Run's `max_direct_amplitudes` does not apply to it. Planning checks it
    at `64 * 2**a * (q + 16)` bytes and `32 * 2**a * q**2` work for q
    system qubits, and with at most `2**q` distinct bands that allows at
    most `2**13` addresses at a `max_work` of `10**8` and `2**16` at the
    default of `10**9`, which QLS shares.

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
    from nwqlib.subroutines.state_preparation import build_qiskit_state_preparation
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import QFTGate

    if not isinstance(operator, BandSpecification):
        raise TypeError("build_banded_block_encoding requires a BandSpecification")
    specification = operator
    _validate_band_specification(specification)

    num_system_qubits = specification.num_qubits
    dimension = 2**num_system_qubits
    num_bands = len(specification.offsets)
    num_control_qubits = (num_bands - 1).bit_length()
    coefficients = np.asarray(specification.coefficients, dtype=complex)
    magnitudes = np.abs(coefficients)
    alpha = float(np.sum(magnitudes))
    phases = np.angle(coefficients)

    system = QuantumRegister(num_system_qubits, "system")
    if num_control_qubits:
        control = QuantumRegister(num_control_qubits, "lcu_control")
        circuit = QuantumCircuit(control, system, name="banded_shift_lcu")
        control_qubits = list(control)
        padded = np.zeros(2**num_control_qubits, dtype=float)
        padded[:num_bands] = magnitudes
        prepare = build_qiskit_state_preparation(
            np.sqrt(padded / alpha).astype(complex)
        ).circuit.to_gate()
        circuit.append(prepare, control_qubits)
    else:
        circuit = QuantumCircuit(system, name="banded_shift_lcu")
        control_qubits = []

    # SELECT shares one QFT/IQFT pair. Each system qubit q receives one UCRZ
    # table with angle 2*pi*b*2**q/N for band b. The P(phi) = exp(i phi/2) RZ(phi)
    # corrections and coefficient phases share one address diagonal.
    circuit.append(QFTGate(num_system_qubits), list(system))
    angle_tables: list[list[float]] = []
    for qubit_index in range(num_system_qubits):
        angles = []
        for offset in specification.offsets:
            angle = 2.0 * np.pi * (2**qubit_index) * offset / dimension
            angles.append(float(angle))
        angle_tables.append(angles)
        append_uniformly_controlled_rz(
            circuit,
            system[qubit_index],
            control_qubits,
            angles,
        )
    diagonal_phases = [
        float(phases[branch])
        + 0.5 * sum(table[branch] for table in angle_tables)
        for branch in range(num_bands)
    ]
    if any(phase != 0.0 for phase in diagonal_phases):
        append_control_diagonal_phases(circuit, control_qubits, diagonal_phases)
    circuit.append(QFTGate(num_system_qubits).inverse(), list(system))

    if num_control_qubits:
        circuit.append(prepare.inverse(), control_qubits)

    ucrz_law = multiplexor_resource_law(
        "ucrz", control_qubits=num_control_qubits
    )
    diagonal_law = multiplexor_resource_law(
        "diagonal", control_qubits=num_control_qubits
    )
    diagonal_is_active = any(phase != 0.0 for phase in diagonal_phases)
    metadata = _base_metadata(
        resolved="banded",
        requested=requested_implementation,
        preprocessing_label="scalable_oracle",
        preprocessing_note="band specification input; no dense matrix materialized anywhere",
        register_order=("lcu_control", "system"),
    )
    metadata["band_specification"] = specification.to_dict()
    metadata["band_offsets"] = [int(offset) for offset in specification.offsets]
    metadata["toeplitz"] = True
    metadata["num_bands"] = num_bands
    metadata["structure_detection"] = "given_band_specification"
    # S**b has adjoint S**(-b). Pairing the stored coefficients proves
    # Hermiticity in O(num_bands) space and time at any register width.
    band_map = {offset % dimension: complex(value) for offset, value in
                zip(specification.offsets, specification.coefficients, strict=True)}
    hermitian = all(value == band_map.get((-offset) % dimension, 0).conjugate()
                    for offset, value in band_map.items())
    metadata["target_operator_is_hermitian"] = hermitian
    metadata["encoded_operator_is_hermitian"] = hermitian
    # Structural law inputs consumed unchanged by the per-query CX law and the
    # LCHS compiled-SELECT census.
    metadata["periodic_gate_counts"] = {
        "qft_pairs": 1,
        "ucrz_multiplexor_gates": num_system_qubits,
        "ucrz_table_entries_per_gate": ucrz_law["table_entries"],
        "ucrz_basis_cx_per_gate": ucrz_law["basis_cx_gates"],
        "control_diagonal_gates": int(diagonal_is_active and num_control_qubits > 0),
        "control_diagonal_table_entries": diagonal_law["table_entries"],
        "control_diagonal_basis_cx": (
            diagonal_law["basis_cx_gates"] if diagonal_is_active else 0
        ),
        "prep_register_qubits": num_control_qubits,
        "prep_pair_cx": 2 * direct_preparation_cx_bound(num_control_qubits, complex_phases=False),
    }
    return BlockEncoding(
        circuit=circuit,
        alpha=alpha,
        num_ancillas=num_control_qubits,
        system_qubits=num_system_qubits,
        error_bound=0.0,
        implementation="banded",
        metadata=metadata,
    )


__all__ = [
    "BandSpecification",
    "build_banded_block_encoding",
]
