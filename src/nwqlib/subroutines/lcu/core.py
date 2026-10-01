"""Linear-combination-of-unitaries circuit helpers.

The PREP-SELECT-PREP^dagger construction absorbs complex coefficient phases into
the unitaries that SELECT applies, so the PREP register only encodes
nonnegative coefficient magnitudes. For coefficients ``c_j`` and unitaries ``U_j``, the
all-zero control block implements

``(1 / alpha) * sum_j c_j U_j,  alpha = sum_j |c_j|``.

Childs and Wiebe, arXiv:1202.5822v1, introduce the two-term
PREP-SELECT-PREP^dagger circuit in Lemma 2 and Fig. 1, reach general
combinations through nested pairwise sums in Theorem 3, and note in Sec. II
that complex weights can be absorbed as phases of the unitaries.
PREP prepares ``sum_j sqrt(|c_j| / alpha) |j>`` and SELECT applies
``(c_j / |c_j|) U_j`` for address ``j``. This is the single-register
standard form of Low and Chuang, arXiv:1610.06546v3, Lemma 5 and Eq. (10),
and the case ``P_L = P_R`` of the subnormalization of a linear combination
of block encodings in Gilyen, Su, Low, and Wiebe, arXiv:1806.01838v1,
Lemma 52, with each phase-adjusted ``U_j`` a ``(1, 0, 0)`` block encoding
of itself (the same paper, Definition 44). Padded addresses beyond the
supplied terms have zero PREP amplitude and act as the identity.

Register convention:
    The control register is placed before the system register in the Qiskit
    circuit. Qiskit's little-endian statevector indexing then stores the
    control bits in the low-order bits, so the all-zero LCU block for system
    basis index ``s`` appears at statevector index ``s << num_control_qubits``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

import numpy as np
from nwqlib._limits import DEFAULT_MAX_BYTES
from nwqlib._validation import integer
from nwqlib.operators.access import _check_bytes
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate

from nwqlib.subroutines._dense_synthesis import controlled_synthesis_size
from nwqlib.subroutines.qiskit_compat import controlled
from nwqlib.subroutines.state_preparation import build_mps_circuit_state_preparation
from nwqlib.subroutines.state_preparation.direct import (
    _build_normalized_state_preparation,
)
from nwqlib.subroutines.lcu.registry import lcu_preparation_implementation_metadata
from nwqlib.subroutines.lcu.data import _lcu_prep_stage_requirements, lcu_coefficient_intake


LCUPreparationBackend = Literal["direct", "mps_circuit"]
# Local intake and construction size ceiling of ``_admit_lcu``, adjustable
# per call. Its law and rationale are registered in
# docs/ENGINEERING_CONSTANTS.md.
DEFAULT_MAX_LCU_WORK = 1_000_000_000


def _admit_lcu(data, *, max_bytes, max_work, dense=False):
    """Raise before the known LCU arrays or work exceed ``max_bytes`` or ``max_work``.

    With N supplied terms, P padded addresses (a qubits) and system
    dimension D, the coefficient and PREP stage is charged an untuned 128
    bytes per address and ``16 P (a + 1)`` work for the a-level preparation
    tree. Identity padding adds PREP entries but no matrices.

    The ``dense`` SELECT stage adds the following terms. The last two apply
    only when a control register exists (a >= 1), because a single term is
    appended as an uncontrolled ``UnitaryGate`` and is not synthesized here.

    - 64 bytes per entry of the N supplied D-square matrices: the complex128
      conversion and the phase-adjusted copy of each and the copy of the
      base matrix that each controlled branch keeps (48), with room for the
      unitarity check that Qiskit's ``UnitaryGate`` constructor runs on one
      matrix at a time. That check is one dense product ``U^dagger U`` per
      matrix, the ``N D**3`` work.
    - The exact synthesis of each controlled branch on ``m = log2(P D)``
      qubits (``build_lcu_select``). With M = P D,
      ``_dense_synthesis.controlled_synthesis_size`` derives
      ``11 M**3 + (m**2 + 5 m + 256) M**2`` work per branch for its
      factorizations, its two-qubit blocks and its elementwise passes, and
      ``256 M**2 + 65536`` bytes for the controlled matrix, the
      factorization arrays and the list of pending gates. The branches are
      synthesized one at a time, so these bytes are counted once.
    - ``N (176 M**2 + 16384)`` bytes for the circuits that the branches keep
      as their definitions, which allow 256 bytes for each of at most
      ``(11/16) M**2`` instructions and 16384 bytes for the circuit and gate
      objects.
    """
    n, p, d, a = data.term_count, data.padded_term_count, data.system_dimension, data.num_control_qubits
    # Branches that build_lcu_select synthesizes. A single term has none.
    synthesized = n if a else 0
    m = (p*d).bit_length() - 1
    branch_work, working_bytes, kept_bytes = controlled_synthesis_size(m) if synthesized else (0, 0, 0)
    prep_bytes, prep_work = _lcu_prep_stage_requirements(p, a)
    size = prep_bytes + (64*n*d*d + working_bytes + synthesized*kept_bytes if dense else 0)
    work = prep_work + (n*d**3 + synthesized*branch_work if dense else 0)
    _check_bytes(size, max_bytes, "LCU selected arrays")
    if work > integer(max_work, "max_work", 1):
        raise ValueError("LCU selected work exceeds max_work")


def _as_unitary_array(unitary: Any, *, dimension: int) -> np.ndarray:
    """Return a square matrix with the expected dimension."""

    matrix = np.asarray(unitary, dtype=complex)
    if matrix.shape != (dimension, dimension):
        raise ValueError("all unitaries must have the same square dimension")
    if not np.isfinite(matrix).all():
        raise ValueError("LCU unitaries must have finite entries")
    return matrix


@dataclass(frozen=True, kw_only=True)
class LCUData:
    """Checked LCU coefficients, PREP amplitudes and phase-adjusted unitaries.

    [`prepare_lcu_data`][nwqlib.subroutines.lcu.core.prepare_lcu_data] and
    [`prepare_lcu_gate_data`][nwqlib.subroutines.lcu.core.prepare_lcu_gate_data]
    return it. `alpha` is `coefficient_l1_norm`. The fields below are
    read-only.

    Attributes:
        original_coefficients: The N supplied complex coefficients ``c_j``,
            one per term, before their phases are absorbed.
        positive_coefficients: The magnitudes ``|c_j|`` that PREP encodes,
            zero-padded to P entries.
        phase_adjusted_unitaries: The N D-square complex128 matrices
            `(c_j / |c_j|) U_j` that SELECT applies, with `U_j` unchanged
            for `c_j = 0`. Only the N supplied matrices are stored, and the
            `P - N` padding addresses act as the identity. Empty for
            gate-level SELECT data (`prepare_lcu_gate_data`), whose branch
            gates carry their own coefficient phases.
        coefficient_l1_norm: ``alpha = sum_j |c_j|``, the subnormalization
            of the encoded block, in the units of the coefficients.
        prep_amplitudes: The P complex amplitudes ``sqrt(|c_j| / alpha)``
            that PREP prepares on the control register, zero on padded
            addresses.
        term_count: Number N of supplied terms before padding.
        padded_term_count: Number P of control-register basis states, the
            smallest power of two at least N.
        num_control_qubits: Number ``a = log2 P`` of control qubits.
        num_system_qubits: Number ``log2 D`` of system qubits.
        system_dimension: Dimension D of each unitary ``U_j``.
        register_order: Register names in circuit order, control register
            first.
    """

    original_coefficients: tuple[complex, ...]
    positive_coefficients: tuple[float, ...]
    phase_adjusted_unitaries: tuple[np.ndarray, ...]
    coefficient_l1_norm: float
    prep_amplitudes: tuple[complex, ...]
    term_count: int
    padded_term_count: int
    num_control_qubits: int
    num_system_qubits: int
    system_dimension: int
    register_order: tuple[str, str] = ("lcu_control", "system")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like LCU metadata summary."""

        return {
            "coefficient_l1_norm": self.coefficient_l1_norm,
            "term_count": self.term_count,
            "padded_term_count": self.padded_term_count,
            "num_control_qubits": self.num_control_qubits,
            "num_system_qubits": self.num_system_qubits,
            "system_dimension": self.system_dimension,
            "register_order": list(self.register_order),
        }


@dataclass(frozen=True, kw_only=True)
class LCUCircuit:
    """A PREP-SELECT-PREP^dagger circuit with its coefficient data and PREP error.

    [`build_lcu_circuit`][nwqlib.subroutines.lcu.core.build_lcu_circuit]
    returns it. The circuit is `circuit`, and `alpha` is
    `data.coefficient_l1_norm`. The fields below are read-only.

    Attributes:
        circuit: The built circuit PREP, SELECT, PREP^dagger on
            ``a + log2 D`` qubits, control register first. With direct PREP
            its all-zero control block is ``sum_j c_j U_j / alpha``. MPS PREP
            approximates that block.
        data: The ``LCUData`` the circuit was built from: coefficients,
            phase-adjusted unitaries, dimensions and register order.
        preparation_backend: PREP construction used in ``circuit``,
            ``"direct"`` or ``"mps_circuit"``.
        preparation_metadata: JSON-like record of that same PREP
            construction, such as its method name, error model and, for
            ``"mps_circuit"``, the TT-SVD summary.
        preparation_l2_error: 2-norm distance between the state PREP
            prepares and the target coefficient state, phases included, in
            the ideal-gate model, as reported by the PREP construction. 0
            for direct PREP, which makes no algorithmic approximation, and
            for a single term, which needs no PREP circuit. None for MPS
            PREP of two or more terms, whose circuit error this construction
            does not evaluate. It has the same phase convention as coherent
            PREP, inverse PREP and controlled use.
    """

    circuit: QuantumCircuit
    data: LCUData
    preparation_backend: str = "direct"
    preparation_metadata: Mapping[str, Any] = field(default_factory=dict)
    preparation_l2_error: float | None = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like LCU circuit summary."""

        return {
            "data": self.data.to_dict(),
            "preparation_backend": self.preparation_backend,
            "preparation_metadata": dict(self.preparation_metadata),
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "gate_counts": {name: int(count) for name, count in self.circuit.count_ops().items()},
        }


@dataclass(frozen=True, kw_only=True)
class _LCUPreparationArtifact:
    """One PREP build shared by the shipped circuit and its error metadata."""

    circuit: QuantumCircuit
    metadata: Mapping[str, Any]
    preparation_l2_error: float | None



def prepare_lcu_gate_data(
    coefficients: Any,
    *,
    system_dimension: int,
    coefficient_atol: float = 0.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_LCU_WORK,
) -> LCUData:
    """Check LCU coefficients and compute the PREP data for a SELECT built from your own gates.

    The coefficient handling is that of
    [`prepare_lcu_data`][nwqlib.subroutines.lcu.core.prepare_lcu_data],
    for callers whose SELECT branches are inserted as controlled gates
    rather than dense `UnitaryGate` matrices. `phase_adjusted_unitaries` is
    empty, and each branch gate must carry its own coefficient phase. The
    PREP register still encodes only nonnegative coefficient magnitudes.
    The returned data cannot be passed to `build_lcu_select`.

    Args:
        coefficients (array_like): One-dimensional complex coefficients.
        system_dimension (int): Dimension D of the system register that the
            branch gates act on, a power of two.
        coefficient_atol (float): Default `0.0`. Threshold on the total
            coefficient 1-norm, as for `prepare_lcu_data`.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the coefficient and PREP arrays.
        max_work (int): Default `1_000_000_000`. Limit on the PREP work.

    Returns:
        data (LCUData): Coefficients, PREP amplitudes and register sizes,
            with an empty `phase_adjusted_unitaries`.
    """

    coefficient_array, dimension, (
        coefficient_l1_norm,
        positive,
        prep_amplitudes,
        padded_term_count,
        num_control_qubits,
        num_system_qubits,
    ) = lcu_coefficient_intake(
        coefficients,
        system_dimension=system_dimension,
        coefficient_atol=coefficient_atol,
        max_bytes=max_bytes,
        max_work=max_work,
    )

    data = LCUData(
        original_coefficients=tuple(complex(value) for value in coefficient_array),
        positive_coefficients=tuple(float(value) for value in positive),
        phase_adjusted_unitaries=(),
        coefficient_l1_norm=coefficient_l1_norm,
        prep_amplitudes=tuple(complex(value) for value in prep_amplitudes),
        term_count=int(coefficient_array.size),
        padded_term_count=int(padded_term_count),
        num_control_qubits=num_control_qubits,
        num_system_qubits=num_system_qubits,
        system_dimension=int(dimension),
    )
    _admit_lcu(data, max_bytes=max_bytes, max_work=max_work)
    return data


def prepare_lcu_data(
    coefficients: Any,
    unitaries: Any,
    *,
    coefficient_atol: float = 0.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_LCU_WORK,
) -> LCUData:
    """Check LCU coefficients and unitaries, and move each coefficient's phase into its unitary.

    The result holds the PREP amplitudes `sqrt(|c_j| / alpha)` with
    `alpha = sum_j |c_j|` and the phase-adjusted unitaries
    `(c_j / |c_j|) U_j` that SELECT applies, ready for
    [`build_lcu_prepare`][nwqlib.subroutines.lcu.core.build_lcu_prepare]
    and [`build_lcu_select`][nwqlib.subroutines.lcu.core.build_lcu_select].
    Coefficient count, finite values and nonzero total weight are checked
    before unitary conversion.

    Args:
        coefficients (array_like): One-dimensional complex coefficients
            `c_j`.
        unitaries (Sequence[array_like]): One square unitary matrix `U_j`
            per coefficient, of power-of-two dimension D.
        coefficient_atol (float): Default `0.0`. Threshold on the total
            coefficient 1-norm. The default rejects only a zero norm, so a
            representable global rescaling is still accepted. This
            threshold does not prune individual terms. Every nonzero
            coefficient contributes its phase.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Byte limit for the coefficient and PREP arrays, the supplied
            matrices and the later dense SELECT, including the synthesized
            branch circuits it keeps.
        max_work (int): Default `1_000_000_000`. Work limit for these
            checks and the later dense SELECT, including the synthesis of
            each controlled branch.

    Returns:
        data (LCUData): Coefficients, PREP amplitudes, phase-adjusted
            unitaries and register sizes. `data.coefficient_l1_norm` is
            `alpha`.

    Raises:
        ValueError: If inputs have inconsistent shapes or all coefficients are
            zero, or if the PREP or later dense SELECT would exceed
            `max_bytes` or `max_work`.

    Examples:
        For `0.5 X - 0.5i Z`, `alpha = 1`, both PREP amplitudes are
        `sqrt(0.5) = 0.7071...`, and SELECT applies `X` and `-i Z`:

        >>> import numpy as np
        >>> from nwqlib.subroutines.lcu import prepare_lcu_data
        >>> X = np.array([[0, 1], [1, 0]])
        >>> Z = np.diag([1, -1])
        >>> data = prepare_lcu_data([0.5, -0.5j], [X, Z])
        >>> print(data.coefficient_l1_norm, data.num_control_qubits)
        1.0 1
        >>> print(np.round(np.real(data.prep_amplitudes), 4))
        [0.7071 0.7071]
        >>> print(np.allclose(data.phase_adjusted_unitaries[1], -1j * Z))
        True
    """

    # Coefficient validity precedes reading or converting any unitary.
    data = prepare_lcu_gate_data(coefficients, system_dimension=1,
        coefficient_atol=coefficient_atol, max_bytes=max_bytes, max_work=max_work)
    if not isinstance(unitaries, (tuple, list, np.ndarray)):
        raise ValueError("LCU unitaries must be a sized matrix table")
    if len(unitaries) != data.term_count:
        raise ValueError("LCU requires one unitary for each coefficient")
    shapes = []
    for unitary in unitaries:
        shape = getattr(unitary, "shape", None)
        if shape is None and isinstance(unitary, (tuple,list)):
            dimension = len(unitary)
            if not dimension or any(not isinstance(row,(tuple,list)) or len(row)!=dimension for row in unitary):
                raise ValueError("unitaries must be square matrices")
            shape = (dimension,dimension)
        if shape is None or len(shape)!=2 or shape[0]!=shape[1]:
            raise ValueError("unitaries must be square matrices with known shape")
        shapes.append(shape)
    dimension = shapes[0][0]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("all unitaries must have the same square dimension")
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("unitary dimension must be a positive power of two")
    data = replace(data, system_dimension=int(dimension), num_system_qubits=int(np.log2(dimension)))
    _admit_lcu(data, max_bytes=max_bytes, max_work=max_work, dense=True)
    matrices = tuple(_as_unitary_array(unitary, dimension=dimension) for unitary in unitaries)
    adjusted: list[np.ndarray] = []
    for coefficient, matrix in zip(data.original_coefficients, matrices, strict=True):
        if coefficient == 0.0:
            adjusted.append(matrix)
        elif abs(coefficient) < np.finfo(float).tiny:
            # A rounded subnormal magnitude is not an accurate phase divisor.
            adjusted.append(np.exp(1.0j * np.angle(coefficient)) * matrix)
        else:
            adjusted.append((coefficient / abs(coefficient)) * matrix)
    return replace(data, phase_adjusted_unitaries=tuple(adjusted))


def _build_lcu_preparation_artifact(
    data: LCUData,
    *,
    preparation_backend: LCUPreparationBackend = "direct",
    mps_max_bond_dim: int | None = None,
    mps_threshold: float = 1.0e-14,
    mps_num_layers: int = 2,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_LCU_WORK,
) -> _LCUPreparationArtifact:
    """Build one LCU PREP circuit together with its matching metadata.

    The MPS defaults ``mps_threshold=1e-14`` and ``mps_num_layers=2`` match
    those of ``build_mps_circuit_state_preparation`` and are registered in
    docs/ENGINEERING_CONSTANTS.md. The MPS path passes ``max_work`` to the builder
    as ``max_svd_work``, the limit of its TT-SVD and its layered construction.

    The direct path uses the same magnitude/phase tree as ``prepare_qiskit``
    but is not a declared PREP block, so ``max_direct_amplitudes`` does not
    apply. ``_admit_lcu`` admits it instead: ``16 P (a + 1)`` work against
    ``max_work`` and 128 bytes per address for P = 2**a addresses. A
    ``max_work`` of 10**8 admits up to 2**18 addresses, and the default of
    10**9, which QLS shares, up to 2**21.
    """

    _admit_lcu(data, max_bytes=max_bytes, max_work=max_work)
    if preparation_backend not in ("direct", "mps_circuit"):
        raise ValueError("preparation_backend must be 'direct' or 'mps_circuit'")
    if data.num_control_qubits == 0:
        return _LCUPreparationArtifact(
            circuit=QuantumCircuit(name="lcu_prepare"),
            metadata={
                "method": "none",
                "resolved_implementation": "none",
                "preparation_l2_error": 0.0,
                "implementation_metadata": {
                    "name": "none",
                    "provider": "nwqlib",
                    "method": "no PREP circuit needed for a single LCU term",
                    "package_names": [],
                    "package_versions": {},
                    "capability_notes": "Single-term LCU has no coefficient register.",
                },
            },
            preparation_l2_error=0.0,
        )
    implementation = lcu_preparation_implementation_metadata(preparation_backend)
    control = QuantumRegister(data.num_control_qubits, data.register_order[0])
    circuit = QuantumCircuit(control, name="lcu_prepare")
    prep_amplitudes = np.asarray(data.prep_amplitudes, dtype=complex)
    if preparation_backend == "direct":
        direct_preparation = _build_normalized_state_preparation(
            prep_amplitudes,
            input_norm=1.0,
            register_name=data.register_order[0],
        )
        circuit.compose(direct_preparation.circuit, qubits=list(control), inplace=True)
        return _LCUPreparationArtifact(
            circuit=circuit,
            metadata={
                "method": "qiskit_state_preparation",
                "resolved_implementation": preparation_backend,
                "preparation_l2_error": direct_preparation.preparation_l2_error,
                "preparation_error_model": "ideal_gate_construction",
                "implementation_metadata": implementation,
            },
            preparation_l2_error=direct_preparation.preparation_l2_error,
        )

    mps_preparation = build_mps_circuit_state_preparation(
        prep_amplitudes,
        max_bond_dim=mps_max_bond_dim,
        threshold=mps_threshold,
        num_layers=mps_num_layers,
        register_name=data.register_order[0],
        max_bytes=max_bytes, max_svd_work=max_work,
    )
    circuit.compose(mps_preparation.circuit, qubits=list(control), inplace=True)
    metadata = mps_preparation.to_dict()
    metadata["method"] = mps_preparation.method
    metadata["resolved_implementation"] = preparation_backend
    metadata["implementation_metadata"] = implementation
    return _LCUPreparationArtifact(
        circuit=circuit,
        metadata=metadata,
        preparation_l2_error=None,
    )


def build_lcu_prepare(
    data: LCUData,
    *,
    preparation_backend: LCUPreparationBackend = "direct",
    mps_max_bond_dim: int | None = None,
    mps_threshold: float = 1.0e-14,
    mps_num_layers: int = 2,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_LCU_WORK,
) -> QuantumCircuit:
    """Build the PREP circuit that loads the LCU coefficient state on the control register.

    Args:
        data (LCUData): Output of `prepare_lcu_data` or
            `prepare_lcu_gate_data`.
        preparation_backend (str): Default `"direct"`, the exact magnitude
            tree of
            [`build_qiskit_state_preparation`][nwqlib.subroutines.state_preparation.direct.build_qiskit_state_preparation].
            `"mps_circuit"` uses the MPS disentangling circuit of
            [`build_mps_circuit_state_preparation`][nwqlib.subroutines.state_preparation.mps_circuit.build_mps_circuit_state_preparation].
        mps_max_bond_dim (int | None): Default `None` (no cap). Rank cap for
            MPS PREP.
        mps_threshold (float): Default `1e-14`. Singular-value threshold for
            MPS PREP. Each singular value at or below it is dropped from the
            TT-SVD of the unit-norm coefficient state, keeping at least one
            per bond. The [constants registry](../../ENGINEERING_CONSTANTS.md)
            gives its reason.
        mps_num_layers (int): Default `2`. Number of MPS disentangling
            layers.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the known coefficient and PREP arrays. It does not
            bound SDK memory.
        max_work (int): Default `1_000_000_000`. Limit on the PREP work and
            the TT-SVD of MPS PREP.

    Returns:
        circuit (QuantumCircuit): Circuit targeting
            `sum_j sqrt(|c_j| / alpha) |j>` from `|0...0>`. Direct PREP
            realizes this ideal relation. Finite-layer MPS PREP approximates
            it with unevaluated circuit error. TT-SVD discarded weight alone
            does not bound that circuit error.
    """

    return _build_lcu_preparation_artifact(
        data,
        preparation_backend=preparation_backend,
        mps_max_bond_dim=mps_max_bond_dim,
        mps_threshold=mps_threshold,
        mps_num_layers=mps_num_layers,
        max_bytes=max_bytes, max_work=max_work,
    ).circuit


def build_lcu_select(data: LCUData, *, max_bytes: int = DEFAULT_MAX_BYTES,
                     max_work: int = DEFAULT_MAX_LCU_WORK) -> QuantumCircuit:
    """Build SELECT, which applies the dense unitary of branch `j` when the control register holds `j`.

    This entry serves small dense validation tables of unrelated
    multi-qubit unitaries. Pauli and product-formula callers use their
    gate-level multiplexor compilers and do not go through this dense
    construction.

    The whole controlled matrix of each branch, with its address as the
    control state, is synthesized exactly (see
    [Controlled dense unitaries](../../development/dense_synthesis.md#controlled-dense-unitaries)).
    SELECT therefore equals the ideal relation to binary64 rounding when
    every branch is unitary to rounding. A branch that passes Qiskit's
    constructor check (`numpy.allclose` of `U^dagger U` with the identity,
    atol 1e-8 and rtol 1e-5) with a larger defect is realized as its unitary
    polar factor, which differs from it by the order of that defect. For
    `U = expm(-i t G)` with random Hermitian G, t from 1e-10 to 10, one to
    three system qubits and one to three controls, the largest entry error
    was 4e-14, also after lowering to U and CX. A branch on
    `m = log2(P D)` qubits takes at most `(25/96) 4**m - 2**m + 4/3` CX
    for m >= 3, the largest count that Qiskit's own synthesis of the same
    controlled matrices reached in tests.

    Args:
        data (LCUData): Output of `prepare_lcu_data`, which holds the dense
            unitaries.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Byte limit for the supplied matrices, the arrays of one branch
            synthesis and the synthesized circuits the branches keep,
            checked before the first synthesis.
        max_work (int): Default `1_000_000_000`. Work limit for the
            unitarity checks and the synthesis of each branch, checked
            before the first synthesis.

    Returns:
        circuit (QuantumCircuit): Circuit applying branch `j` when the
            control register stores `j`.

    Raises:
        ValueError: If the dense unitary tuple is empty or does not cover the
            supplied term count. Extra padding branches act as identity.
    """

    unitary_count = len(data.phase_adjusted_unitaries)
    if unitary_count == 0:
        raise ValueError(
            "build_lcu_select requires dense phase_adjusted_unitaries; received "
            "the empty gate-level sentinel produced by prepare_lcu_gate_data. "
            "Dense callers must use prepare_lcu_data; gate-level callers "
            "assemble SELECT directly from their branch circuits."
        )
    if unitary_count != data.term_count:
        raise ValueError(
            "build_lcu_select requires phase_adjusted_unitaries length to equal "
            f"term_count; got {unitary_count} unitaries for "
            f"term_count={data.term_count}"
        )
    _admit_lcu(data, max_bytes=max_bytes, max_work=max_work, dense=True)

    system = QuantumRegister(data.num_system_qubits, data.register_order[1])
    if data.num_control_qubits == 0:
        circuit = QuantumCircuit(system, name="lcu_select")
        circuit.append(UnitaryGate(data.phase_adjusted_unitaries[0], label="U_0"), list(system))
        return circuit

    control = QuantumRegister(data.num_control_qubits, data.register_order[0])
    circuit = QuantumCircuit(control, system, name="lcu_select")
    control_qubits = list(control)
    system_qubits = list(system)
    for branch_index, unitary in enumerate(data.phase_adjusted_unitaries):
        controlled_gate = _controlled_branch_gate(
            unitary,
            num_control_qubits=data.num_control_qubits,
            branch_index=branch_index,
        )
        circuit.append(controlled_gate, control_qubits + system_qubits)
    return circuit


def _controlled_branch_gate(
    unitary: np.ndarray,
    *,
    num_control_qubits: int,
    branch_index: int,
) -> Any:
    """Synthesize the original branch once, preserving the details of a failure.

    ``ctrl_state=branch_index`` follows Qiskit's little-endian control-state
    convention, so branch ``j`` acts when the control register, least
    significant qubit first, holds ``j``.
    """

    try:
        gate = UnitaryGate(unitary, label=f"U_{branch_index}")
        return controlled(gate, num_control_qubits, ctrl_state=branch_index)
    except Exception as error:
        error.add_note(
            f"LCU branch {branch_index}, matrix shape {getattr(unitary, 'shape', None)}, "
            f"controls={num_control_qubits}; NWQLib does not retry the synthesis."
        )
        raise


def _assemble_prepared_lcu(
    data: LCUData,
    preparation: _LCUPreparationArtifact,
    select: QuantumCircuit,
    *,
    circuit: QuantumCircuit,
    preparation_qubits: list[Any],
    select_qubits: list[Any],
    preparation_backend: str,
) -> LCUCircuit:
    """Compose one private PREP/SELECT/PREP-dagger LCU shell."""

    circuit.compose(preparation.circuit, qubits=preparation_qubits, inplace=True)
    circuit.compose(select, qubits=select_qubits, inplace=True)
    circuit.compose(
        preparation.circuit.inverse(),
        qubits=preparation_qubits,
        inplace=True,
    )
    return LCUCircuit(
        circuit=circuit,
        data=data,
        preparation_backend=preparation_backend,
        preparation_metadata=dict(preparation.metadata),
        preparation_l2_error=preparation.preparation_l2_error,
    )


def build_lcu_circuit(
    coefficients: Any,
    unitaries: Any,
    *,
    preparation_backend: LCUPreparationBackend = "direct",
    mps_max_bond_dim: int | None = None,
    mps_threshold: float = 1.0e-14,
    mps_num_layers: int = 2,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_LCU_WORK,
) -> LCUCircuit:
    """Build the PREP-SELECT-PREP^dagger circuit whose all-zero control block is `sum_j c_j U_j / alpha`.

    The function runs `prepare_lcu_data`, `build_lcu_prepare` and
    `build_lcu_select` and composes PREP, SELECT and the inverse of PREP,
    with `alpha = sum_j |c_j|`.

    Args:
        coefficients (array_like): Complex LCU coefficients `c_j`.
        unitaries (Sequence[array_like]): One D-square unitary matrix `U_j`
            per coefficient, with D a power of two.
        preparation_backend (str): Default `"direct"`, exact PREP.
            `"mps_circuit"` uses MPS disentangling, as for
            `build_lcu_prepare`.
        mps_max_bond_dim (int | None): Default `None` (no cap). Rank cap for
            MPS PREP.
        mps_threshold (float): Default `1e-14`. Singular-value threshold for
            MPS PREP. Each singular value at or below it is dropped from the
            TT-SVD of the unit-norm coefficient state, keeping at least one
            per bond.
        mps_num_layers (int): Default `2`. Number of MPS disentangling
            layers.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Byte limit for the supplied matrices, PREP and the dense SELECT
            with its kept synthesized branch circuits.
        max_work (int): Default `1_000_000_000`. Work limit for the input
            checks, the SELECT synthesis and the TT-SVD of MPS PREP.

    Returns:
        lcu (LCUCircuit): The circuit in `lcu.circuit`, targeting the
            normalized all-zero block `sum_j c_j U_j / sum_j |c_j|`, and
            `alpha` in `lcu.data.coefficient_l1_norm`. Direct PREP realizes
            the coefficient amplitudes exactly, and SELECT realizes each
            controlled branch to binary64 rounding when the branch is
            unitary to rounding. MPS PREP approximates the coefficient
            amplitudes, so the realized block can differ. Its error remains
            unevaluated in the preparation metadata, and
            `preparation_l2_error` is `None`.

    Examples:
        The control register is the low-order bits, so the block is read at
        stride 2. `alpha` times the block equals `0.5 X - 0.5i Z`:

        >>> import numpy as np
        >>> from qiskit.quantum_info import Operator
        >>> from nwqlib.subroutines.lcu import build_lcu_circuit
        >>> X = np.array([[0, 1], [1, 0]])
        >>> Z = np.diag([1, -1])
        >>> lcu = build_lcu_circuit([0.5, -0.5j], [X, Z])
        >>> block = Operator(lcu.circuit).data[::2, ::2]
        >>> alpha = lcu.data.coefficient_l1_norm
        >>> print(np.allclose(alpha * block, 0.5 * X - 0.5j * Z))
        True
    """

    data = prepare_lcu_data(coefficients, unitaries, max_bytes=max_bytes, max_work=max_work)
    preparation = _build_lcu_preparation_artifact(
        data,
        preparation_backend=preparation_backend,
        mps_max_bond_dim=mps_max_bond_dim,
        mps_threshold=mps_threshold,
        mps_num_layers=mps_num_layers,
        max_bytes=max_bytes, max_work=max_work,
    )
    system = QuantumRegister(data.num_system_qubits, data.register_order[1])
    if data.num_control_qubits == 0:
        circuit = QuantumCircuit(system, name="lcu")
        circuit.compose(build_lcu_select(data, max_bytes=max_bytes, max_work=max_work), qubits=list(system), inplace=True)
        return LCUCircuit(
            circuit=circuit,
            data=data,
            preparation_backend=preparation_backend,
            preparation_metadata=dict(preparation.metadata),
            preparation_l2_error=preparation.preparation_l2_error,
        )
    control = QuantumRegister(data.num_control_qubits, data.register_order[0])
    circuit = QuantumCircuit(control, system, name="lcu")
    select = build_lcu_select(data, max_bytes=max_bytes, max_work=max_work)
    control_qubits = list(control)
    all_qubits = control_qubits + list(system)
    return _assemble_prepared_lcu(
        data,
        preparation,
        select,
        circuit=circuit,
        preparation_qubits=control_qubits,
        select_qubits=all_qubits,
        preparation_backend=preparation_backend,
    )


__all__ = [
    "_controlled_branch_gate",
    "LCUCircuit",
    "LCUData",
    "LCUPreparationBackend",
    "build_lcu_circuit",
    "build_lcu_prepare",
    "build_lcu_select",
    "prepare_lcu_data",
]
