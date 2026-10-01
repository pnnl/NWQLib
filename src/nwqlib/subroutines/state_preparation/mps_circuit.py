"""Circuit-level MPS state preparation via layered disentangling.

This backend constructs a Qiskit circuit without calling Qiskit's dense
``StatePreparation`` synthesis. The NumPy TT-SVD and its compression
analysis are
[`decompose_state_to_mps`][nwqlib.subroutines.state_preparation.mps.decompose_state_to_mps]
and
[`analyze_mps_state_compression`][nwqlib.subroutines.state_preparation.mps.analyze_mps_state_compression].

The layered disentangling scheme implemented here (truncate to bond
dimension 2, extract a layer of local unitaries, repeat on the residual
state) follows Ran, arXiv:1908.07958v2, Sec. III, steps 1-4, and that
paper's Eqs. (6)-(9) complete each truncated tensor to a local unitary.
"""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.synthesis import OneQubitEulerDecomposer

from nwqlib._numerics import stable_vector_norm
from nwqlib.subroutines._dense_synthesis import dense_unitary_circuit
from nwqlib.subroutines._registry_utils import package_versions
from nwqlib.subroutines.state_preparation import mps as mps_module
from nwqlib.subroutines.state_preparation.mps import (
    MPSDecomposition,
)

def _require_scikit_tt() -> tuple[Any, Any]:
    """Import optional ``scikit_tt`` objects or raise a user-facing error."""

    try:
        import scikit_tt.tensor_train as tt
        from scikit_tt import TT
    except ImportError as exc:
        raise ImportError(
            "lcu_state_preparation='mps_circuit' requires the optional tensor "
            "dependency scikit_tt, which is not on PyPI. Install it with "
            "`python -m pip install 'scikit_tt @ git+https://github.com/PGelss/scikit_tt.git'`."
        ) from exc
    return tt, TT


def _complete_unitary_from_fixed_columns(
    matrix: np.ndarray,
    fixed_cols: tuple[int, ...],
    *,
    atol: float = 1.0e-9,
) -> np.ndarray:
    """Complete a partially filled unitary using an SVD null-space basis.

    The fixed columns hold normalized MPS tensor entries, and the remaining
    columns become an orthonormal basis of their orthogonal complement, as
    in Ran (arXiv:1908.07958v2), Eqs. (7)-(9). ``atol`` is the canonical-form
    tolerance for the norms and overlaps of the fixed columns. Its default
    ``1e-9`` is an untuned column-admission window registered in
    docs/ENGINEERING_CONSTANTS.md.
    """

    completed = np.asarray(matrix, dtype=complex).copy()
    fixed_vectors = []
    for column in fixed_cols:
        vector = completed[:, column].copy()
        norm = np.linalg.norm(vector)
        if abs(norm - 1.0) > atol:
            raise ValueError("MPS is not in canonical form; fixed column is not normalized")
        fixed_vectors.append(vector)

    for index, first in enumerate(fixed_vectors):
        for second in fixed_vectors[index + 1 :]:
            if abs(np.vdot(first, second)) > atol:
                raise ValueError("MPS is not in canonical form; fixed columns are not orthogonal")

    fixed_matrix = np.column_stack(fixed_vectors)
    _, singular_values, vh_matrix = np.linalg.svd(fixed_matrix.conj().T)
    rank = int(np.sum(singular_values > atol))
    complement = vh_matrix[rank:].conj().T
    remaining_cols = [column for column in range(completed.shape[1]) if column not in fixed_cols]
    if complement.shape[1] < len(remaining_cols):
        raise ValueError("Unable to complete MPS local tensor to a unitary")

    for basis_index, column in enumerate(remaining_cols):
        completed[:, column] = complement[:, basis_index]
    return completed


def _product_state_circuit_from_cores(cores: tuple[np.ndarray, ...]) -> QuantumCircuit:
    """Build single-qubit gates for a rank-1 product-state MPS.

    Each core has shape ``(1, 2, 1, 1)``, giving a two-component amplitude
    vector. The unitary whose first column matches the amplitude is decomposed
    into a Qiskit ``u`` gate. Core ``i`` maps to qubit ``n - 1 - i`` to preserve
    the MSB-first flattened basis order.
    """

    num_qubits = len(cores)
    circuit = QuantumCircuit(num_qubits, name="mps_circuit_state_preparation")
    decomposer = OneQubitEulerDecomposer("U")
    for site, core in enumerate(cores):
        vector = core.flatten()
        norm = np.linalg.norm(vector)
        if norm <= 0.0:
            raise ValueError("MPS product-state core has zero norm")
        vector = vector / norm
        unitary = np.zeros((2, 2), dtype=complex)
        unitary[:, 0] = vector
        unitary[:, 1] = [-np.conj(vector[1]), np.conj(vector[0])]
        circuit.compose(
            decomposer(unitary, simplify=False), qubits=[num_qubits - 1 - site], inplace=True
        )
    return circuit


def _scikit_tt_mps_from_cores(
    cores: tuple[np.ndarray, ...],
    *,
    tt_class: Any,
) -> Any:
    """Build and right-canonicalize a ``scikit_tt.TT`` from existing cores."""

    mps = tt_class([np.asarray(core, dtype=complex).copy() for core in cores])
    mps.ortho()
    return mps


def mps_to_circuit(mps: Any, *, num_layers: int = 1) -> QuantumCircuit:
    """Convert a right-canonical MPS to a Qiskit circuit by disentangling.

    At each layer, the MPS is truncated to bond dimension 2. Local one- and
    two-qubit unitaries are extracted, and their inverses are applied back to
    the MPS so the next layer works on a less-entangled residual state.

    This is Ran (arXiv:1908.07958v2), Sec. III, steps 1-4. Site ``i`` of the
    MSB-first cores acts on qubit ``n - 1 - i``. A site with bonds ``(1, 2)``
    becomes a two-qubit unitary whose first column is the core (Eq. (9)), a
    site with bonds ``(2, 2)`` fixes the two columns given by its left-bond
    slices (Eq. (7)), a site with bonds ``(2, 1)`` is the one-qubit gate
    formed from its core (Eq. (6)), and a product site is one one-qubit gate.
    Each new layer is composed at the front, so the finished circuit applies
    the last extracted layer first and the first extracted layer last.

    Each two-qubit unitary is synthesized exactly to rounding with at most
    three CX, so the synthesis adds only
    rounding to the error of the layered construction. Qiskit's
    ``TwoQubitBasisDecomposer`` replaces a unitary by a class with fewer CX
    whenever the average gate fidelity between them is at least
    ``1 - 1e-9``. For states within about 1e-5 of a product state it saved
    one or two CX per unitary and erred by about the distance from the
    product state (1e-9 to 1e-5 in tests), and the exact synthesis keeps
    those CX.

    Args:
        mps (scikit_tt.TT): Right-canonical tensor train with cores of shape
            `(a, 2, 1, b)`.
        num_layers (int): Default `1`. Number of disentangling layers.

    Returns:
        QuantumCircuit that approximates the MPS state from ``|0...0>``.
        A finite layer count can leave a residual even when the original
        TT-SVD discarded weight is zero; no circuit error is measured here.
    """

    if num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    tt, TT = _require_scikit_tt()

    num_sites = len(mps.cores)
    circuit = QuantumCircuit(num_sites, name="mps_circuit_state_preparation")
    one_qubit_decomposer = OneQubitEulerDecomposer("U")
    working_mps = mps.copy()

    for layer_index in range(num_layers):
        if max(working_mps.ranks) == 1:
            product_mps = (1.0 / working_mps.norm()) * working_mps
            product_circuit = _product_state_circuit_from_cores(tuple(product_mps.cores))
            circuit.compose(product_circuit, front=True, inplace=True)
            break

        layer_circuit = QuantumCircuit(num_sites)
        truncated_mps = working_mps.copy()
        truncated_mps.ortho(max_rank=2)
        cores = ((1.0 / truncated_mps.norm()) * truncated_mps).cores
        extracted_unitaries = []

        for site, core in enumerate(cores):
            left_bond, _, _, right_bond = core.shape

            if left_bond <= 1 and right_bond <= 1:
                vector = core.flatten()
                unitary = np.zeros((2, 2), dtype=complex)
                unitary[:, 0] = vector
                unitary[:, 1] = [-np.conj(vector[1]), np.conj(vector[0])]
                layer_circuit.compose(
                    one_qubit_decomposer(unitary, simplify=False),
                    qubits=[num_sites - 1 - site], inplace=True,
                )
                extracted_unitaries.append(("1q", unitary.conj().T, site))

            elif left_bond <= 1 and right_bond == 2:
                unitary = np.zeros((4, 4), dtype=complex)
                unitary[:, 0] = core.flatten()
                unitary = _complete_unitary_from_fixed_columns(unitary, (0,))
                layer_circuit.compose(
                    dense_unitary_circuit(unitary),
                    qubits=[num_sites - 2 - site, num_sites - 1 - site],
                    inplace=True,
                )
                extracted_unitaries.append(("2q", unitary.conj().T, site))

            elif left_bond == 2 and right_bond <= 1:
                unitary = core.reshape((2, 2)).T
                layer_circuit.compose(
                    one_qubit_decomposer(unitary, simplify=False),
                    qubits=[num_sites - 1 - site], inplace=True,
                )
                extracted_unitaries.append(("1q", unitary.conj().T, site))

            else:
                unitary = np.zeros((4, 4), dtype=complex)
                unitary[:, 0] = core[0, :, :, :].flatten()
                unitary[:, 2] = core[1, :, :, :].flatten()
                unitary = _complete_unitary_from_fixed_columns(unitary, (0, 2))
                layer_circuit.compose(
                    dense_unitary_circuit(unitary),
                    qubits=[num_sites - 2 - site, num_sites - 1 - site],
                    inplace=True,
                )
                extracted_unitaries.append(("2q", unitary.conj().T, site))

        circuit.compose(layer_circuit, front=True, inplace=True)

        # The residual is consumed only by the next layer.
        if layer_index + 1 == num_layers:
            break

        for gate_type, inverse_unitary, core_position in reversed(extracted_unitaries):
            if gate_type == "1q":
                operator_tt = TT(inverse_unitary.reshape((2, 2)))
                left_sites = core_position
                right_sites = num_sites - core_position - 1
            else:
                operator_tt = TT(inverse_unitary.reshape((2, 2, 2, 2)))
                left_sites = core_position
                right_sites = num_sites - core_position - 2

            mpo = operator_tt
            if left_sites > 0:
                mpo = tt.eye([2] * left_sites).concatenate(mpo)
            if right_sites > 0:
                mpo = mpo.concatenate(tt.eye([2] * right_sites))

            working_mps = mpo.dot(working_mps)
            # Right-orthonormalize the residual, with a 1e-12 threshold on the
            # reduced SVDs, before the next layer's rank-2 truncation.
            working_mps.ortho(threshold=1.0e-12)

    return circuit


@dataclass(frozen=True, kw_only=True)
class MPSCircuitStatePreparation:
    """A layered MPS state-preparation circuit with its compression data and optional evaluated fidelity.

    [`build_mps_circuit_state_preparation`][nwqlib.subroutines.state_preparation.mps_circuit.build_mps_circuit_state_preparation]
    returns it with the fidelity fields unset, and
    [`validate_mps_circuit_state_preparation`][nwqlib.subroutines.state_preparation.mps_circuit.validate_mps_circuit_state_preparation]
    returns a copy with them evaluated. The circuit is `circuit`. It stores
    the decomposition and the normalized target used by the validator, and
    no compression-analysis object or other reconstructed vector. The
    fields below are read-only.

    Attributes:
        circuit: The layered MPS disentangling state-preparation circuit.
        decomposition: Compressed target tensor used by circuit construction.
        target_state: Normalized target input in the original flattened basis order.
        prepared_state: Explicitly evaluated circuit output, or None when no circuit simulation was requested.
        input_norm: Original input magnitude removed before normalized preparation.
        fidelity_to_target: Evaluated circuit fidelity to target, or None if not evaluated.
        preparation_l2_error: Evaluated phase-sensitive circuit-to-target distance ``||prepared - target||``, or None if unavailable.
        num_layers: Number of disentangling circuit layers.
        max_bond_dim: Optional requested compression rank cap.
        threshold: Individual singular-value truncation threshold used by compression.
        circuit_fidelity_status: `"evaluated"` or `"not_evaluated"` for the circuit fidelity. Compression estimates do not fill it.
        method: Name of the construction, `"mps_disentangling_circuit"`.
    """

    circuit: QuantumCircuit
    decomposition: MPSDecomposition
    target_state: np.ndarray
    prepared_state: np.ndarray | None
    input_norm: float
    fidelity_to_target: float | None
    preparation_l2_error: float | None
    num_layers: int
    max_bond_dim: int | None
    threshold: float
    circuit_fidelity_status: str = "not_evaluated"
    method: str = "mps_disentangling_circuit"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like circuit state-preparation summary."""

        return {
            "method": self.method,
            "input_norm": self.input_norm,
            "num_qubits": self.decomposition.num_qubits,
            "dimension": self.decomposition.original_dimension,
            "bond_dimensions": list(self.decomposition.bond_dimensions),
            "max_bond_dim": self.max_bond_dim,
            "threshold": self.threshold,
            "num_layers": self.num_layers,
            "circuit_fidelity_status": self.circuit_fidelity_status,
            "fidelity_to_target": self.fidelity_to_target,
            "preparation_l2_error": self.preparation_l2_error,
            "compression_fidelity_to_target": None,
            "circuit_depth": self.circuit.depth(),
            "gate_counts": {name: int(count) for name, count in self.circuit.count_ops().items()},
            "optional_package_versions": package_versions(("scikit_tt",)),
            "decomposition": self.decomposition.to_dict(),
        }


def build_mps_circuit_state_preparation(
    vector: Any,
    *,
    max_bond_dim: int | None = None,
    threshold: float = 1.0e-14,
    num_layers: int = 2,
    register_name: str = "system",
    _selected_decomposition: MPSDecomposition | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_svd_work: int = mps_module.DEFAULT_MAX_SVD_WORK,
) -> MPSCircuitStatePreparation:
    """Build a layered MPS circuit that approximately prepares a state vector from `|0...0>`.

    The vector is compressed by one TT-SVD sweep, as in
    `decompose_state_to_mps`, and `mps_to_circuit` builds the layered
    disentangling circuit (Ran, arXiv:1908.07958v2, Sec. III). The circuit
    is not simulated, so its fidelity is not evaluated.

    Args:
        vector (array_like): State amplitudes to prepare.
        max_bond_dim (int | None): Default `None` (no cap). TT-SVD maximum
            bond dimension.
        threshold (float): Default `1e-14`. Singular-value pruning threshold
            for the core decomposition.
        num_layers (int): Default `2`. Number of MPS disentangling layers.
        register_name (str): Default `"system"`. Name for the prepared
            quantum register.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the known input and TT-SVD arrays. It does not measure
            process memory. The layered construction
            ([layered construction law](../../ENGINEERING_CONSTANTS.md#layered-mps-construction-law))
            and the exact syntheses of the two-qubit unitaries are checked
            against it separately, before either starts.
        max_svd_work (int): Default `100_000_000`. Limit on the whole-sweep
            dense-SVD work. The layered construction's scikit_tt sweeps and
            products and the at most `num_layers * (n - 1)` exact syntheses
            of two-qubit unitaries are checked against it separately, before
            either starts.

    Returns:
        preparation (MPSCircuitStatePreparation): The circuit with its
            compression data, and unevaluated circuit diagnostics. The
            TT-SVD discarded weight describes core compression, not the
            finite-layer circuit error.

    Examples:
        The 3-qubit GHZ state has bond dimension 2, and the validated
        circuit prepares it with fidelity 1. The layered construction needs
        `scikit_tt`, which is installed separately from NWQLib's extras:

        >>> import numpy as np
        >>> from nwqlib.subroutines.state_preparation import (
        ...     build_mps_circuit_state_preparation,
        ...     validate_mps_circuit_state_preparation)
        >>> ghz = np.zeros(8)
        >>> ghz[[0, 7]] = 1
        >>> preparation = build_mps_circuit_state_preparation(ghz)
        >>> print(preparation.decomposition.bond_dimensions)
        (1, 2, 2, 1)
        >>> checked = validate_mps_circuit_state_preparation(preparation)
        >>> print(round(checked.fidelity_to_target, 10))
        1.0
    """

    if num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    if _selected_decomposition is None:
        target_state, input_norm = mps_module._input(vector, max_bytes)
        decomposition = mps_module._decompose_normalized_state(
            target_state, max_bond_dim=max_bond_dim, threshold=threshold,
            max_bytes=max_bytes, max_svd_work=max_svd_work, input_norm=input_norm)
    else:
        # The selected factory owns this already-normalized immutable target and
        # the exact cores; no TT-SVD or full-state reconstruction is repeated.
        target_state, input_norm, decomposition = vector, 1.0, _selected_decomposition
        from nwqlib.operators.access import _check_bytes
        from nwqlib.operators.inputs import _digest
        if not isinstance(target_state, np.ndarray) or target_state.ndim != 1:
            raise ValueError("selected MPS requires its actual normalized vector")
        _check_bytes(target_state.nbytes + decomposition.core_bytes, max_bytes, "selected MPS input")
        if (decomposition.original_dimension != len(target_state) or decomposition.max_bond_dim != max_bond_dim
                or decomposition.threshold != threshold
                or decomposition.input_id != _digest("mps.normalized.C-order", (len(target_state),), (target_state,))):
            raise ValueError("MPS construction differs from the selected core parameters")
    num_qubits = decomposition.num_qubits
    if num_qubits == 0:
        circuit = QuantumCircuit(name="mps_circuit_state_preparation")
        circuit.global_phase = float(np.angle(target_state[0]))
    elif max(decomposition.bond_dimensions) == 1:
        base_circuit = _product_state_circuit_from_cores(decomposition.cores)
        register = QuantumRegister(num_qubits, register_name)
        circuit = QuantumCircuit(register, name="mps_circuit_state_preparation")
        circuit.compose(base_circuit, qubits=list(register), inplace=True)
    else:
        # Each layer of mps_to_circuit synthesizes at most n - 1 two-qubit
        # unitaries with dense_unitary_circuit, one per site with a right
        # bond, and keeps their circuits.
        from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses, dense_synthesis_size
        syntheses = num_layers * (num_qubits - 1)
        if syntheses * dense_synthesis_size(2)[0] > max_svd_work:
            raise ValueError(f"MPS circuit needs {syntheses} two-qubit syntheses, exceeding max_svd_work")
        admit_dense_syntheses((2,) * syntheses, max_work=max_svd_work, max_bytes=max_bytes,
                              operation="MPS two-qubit synthesis")
        mps_module.admit_layered_construction(decomposition.bond_dimensions, num_layers,
                                              max_svd_work=max_svd_work, max_bytes=max_bytes)
        _, TT = _require_scikit_tt()
        scikit_mps = _scikit_tt_mps_from_cores(
            decomposition.cores,
            tt_class=TT,
        )
        base_circuit = mps_to_circuit(scikit_mps, num_layers=num_layers)
        register = QuantumRegister(num_qubits, register_name)
        circuit = QuantumCircuit(register, name="mps_circuit_state_preparation")
        circuit.compose(base_circuit, qubits=list(register), inplace=True)
    return MPSCircuitStatePreparation(
        circuit=circuit,
        decomposition=decomposition,
        target_state=target_state,
        prepared_state=None,
        input_norm=input_norm,
        fidelity_to_target=None,
        preparation_l2_error=None,
        num_layers=num_layers,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
    )


def validate_mps_circuit_state_preparation(
    preparation: MPSCircuitStatePreparation,
) -> MPSCircuitStatePreparation:
    """Simulate a built MPS circuit once and return a copy with its fidelity to the target.

    This helper performs one statevector simulation, whose cost grows
    exponentially with the qubit count, for small-instance validation. It
    compares the circuit output with the stored normalized target and
    returns a copy of `preparation` with `prepared_state`,
    `fidelity_to_target`, the phase-sensitive `preparation_l2_error` and
    `circuit_fidelity_status="evaluated"`.

    Args:
        preparation (MPSCircuitStatePreparation): Output of
            `build_mps_circuit_state_preparation`.

    Returns:
        checked (MPSCircuitStatePreparation): The copy with the evaluated
            fields.
    """

    prepared_state = np.asarray(
        Statevector.from_instruction(preparation.circuit).data,
        dtype=complex,
    )
    fidelity = float(abs(np.vdot(preparation.target_state, prepared_state)) ** 2)
    preparation_l2_error = stable_vector_norm(
        prepared_state - preparation.target_state
    )
    return replace(
        preparation,
        prepared_state=prepared_state,
        fidelity_to_target=fidelity,
        preparation_l2_error=preparation_l2_error,
        circuit_fidelity_status="evaluated",
    )


__all__ = [
    "MPSCircuitStatePreparation",
    "build_mps_circuit_state_preparation",
    "mps_to_circuit",
    "validate_mps_circuit_state_preparation",
]
