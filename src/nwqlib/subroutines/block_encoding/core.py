"""Block-encoding record, selection slot, and dense constructions.

Contract (Gilyen, Su, Low, and Wiebe, arXiv:1806.01838v1, Definition 43):
a unitary ``U`` on ``a + n`` qubits is an
``(alpha, a, eps)`` block encoding of an ``n``-qubit operator ``A`` when
``|| A - alpha * (<0^a| (x) I) U (|0^a> (x) I) || <= eps``. The all-zero
ancilla block of ``U`` is then ``A / alpha``.

Implementations, each with ``A / alpha`` in the all-zero ancilla block:

* ``multiplexed_pauli`` encodes ``A = sum_j c_j P_j`` from a Pauli
  decomposition with ``alpha = sum_j |c_j|`` and ``ceil(log2 L)`` address
  ancillas for ``L`` terms. PREP prepares ``sum_j sqrt(|c_j| / alpha) |j>``
  and SELECT applies ``exp(i arg c_j) P_j``. This is the standard-form
  encoding of Low and Chuang, arXiv:1610.06546v3, Lemma 5 and Eq. (10),
  and its alpha is that of a linear combination of block encodings, Gilyen
  et al., arXiv:1806.01838v1, Lemma 52. It is exact for the supplied terms
  and accepts any square power-of-two matrix, and its cost grows with the
  number of terms.
* ``banded`` encodes a circulant ``A = sum_b beta_b S^b`` as an exact LCU of
  cyclic shifts with ``alpha = sum_b |beta_b|`` and ``ceil(log2 B)``
  ancillas for ``B`` bands (``banded.py``). It never forms a dense matrix.
* ``dense_dilation`` uses one ancilla and the smallest zero-error
  ``alpha = ||A||_2``. The completed unitary on ``n + 1`` qubits is
  synthesized as one dense gate, whose cost grows as ``4**(n + 1)``, so it
  serves small validation instances. Routing prices it with the CX count of
  Shende, Bullock, and Markov, quant-ph/0406176v5, Table 1 (p. 14) and
  Eq. (19) (p. 16) (``_dense_dilation_predicted_cx``).

Register convention:
    Ancilla registers are placed before the system register, following the
    LCU convention in ``subroutines/lcu/core.py``. Qiskit's little-endian
    statevector indexing then stores the ancilla bits in the low-order bits,
    so the encoded block entry ``(A / alpha)[s', s]`` sits at unitary-matrix
    index ``(s' << num_ancillas, s << num_ancillas)``, which
    ``block_encoding_top_left`` extracts.
"""

from __future__ import annotations


from dataclasses import dataclass, replace
from math import fsum
from fractions import Fraction
from typing import Any, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

import numpy as np
from nwqlib._validation import finite_real, integer
from nwqlib._limits import DEFAULT_MAX_BYTES
from nwqlib._linalg_laws import singular_values_work
from nwqlib._numerics import normalized_matrix
from nwqlib.operators.access import _check_bytes
from nwqlib.evidence._work import BOOKKEEPING_BYTES, integer_object_bytes

from nwqlib.backends.resources import dense_unitary_cx_qsd_upper_bound
from nwqlib.subroutines._multiplexors import (
    append_control_diagonal_phases,
    append_projected_unitary_table,
    local_pauli_dependencies,
    multiplexor_resource_law,
    project_local_pauli_table,
    projected_unitary_resource_law,
)
from nwqlib._preparation_laws import direct_preparation_cx_bound
from nwqlib.subroutines._registry_utils import implementation_metadata
from nwqlib.subroutines.block_encoding.registry import BLOCK_ENCODING_IMPLEMENTATIONS
from nwqlib.subroutines.pauli_decomposition import (
    PauliDecomposition,
    PauliTerm,
)
from nwqlib.operators._pauli import _pauli_masks, pauli_coefficients
from nwqlib.subroutines._serialization import array_to_json


@dataclass(frozen=True, kw_only=True)
class BlockEncoding:
    """A block-encoding circuit with its `alpha`, ancilla count and error bound.

    [`build_block_encoding`][nwqlib.subroutines.block_encoding.build_block_encoding],
    [`build_block_encoding_from_plan`][nwqlib.subroutines.block_encoding.build_block_encoding_from_plan]
    and [`build_banded_block_encoding`][nwqlib.subroutines.block_encoding.build_banded_block_encoding]
    return it, and so do `build_qsp_evolution_encoding` and
    `build_control_diagonal_generator_encoding` of `nwqlib.subroutines.qsp`.
    Its `circuit` field holds the result, and the all-zero ancilla block of
    that circuit is `A / alpha`, which
    [`block_encoding_top_left`][nwqlib.subroutines.block_encoding.block_encoding_top_left]
    reads from the circuit's unitary. Construction requires a finite
    positive `alpha`, a finite nonnegative `error_bound` when one is given,
    and nonnegative integer widths. These checks do not inspect the circuit
    or prove its operator relation. The fields below are read-only.

    Attributes:
        circuit: Unitary circuit whose all-zero ancilla block encodes
            `A / alpha`.
        alpha: Subnormalization `alpha` in the units of `A`.
        num_ancillas: Number of ancilla qubits (placed before the system
            register, low-order bits).
        system_qubits: Number of system qubits.
        error_bound: Operator-norm bound on
            `|| A - alpha * encoded block ||` in the units of `A`, or None
            when unavailable.
        implementation: Construction used: `"multiplexed_pauli"`,
            `"banded"` or `"dense_dilation"` from the builders of this
            module, `"qsp_jacobi_anger_evolution"` from
            `build_qsp_evolution_encoding`, or
            `"control_diagonal_generator_lcu"` from
            `build_control_diagonal_generator_encoding`.
        metadata: JSON-like record of how the circuit was built. Where it
            repeats the subnormalization, ancilla count, error or
            construction, the fields above are the values to use.
    """

    circuit: QuantumCircuit
    alpha: float
    num_ancillas: int
    system_qubits: int
    error_bound: float | None
    implementation: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if finite_real(self.alpha, "alpha") <= 0:
            raise ValueError("alpha must be positive")
        if self.error_bound is not None and finite_real(self.error_bound, "error_bound") < 0:
            raise ValueError("error_bound must be nonnegative")
        integer(self.num_ancillas, "num_ancillas", 0)
        integer(self.system_qubits, "system_qubits", 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like block-encoding summary."""

        return {
            "alpha": self.alpha,
            "num_ancillas": self.num_ancillas,
            "system_qubits": self.system_qubits,
            "error_bound": self.error_bound,
            "implementation": self.implementation,
            "metadata": _json_ready(self.metadata),
            "num_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "gate_counts": {name: int(count) for name, count in self.circuit.count_ops().items()},
        }


@dataclass(frozen=True, kw_only=True)
class BlockEncodingPlan:
    """The block encoding to build, chosen before any circuit exists.

    [`plan_block_encoding`][nwqlib.subroutines.block_encoding.plan_block_encoding]
    returns it, and
    [`build_block_encoding_from_plan`][nwqlib.subroutines.block_encoding.build_block_encoding_from_plan]
    builds exactly this construction from `source` and `decomposition`
    without choosing again. `alpha`, `num_ancillas` and `error_bound` are
    known before the circuit is built. Construction requires a finite
    `alpha`, positive except for `"exact_zero"`, a finite nonnegative
    `error_bound` and nonnegative integer widths. The fields below are
    read-only.

    Attributes:
        requested_implementation: Name the caller passed: `"auto"`,
            `"pauli_lcu"`, `"multiplexed_pauli"`, `"banded"` or
            `"dense_dilation"`.
        implementation: Construction the builder will use:
            `"multiplexed_pauli"`, `"banded"` or `"dense_dilation"`.
            LCHS's compiled QSP SELECT also records `"exact_zero"` for the
            H part of `A = L + iH` when relative pruning keeps none of its
            Pauli terms. No encoding is built for that part. The record has
            zero `alpha`, error and ancilla count, no source and no
            decomposition, and it cannot be built into a circuit.
        alpha: Subnormalization in the units of A. The all-zero ancilla
            block of the unitary is `A / alpha`. It is the Pauli
            coefficient 1-norm for multiplexed Pauli, the band-coefficient
            1-norm for banded, and `||A||_2` or the supplied normalization
            for dense dilation. Zero only for `"exact_zero"`.
        num_ancillas: Ancilla qubits of the encoding: `ceil(log2 L)` for
            L Pauli terms, `ceil(log2 B)` for B bands, 1 for dense
            dilation and 0 for `"exact_zero"`.
        system_qubits: Number n of system qubits. A acts on dimension `2**n`.
        error_bound: Upper bound on `||A - alpha * block||_2` in the units
            of A. `plan_block_encoding` sets 0 for multiplexed Pauli, the
            circulant-detection bound for a detected banded input, and the
            outward-rounded normalization residual bound for dense dilation.
            A caller that pruned Pauli terms first, such as LCHS's compiled
            QSP SELECT, adds the pruned coefficient mass.
        source: Input the builder reads: the complex128 `2**n`-square
            matrix for dense dilation (the converted input, or the dense
            expansion of a Pauli input), the `BandSpecification` for
            banded, the converted dense matrix or None (Pauli input) for
            multiplexed Pauli, and None for `"exact_zero"`.
        decomposition: The `PauliDecomposition` whose terms SELECT applies
            for multiplexed Pauli. For dense dilation it is the Pauli form of
            the input, supplied or computed for the cost comparison, or None
            when dense dilation was requested for a dense matrix.
            `plan_block_encoding` sets None for banded plans. LCHS's compiled
            QSP SELECT attaches its kept terms to every plan it returns
            except `"exact_zero"`, which has None.
        detail: JSON-like record of the choice. After a Pauli cost
            comparison it holds the SELECT gate counts (term count, padded
            table size, per-qubit effective control counts, the per-qubit
            projected address supports and CX counts), the predicted
            per-query CX of both candidates, the Pauli alpha and, when it
            was computed, the dense-dilation alpha, and the reason for the
            choice. For banded it holds the band count, how the bands were
            detected and the gate counts. It is empty when dense dilation
            was requested for a dense matrix. The Pauli and dense-dilation
            builders copy it into the circuit metadata.
    """

    requested_implementation: str
    implementation: str
    alpha: float
    num_ancillas: int
    system_qubits: int
    error_bound: float
    source: Any
    decomposition: Any | None
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        alpha = finite_real(self.alpha, "alpha")
        error = finite_real(self.error_bound, "error_bound")
        integer(self.num_ancillas, "num_ancillas", 0)
        integer(self.system_qubits, "system_qubits", 0)
        if self.implementation == "exact_zero":
            if (alpha != 0 or error != 0 or self.num_ancillas != 0
                    or self.source is not None or self.decomposition is not None):
                raise ValueError("exact_zero must represent an omitted zero child")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")
        if error < 0:
            raise ValueError("error_bound must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        """Return the plan's scalar fields and its `detail` record as JSON-like data."""

        return {
            "requested_implementation": self.requested_implementation,
            "implementation": self.implementation,
            "alpha": self.alpha,
            "num_ancillas": self.num_ancillas,
            "system_qubits": self.system_qubits,
            "error_bound": self.error_bound,
            **_json_ready(self.detail),
        }


def _json_ready(value: Any) -> Any:
    """Convert NumPy arrays and scalars, complex numbers, mappings and sequences to JSON types."""
    if isinstance(value, np.ndarray):
        return array_to_json(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def block_encoding_top_left(unitary: Any, *, num_ancillas: int) -> np.ndarray:
    """Return the all-zero ancilla block of a block-encoding unitary matrix.

    The ancilla register precedes the system register, so the ancilla bits
    are the low-order bits of Qiskit's little-endian index, and the block
    entry `(A / alpha)[s', s]` sits at row `s' * 2**num_ancillas` and column
    `s * 2**num_ancillas`. The function reads the matrix at that stride.

    Args:
        unitary (array_like): Full unitary matrix of the circuit, for
            example `Operator(encoding.circuit).data`.
        num_ancillas (int): Number of ancilla qubits.

    Returns:
        block (numpy.ndarray): The encoded `A / alpha` block.
    """

    matrix = np.asarray(unitary, dtype=complex)
    step = 2**num_ancillas
    return matrix[::step, ::step]


def _as_power_of_two_matrix(matrix: Any) -> np.ndarray:
    """Return a square complex matrix with power-of-two dimension."""

    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("block encoding requires a square two-dimensional matrix")
    dimension = array.shape[0]
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("block encoding requires a power-of-two matrix dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("block encoding requires finite matrix entries")
    return array


def _base_metadata(
    *,
    resolved: str,
    requested: str,
    preprocessing_label: str,
    preprocessing_note: str,
    register_order: tuple[str, str],
) -> dict[str, Any]:
    """Return shared block-encoding construction provenance."""

    return {
        "requested_block_encoding_implementation": requested,
        "preprocessing_label": preprocessing_label,
        "classical_preprocessing_note": preprocessing_note,
        "implementation_metadata": implementation_metadata(BLOCK_ENCODING_IMPLEMENTATIONS, resolved,
                                                           slot="block encoding"),
        "register_order": list(register_order),
    }


# Local plan/build work ceiling in the size units of ``_admit_dense_input``
# and ``_admit_pauli_plan``. Callers can lower or raise it. The value and the
# dense completion law are registered in docs/ENGINEERING_CONSTANTS.md.
DEFAULT_MAX_BLOCK_WORK = 1_000_000_000

# Relative rounding slack of a dense-dilation alpha against a computed value
# that is at most ||A||_2 in exact arithmetic, registered in
# docs/ENGINEERING_CONSTANTS.md as the "Dense-dilation normalization
# relative slack".
_DENSE_ALPHA_RELATIVE_SLACK = 1.0e-12


def _require_alpha_covers(alpha: float, magnitude: float, name: str) -> None:
    """Refuse a dense-dilation ``alpha`` that lies below ``magnitude`` by more than rounding.

    ``magnitude`` is a computed value that is at most ``||A||_2`` in exact
    arithmetic, either the largest entry magnitude or the largest singular
    value of a computed SVD. LAPACK's SVD is backward stable, so it
    returns the singular values of ``A + E`` with ``||E||_2`` a modest
    multiple of the unit roundoff times ``||A||_2``, and by Weyl's
    inequality each computed singular value lies within ``||E||_2`` of the
    exact one. A norm computed that way, and an alpha taken from it, can
    therefore fall a few units in the last place below an entry magnitude
    or below the exact norm. The test admits an alpha within
    ``_DENSE_ALPHA_RELATIVE_SLACK`` times alpha of ``magnitude``, about 9000
    unit roundoffs, which covers that error at the dimensions a dense
    dilation serves, and refuses a larger deficit. The message keeps both
    raw values.
    """
    if alpha + _DENSE_ALPHA_RELATIVE_SLACK * alpha < magnitude:
        raise ValueError(f"planned dense-dilation alpha {alpha!r} is below ||A||_2: {name} is {magnitude!r}")


def _pauli_dependency_supports(decomposition: Any) -> list[list[int]]:
    """Return each system qubit's projected address support, from X/Z masks packed once.

    The masks are packed qubit zero first, as ``_pauli_masks`` numbers them
    (bit j of word j // 64 is system qubit j), and ``local_pauli_dependencies``
    gives the ordered support of each system qubit.
    """
    term_count = len(decomposition.terms)
    words = max(1, -(-decomposition.num_qubits // 64))
    x = np.zeros((term_count, words), dtype=np.uint64)
    z = np.zeros((term_count, words), dtype=np.uint64)
    for index, term in enumerate(decomposition.terms):
        flip, phase, _ = _pauli_masks(term.label)
        for word in range(words):
            x[index, word] = (flip >> (64 * word)) & 0xFFFFFFFFFFFFFFFF
            z[index, word] = (phase >> (64 * word)) & 0xFFFFFFFFFFFFFFFF
    return [list(local_pauli_dependencies(x, z, system_index))
            for system_index in range(decomposition.num_qubits)]


def _pauli_plan_detail(decomposition: Any) -> dict[str, Any]:
    """Return the dependency-projected Pauli SELECT structural census.

    The census predicts what ``_pauli_lcu_encoding`` builds without building
    it. Each system qubit gets one UCG over the projected address support of
    its local Pauli factors (``projected_unitary_resource_law``). Nonzero
    coefficient phases add one diagonal on the a-qubit padded address
    register, ``2**a - 2`` CX by the Theorem 7 recursion of Shende et al.,
    quant-ph/0406176v5, the diagonal law of ``multiplexor_resource_law``
    written inline here. The PREP pair is two
    positive-amplitude magnitude trees of ``2**a - 2`` CX each
    (``direct_preparation_cx_bound``).

    ``_pauli_dependency_supports`` computes the projected address support of
    each system qubit, which is the support that
    ``project_unitary_table_dependencies`` defines, and the plan stores it as
    ``dependency_supports``. The builder reads it from the plan and appends
    the projected tables it determines (``project_local_pauli_table``).
    Coefficient phases stay in their separate diagonal.
    """

    term_count = len(decomposition.terms)
    padded_table_size = 1 << max(term_count - 1, 0).bit_length()
    control_qubits = max(term_count - 1, 0).bit_length()
    supports = _pauli_dependency_supports(decomposition)
    effective_controls: list[int] = []
    core_cx = 0
    completion_cx = 0
    for support in supports:
        law = projected_unitary_resource_law(len(support))
        effective_controls.append(len(support))
        core_cx += law["ucg_core_cx"]
        completion_cx += law["ucg_completion_diagonal_cx"]
    phases = tuple(float(np.angle(term.coefficient)) for term in decomposition.terms)
    coefficient_diagonal_cx = (
        max(0, padded_table_size - 2)
        if any(phase != 0.0 for phase in phases)
        else 0
    )
    prep_pair_cx = 2 * direct_preparation_cx_bound(control_qubits, complex_phases=False)
    return {
        "term_count": term_count,
        "padded_table_size": padded_table_size,
        "effective_control_counts": effective_controls,
        "dependency_supports": supports,
        "ucg_core_cx": core_cx,
        "ucg_completion_diagonal_cx": completion_cx,
        "coefficient_diagonal_cx": coefficient_diagonal_cx,
        "prep_pair_cx": prep_pair_cx,
    }


def _pauli_lcu_encoding(
    plan: BlockEncodingPlan,
    *, max_bytes=DEFAULT_MAX_BYTES, max_work=DEFAULT_MAX_BLOCK_WORK,
) -> BlockEncoding:
    """Build the multiplexed-Pauli block encoding (PREP, dependency-projected SELECT, PREP adjoint) of ``plan``.

    For ``A = sum_j c_j P_j`` with ``L`` terms, PREP prepares
    ``sum_j sqrt(|c_j| / alpha) |j>`` on ``ceil(log2 L)`` address qubits and
    SELECT applies ``exp(i arg c_j) P_j`` for address ``j``. The all-zero
    address block is then ``A / alpha`` with ``alpha = sum_j |c_j|`` (Low and
    Chuang, arXiv:1610.06546v3, Lemma 5). SELECT applies the coefficient
    phases as one diagonal on the address register and then one exact UCG
    per system qubit. ``term.label[-1 - q]`` is the factor on system qubit
    ``q`` because Qiskit labels list qubit 0 last. Padded addresses have zero
    PREP amplitude, identity factors and zero phase.
    """
    from nwqlib.subroutines.lcu.core import _assemble_prepared_lcu, _build_lcu_preparation_artifact, prepare_lcu_gate_data
    from qiskit import QuantumCircuit, QuantumRegister


    decomposition = plan.decomposition
    if decomposition is None:
        raise ValueError("multiplexed Pauli plan is missing its decomposition")
    if not decomposition.terms:
        raise ValueError(
            "pauli_lcu cannot block-encode the zero matrix (no Pauli terms, alpha would be 0)"
        )
    coefficients = [term.coefficient for term in decomposition.terms]
    data = prepare_lcu_gate_data(
        coefficients,
        system_dimension=decomposition.operator_dimension,
        coefficient_atol=decomposition.atol,
        max_bytes=max_bytes, max_work=max_work,
    )
    preparation = _build_lcu_preparation_artifact(data, max_bytes=max_bytes, max_work=max_work)

    if data.num_control_qubits:
        control = QuantumRegister(data.num_control_qubits, "lcu_control")
        system = QuantumRegister(data.num_system_qubits, "system")
        select = QuantumCircuit(control, system, name="multiplexed_pauli_select")
        control_qubits = list(control)
    else:
        system = QuantumRegister(data.num_system_qubits, "system")
        select = QuantumCircuit(system, name="multiplexed_pauli_select")
        control_qubits = []

    phases = [float(np.angle(term.coefficient)) for term in decomposition.terms]
    if any(phase != 0.0 for phase in phases):
        append_control_diagonal_phases(select, control_qubits, phases)
    labels = [term.label for term in decomposition.terms]
    # The projected address supports planned by _pauli_plan_detail.
    supports = plan.detail["dependency_supports"]
    for system_index, target in enumerate(system):
        append_projected_unitary_table(select, target, control_qubits,
            project_local_pauli_table(labels, system_index, tuple(supports[system_index])))

    shell = QuantumCircuit(*select.qregs, name="multiplexed_pauli")
    lcu = _assemble_prepared_lcu(
        data,
        preparation,
        select,
        circuit=shell,
        preparation_qubits=control_qubits,
        select_qubits=list(shell.qubits),
        preparation_backend="direct",
    )
    metadata = _base_metadata(
        resolved="multiplexed_pauli",
        requested=plan.requested_implementation,
        preprocessing_label="small_dense_validation",
        preprocessing_note=(
            "Pauli decomposition taken of a dense input matrix (up to 4^n terms); "
            "small-dense validation preprocessing, no quantum-advantage claim."
        ),
        register_order=("lcu_control", "system"),
    )
    metadata["pauli_term_count"] = len(decomposition.terms)
    # Every Pauli string is Hermitian. Real coefficients prove the premise
    # directly in the stored representation, without a 2**n square matrix.
    # Unique Pauli words are linearly independent, so an imaginary coefficient
    # then disproves the premise. Repeated words can cancel and stay unknown.
    hermitian = (True if all(complex(term.coefficient).imag == 0.0
                             for term in decomposition.terms) else
                 False if len({term.label for term in decomposition.terms}) == len(decomposition.terms)
                 else None)
    metadata["target_operator_is_hermitian"] = hermitian
    metadata["encoded_operator_is_hermitian"] = hermitian
    metadata["lcu"] = lcu.to_dict()
    metadata.update(dict(plan.detail))
    return BlockEncoding(
        circuit=lcu.circuit,
        alpha=plan.alpha,
        num_ancillas=lcu.data.num_control_qubits,
        system_qubits=lcu.data.num_system_qubits,
        error_bound=plan.error_bound,
        implementation="multiplexed_pauli",
        metadata=metadata,
    )


def _dense_dilation_encoding(
    matrix: Any,
    *,
    requested_implementation: str,
    normalization: float,
    planned_error_bound: float,
    selected_svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> BlockEncoding:
    """Build the one-ancilla dense dilation of ``A``, reusing a supplied SVD and error bound.

    ``normalization`` and ``planned_error_bound`` are the plan's alpha and
    bound. ``selected_svd`` is the SVD ``(W, s, V^dagger)`` of the matrix
    computed during planning. Only the SVD is computed here when absent.

    The dilation is ``U = [[A/alpha, K], [K, -A/alpha]]`` with
    ``K = W sqrt(I - S^2) V^dagger`` from the SVD ``A = W (alpha S) V^dagger``.
    ``U`` is exactly unitary because ``K`` shares the singular frames of
    ``A/alpha``: in those frames each 2x2 block row is
    ``[s_i, sqrt(1 - s_i^2)]`` and its partner, which are orthonormal. The
    ``error_bound`` bounds the normalization residual ``A - alpha * (A / alpha)``
    in the supplied operator's units. The separate unitarity check tests the
    completed dilation and does not certify circuit-synthesis roundoff.

    The caller that supplies the factors is responsible for their being an
    SVD of this matrix. QLS, for example, extends the SVD of its unpadded
    matrix with identity frames and singular value ``alpha`` on the padding
    coordinates, so the singular values are not sorted. This constructor
    never recomputes supplied factors.

    The single ancilla is circuit qubit 0, the low-order bit.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    array = _as_power_of_two_matrix(matrix)
    dimension = array.shape[0]
    alpha = float(normalization)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("dense-dilation normalization must be finite and positive")
    # Every entry magnitude is at most ||A||_2, so this cheap test rejects
    # some alphas below the spectral norm before the SVD.
    _require_alpha_covers(alpha, float(np.max(np.abs(array))), "the largest entry magnitude")
    if selected_svd is None:
        left, singular_values, right_h = np.linalg.svd(array)
    else:
        left, singular_values, right_h = selected_svd
        if (left.shape != (dimension, dimension) or right_h.shape != (dimension, dimension)
                or singular_values.shape != (dimension,) or np.any(singular_values < 0)
                or not all(np.isfinite(value).all() for value in selected_svd)):
            raise ValueError("selected singular frames differ from the dense-dilation dimensions/domain")
    largest_singular = float(np.max(singular_values))
    _require_alpha_covers(alpha, largest_singular, "the largest singular value")
    scaled = normalized_matrix(array, alpha)
    scaled_singular = singular_values / alpha
    # A singular value above alpha within the slack of _require_alpha_covers
    # makes 1 - s**2 slightly negative. Clipping it to zero avoids a NaN in
    # K, and the unitarity check below bounds the resulting deviation.
    complement = (left * np.sqrt(np.clip(1.0 - scaled_singular**2, 0.0, None))) @ right_h

    # Ancilla is circuit qubit 0 (low-order bit): index = (s << 1) | a.
    full = np.zeros((2 * dimension, 2 * dimension), dtype=complex)
    full[0::2, 0::2] = scaled
    full[1::2, 0::2] = complement
    full[0::2, 1::2] = complement
    full[1::2, 1::2] = -scaled

    unitarity_deviation = float(
        np.max(np.abs(full @ full.conj().T - np.eye(2 * dimension)))
    )
    # Construction guard, not a physical tolerance. Float64 SVD roundoff lands near 1e-14 at
    # validation dimensions, and a construction defect lands at O(1). 1e-10 is four orders
    # above the roundoff. Registered in docs/ENGINEERING_CONSTANTS.md. Revisit for larger
    # dimensions or another precision.
    if not np.isfinite(unitarity_deviation) or unitarity_deviation > 1.0e-10:
        raise RuntimeError(
            f"SVD dilation failed the unitarity check (deviation {unitarity_deviation:.3e})"
        )

    num_system_qubits = int(np.log2(dimension))
    ancilla = QuantumRegister(1, "dilation_ancilla")
    system = QuantumRegister(num_system_qubits, "system")
    circuit = QuantumCircuit(ancilla, system, name="dense_dilation")
    circuit.append(
        UnitaryGate(full, label="dense_dilation", check_input=False), [ancilla[0], *system]
    )

    metadata = _base_metadata(
        resolved="dense_dilation",
        requested=requested_implementation,
        preprocessing_label="small_dense_validation",
        preprocessing_note=(
            "dense SVD dilation synthesized as one dense UnitaryGate; eager "
            "O(4^(n+1)) synthesis for small validation circuits only."
        ),
        register_order=("dilation_ancilla", "system"),
    )
    # This construction already owns the dense input and its normalized copy.
    # Compare their stored entries with O(d) row workspace, without making an
    # adjoint matrix or expanding a structured operator or circuit. Real scalar
    # normalization preserves a proved Hermitian input, including signed zero.
    target_hermitian = _stored_dense_is_hermitian(array)
    metadata["target_operator_is_hermitian"] = target_hermitian
    metadata["encoded_operator_is_hermitian"] = (
        True if target_hermitian else _stored_dense_is_hermitian(scaled)
    )
    metadata["unitarity_deviation"] = unitarity_deviation
    metadata["matrix_dimension"] = dimension
    metadata["error_evaluation"] = "row/column upper bound on the normalization residual"
    return BlockEncoding(
        circuit=circuit,
        alpha=alpha,
        num_ancillas=1,
        system_qubits=num_system_qubits,
        error_bound=planned_error_bound,
        implementation="dense_dilation",
        metadata=metadata,
    )


def _stored_dense_is_hermitian(array):
    """Return whether the stored entries satisfy ``A[i, j] == conj(A[j, i])`` exactly, in O(d²) time and O(d) workspace."""
    return all(np.array_equal(array[row, row:], array[row:, row].conj())
               for row in range(len(array)))


def _dense_dilation_predicted_cx(system_qubits: int) -> float:
    """Return the paper's QSD CX count at block-encoding routing width, the estimate that routing compares.

    The count is ``(23/48) 4**m - (3/2) 2**m + 4/3`` at width ``m = n + 1``,
    which includes the dilation ancilla. It is Eq. (19) (p. 16) and the row
    "QSD (l = 2, optimized)" of Table 1 (p. 14) in Shende et al.,
    quant-ph/0406176v5. That count comes from the quantum Shannon
    decomposition (Theorem 13) recursed down to two-qubit blocks with the
    optimizations of their Appendix A. Routing uses this paper count for the
    dense candidate. A lowered dilation is synthesized by
    ``_dense_synthesis.dense_unitary_circuit``, which usually takes fewer CX
    and can take more (``backends.resources.dense_unitary_cx_qsd_upper_bound``).
    """

    return float(dense_unitary_cx_qsd_upper_bound(system_qubits + 1))


def _spectral_norm_upper_bound(array: np.ndarray) -> float:
    """Return ``sqrt(||A||_1 ||A||_inf)`` without spectral decomposition.

    This bounds ``||A||_2`` because ``||A||_2**2 <= ||A||_1 ||A||_inf``.
    Nonzero entries are rounded up by one ulp before the outward-rounded row
    and column sums.
    """

    absolute = np.abs(array)
    np.nextafter(absolute, np.inf, out=absolute, where=absolute != 0.0)
    return _norm_bound_from_sums(np.sum(absolute, axis=1), np.sum(absolute, axis=0))


def _norm_bound_from_sums(row_sums: np.ndarray, column_sums: np.ndarray) -> float:
    """Bound the spectral norm from nonnegative row and column sums with outward rounding."""

    row_max, column_max = float(np.max(row_sums)), float(np.max(column_sums))
    if row_max == 0.0 or column_max == 0.0:
        return 0.0
    # Each nonnegative addend is attenuated by at most (1-u)^(n-1) >= 1-(n-1)u.
    unit_roundoff = np.finfo(float).eps / 2.0
    row_max = np.nextafter(row_max / (1.0 - (column_sums.size - 1) * unit_roundoff), np.inf)
    column_max = np.nextafter(column_max / (1.0 - (row_sums.size - 1) * unit_roundoff), np.inf)
    # Separate square roots avoid underflow or overflow of the product.
    bound = float(
        np.nextafter(np.sqrt(row_max), np.inf)
        * np.nextafter(np.sqrt(column_max), np.inf)
    )
    return float(np.nextafter(bound, np.inf))


def _normalization_error_bound(
    array: np.ndarray, alpha: float, scaled: np.ndarray
) -> float:
    """Bound ``A - alpha * scaled`` including multiplication roundoff hidden by a rounded zero residual."""

    component_bounds = []
    alpha_is_power_of_two = np.frexp(alpha)[0] == 0.5
    for original, component in ((array.real, scaled.real), (array.imag, scaled.imag)):
        product = alpha * component
        residual = np.abs(original - product)
        np.nextafter(residual, np.inf, out=residual, where=residual != 0.0)
        # Multiplication by a power of two is exact when its result is normal.
        exact_product = (component == 0.0) | (
            (alpha_is_power_of_two | (np.abs(np.frexp(component)[0]) == 0.5))
            & (np.abs(product) >= np.finfo(float).tiny)
        )
        product_roundoff = np.zeros_like(product)
        np.spacing(np.abs(product), out=product_roundoff, where=~exact_product)
        residual += product_roundoff
        np.nextafter(residual, np.inf, out=residual, where=residual != 0.0)
        component_bounds.append(residual)
    # |Re R| + |Im R| bounds |R| entrywise without another square root.
    absolute_bound = component_bounds[0] + component_bounds[1]
    np.nextafter(absolute_bound, np.inf, out=absolute_bound, where=absolute_bound != 0.0)
    return _spectral_norm_upper_bound(absolute_bound)


def _pauli_predicted_cx(detail: Mapping[str, Any]) -> float:
    """Return the per-query CX count of the Pauli encoding from its ``_pauli_plan_detail`` census.

    It sums the UCG cores and their completing diagonals over the system
    qubits, the coefficient-phase diagonal and the PREP pair.
    """

    return float(
        int(detail["ucg_core_cx"])
        + int(detail["ucg_completion_diagonal_cx"])
        + int(detail["coefficient_diagonal_cx"])
        + int(detail["prep_pair_cx"])
    )


def _shift_pauli_trace_sum(
    *,
    num_qubits: int,
    offset: int,
    flip_mask: int,
    phase_mask: int,
) -> int:
    """Sum Pauli phases over a cyclic shift using a two-carry-state bit DP.

    Returns the sum over ``x`` with ``x XOR f == (x + b) mod 2**n`` of
    ``(-1)**popcount(z & ((x + b) mod 2**n))``, for flip mask ``f``, phase
    mask ``z`` and offset ``b``. For ``P = i**y X**f Z**z`` with ``y`` Y
    factors, ``i**y`` times this sum is ``Tr(P^dagger S**b)``. Scanning the
    bits from least to most significant with the addition carry as state
    costs ``O(n)`` instead of ``O(2**n)``.
    """

    states = [1, 0]
    for qubit in range(num_qubits):
        next_states = [0, 0]
        for carry, subtotal in enumerate(states):
            for input_bit in (0, 1):
                total = input_bit + ((offset >> qubit) & 1) + carry
                output_bit = total & 1
                if input_bit ^ output_bit != ((flip_mask >> qubit) & 1):
                    continue
                signed = -subtotal if ((phase_mask >> qubit) & 1 and output_bit) else subtotal
                next_states[total >> 1] += signed
        states = next_states
    return sum(states)


def _detect_banded_pauli_structure(
    decomposition: PauliDecomposition,
) -> tuple[Any, float]:
    """Certify a circulant from kept Pauli terms in polynomial work.

    With ``P = i**y X**f Z**z``, ``P|0> = i**y |f>``, so the candidate band at
    offset ``f`` is ``beta_f = A[f, 0] = sum_{P with flip mask f} c_P i**y``.
    The candidate circulant ``C = sum_f beta_f S**f`` has Pauli coefficients
    ``c_P(C) = 2**-n sum_f beta_f Tr(P^dagger S**f)``, which are evaluated
    exactly in rationals with ``_shift_pauli_trace_sum``. Every supplied
    coefficient must match its prediction within the relative window below.
    Pauli strings are orthogonal, so ``||M||_F**2 = 2**n sum_P |c_P(M)|**2``
    and ``sum_P |c_P(C)|**2 = sum_f |beta_f|**2``. The squared mismatch on
    supplied labels plus the candidate mass on absent labels is therefore
    ``||A - C||_F**2 / 2**n``, and the returned error evaluates
    ``||A - C||_F``, which bounds the operator-norm difference. Automatic
    routing accepts only a zero error.
    """

    from nwqlib.subroutines.block_encoding.banded import BandSpecification

    dimension = decomposition.operator_dimension
    band_coefficients: dict[int, complex] = {}
    parsed_terms = []
    for term in decomposition.terms:
        masks = _pauli_masks(term.label)
        parsed_terms.append((term, masks))
        flip_mask, _phase_mask, y_count = masks
        band_coefficients[flip_mask] = (
            band_coefficients.get(flip_mask, 0.0j)
            + term.coefficient * (1.0j**y_count)
        )
    band_coefficients = {
        offset: coefficient
        for offset, coefficient in band_coefficients.items()
        if coefficient != 0.0
    }
    if not parsed_terms or not band_coefficients:
        raise ValueError("banded structure detection found only zero bands")

    # Untuned relative window for binary64 coefficients against their exact rational
    # prediction. It grows with the qubit, term and band counts, and a mismatch above it
    # rejects the circulant candidate. The exact residual is computed separately.
    # Registered in docs/ENGINEERING_CONSTANTS.md. Revisit with a changed accumulation or
    # classification algorithm.
    relative_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1,
            decomposition.num_qubits,
            len(parsed_terms),
            len(band_coefficients),
        )
    )
    def exact_complex(value: complex) -> tuple[Fraction, Fraction]:
        return (
            Fraction.from_float(float(value.real)),
            Fraction.from_float(float(value.imag)),
        )

    exact_band_coefficients = {
        offset: exact_complex(coefficient)
        for offset, coefficient in band_coefficients.items()
    }
    expected_coefficients: list[tuple[Fraction, Fraction]] = []
    for term, (flip_mask, phase_mask, y_count) in parsed_terms:
        expected_real = expected_imag = Fraction()
        for offset, (band_real, band_imag) in exact_band_coefficients.items():
            trace_sum = _shift_pauli_trace_sum(
                num_qubits=decomposition.num_qubits,
                offset=offset,
                flip_mask=flip_mask,
                phase_mask=phase_mask,
            )
            expected_real += trace_sum * band_real
            expected_imag += trace_sum * band_imag
        expected_real /= dimension
        expected_imag /= dimension
        for _ in range(y_count % 4):
            expected_real, expected_imag = -expected_imag, expected_real
        expected = complex(float(expected_real), float(expected_imag))
        scale = max(abs(term.coefficient), abs(expected))
        if abs(term.coefficient - expected) > relative_tolerance * scale:
            raise ValueError("Pauli decomposition is not a certified circulant operator")
        expected_coefficients.append((expected_real, expected_imag))

    def exact_squared_mass(values) -> Fraction:
        return sum(
            (component**2 for value in values for component in value),
            Fraction(),
        )

    band_mass = exact_squared_mass(exact_band_coefficients.values())
    projected_mass = exact_squared_mass(expected_coefficients)
    missing_mass = band_mass - projected_mass
    if missing_mass < 0:
        raise ValueError("Pauli decomposition is not a certified circulant operator")
    supplied_residual_mass = sum(
        (
            (component - expected_component) ** 2
            for (term, _), expected in zip(
                parsed_terms, expected_coefficients, strict=True
            )
            for component, expected_component in zip(
                exact_complex(term.coefficient), expected, strict=True
            )
        ),
        Fraction(),
    )
    residual_mass = supplied_residual_mass + missing_mass
    if residual_mass == 0:
        detection_error = 0.0
    else:
        try:
            residual_mass_upper = float(np.nextafter(float(residual_mass), np.inf))
            # ||R||_2 <= ||R||_F = sqrt(2**n * residual_mass). Round each
            # floating operation outward, since an upper input alone does
            # not make nearest-rounded sqrt and multiplication upper bounds.
            root_upper = np.nextafter(np.sqrt(residual_mass_upper), np.inf)
            dimension_root_upper = np.nextafter(
                2.0 ** (decomposition.num_qubits / 2.0), np.inf)
            detection_error = float(np.nextafter(root_upper * dimension_root_upper, np.inf))
        except OverflowError as error:
            raise OverflowError("circulant certificate error bound overflowed") from error
        if not np.isfinite(detection_error):
            raise FloatingPointError("circulant certificate error bound is not finite")

    bands = [
        (
            offset - dimension if offset > dimension // 2 else offset,
            complex(coefficient),
        )
        for offset, coefficient in band_coefficients.items()
    ]
    bands.sort(key=lambda item: item[0])
    return (
        BandSpecification(
            offsets=tuple(offset for offset, _ in bands),
            coefficients=tuple(coefficient for _, coefficient in bands),
            num_qubits=decomposition.num_qubits,
        ),
        detection_error,
    )


def _admit_dense_input(operator, *, max_bytes, max_work, construction=False, selected_svd=False):
    """Check the dense byte and work laws before any conversion, SVD or completion.

    Only the shape, or the row lengths of a nested sequence, is read first,
    so an input above the caller's limits fails before any array is formed.
    The ``DEFAULT_MAX_BLOCK_WORK`` row in docs/ENGINEERING_CONSTANTS.md
    registers these laws.

    With ``construction=True`` the 384 bytes per entry of the d-square
    matrix count the main arrays of ``_dense_dilation_encoding`` as if they
    were alive together: the complex128 source, ``W``, ``V^dagger``,
    ``A/alpha``, the product ``W sqrt(1 - s**2)`` and the complement ``K``
    (6 x 16 = 96), the 2d-square dilation ``U``, its conjugate copy,
    ``U U^dagger`` and its difference from the identity (4 x 64 = 256), and
    the float64 2d-square identity (32). The conjugate copy and the product
    ``W sqrt(1 - s**2)`` are freed before the identity is formed, so at
    most 304 bytes per entry are alive at once, or 272 when NumPy writes
    the difference into ``U U^dagger``. The later temporaries, such as the
    magnitudes of the difference, are smaller than the arrays freed before
    them. The QLS padded completion
    (``algorithms/qls/quantum.py``) uses the margin for the unpadded
    singular frames it keeps beside the padded ones. The ``32 d`` bytes are
    four float64 vectors: the singular values, ``s/alpha`` and two
    temporaries of ``sqrt(1 - s**2)``.

    Without construction, 192 bytes per entry (12 complex128 d-square
    arrays) is an allowance for planning a dense input. Its largest counted
    step, the normalization bound of a dense dilation, holds the source,
    ``A/alpha`` and at most 50 bytes per entry of float64 and Boolean
    temporaries, 82 in total. The allowance also covers Qiskit's conversion
    of a Pauli input to a dense matrix, whose internal arrays are not
    documented.
    """
    shape = getattr(operator, "shape", None)
    if shape is None:
        if not isinstance(operator, (list, tuple)) or not operator:
            raise ValueError("block encoding requires a square two-dimensional matrix")
        d = len(operator)
        if any(not isinstance(row, (list, tuple, np.ndarray)) or len(row) != d for row in operator):
            raise ValueError("block encoding requires a square two-dimensional matrix")
    else:
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("block encoding requires a square two-dimensional matrix")
        d = int(shape[0])
    if d <= 0 or d & (d - 1):
        raise ValueError("block encoding requires a power-of-two matrix dimension")
    # 384 d**2 counts the completion arrays as if alive together and
    # 192 d**2 is the planning allowance (see the docstring).
    # 32 d: four float64 vectors.
    _check_bytes((384 if construction else 192) * d * d + 32*d, max_bytes,
        "block-encoding dense arrays")
    # Completion work is 8d**3 for the SVD, d**3 for the complement product
    # and 8d**3 = (2d)**3 for the unitarity product of the full 2d-square
    # dilation, 17d**3 in total, or 9d**3 when the SVD is supplied. Without
    # construction, 16 units per entry cover the intake scan.
    work = (9 if selected_svd else 17) * d**3 if construction else 16 * d * d
    cap = integer(max_work, "max_work", 1)
    if work > cap:
        stage = "completion" if construction else "intake scan"
        raise ValueError(f"block-encoding dense work exceeds max_work: {stage} work {work} for dimension {d}, max_work={cap}")
    return d


def _circulant_classification_work(q, terms, bands):
    """Logical-visit charge C(q, m, b) of one circulant classification of m kept terms.

    The classifier path is ``_detect_circulant`` -> ``_detect_banded_pauli_structure``
    -> ``_shift_pauli_trace_sum``. It parses each label into flip and phase
    masks and accumulates the candidate first-column coefficients. There are
    only d = 2**q possible flip masks, so the nonzero candidate band count b
    is at most min(m, d). The trace kernel is called once per term and band
    checked, at most m*b times, and always scans q bits with two carry states
    times two candidate input bits per bit (a carry dynamic program for
    ``x XOR f == (x + b) mod d``), with four rational multiply/add operations
    per term-band visit. Hence

        C(q,m,b) = (q+32)m + 4(q+1)mb + 16b + 2b*ceil(log2(max(1,b))) + 32,

    with C = 0 for a child with no kept terms. The qm term parses labels, the
    32 per term covers band accumulation and the fixed per-term
    normalization, mismatch and rational mass passes, the 16 per band covers
    filtering, rational conversion, squared mass and band-record formation,
    and the sorting allowance covers the final integer-offset sort. Work
    counts logical visits: a numerical array-element addition,
    multiplication, negation, absolute value, maximum or comparison is one
    visit, a scan or reduction visits each entry once, character production
    or first hashing costs one visit per character, and a coefficient
    conversion, output-record emission or dictionary insertion is one
    visit. A carry-transition visit and a rational arithmetic visit are
    counted separately. Fractions can have thousands of bits, so one
    rational visit is not a constant CPU-time operation. These units are
    admission proxies, not timings or equal-cost CPU operations.
    """
    if terms == 0:
        return 0
    sort_levels = max(0, (bands - 1).bit_length())
    return ((q + 32) * terms
            + 4 * (q + 1) * terms * bands
            + 16 * bands + 2 * bands * sort_levels + 32)


def _admit_pauli_plan(decomposition, *, held_bytes, max_bytes, max_work, classify):
    r"""Admit the Pauli SELECT tables and optional circulant check before work.

    With P padded addresses, n system qubits and m terms, the SELECT tables
    are charged ``64 P (n + 16)`` bytes, 64 bytes (one 2x2 complex128
    factor) per address and system qubit plus an untuned 16 such slots per
    address, and ``32*P*max(1,n)**2`` work. ``_build_block_encoding_from_plan``
    maps both terms to the tables the builder holds and the operations it
    performs, and the banded plan uses the same law.

    ``held_bytes`` is B_held, the caller's already admitted payloads that
    stay live in this phase, each counted once under its owner's law: each
    distinct live dense matrix at 16D^2, the decomposition's labels and
    coefficient records, and completed child plans or tables that remain
    live. Passing the original max_bytes independently to two children is
    not total-memory admission.

    SELECT construction adds ``64P(n+16)``, and an active circulant
    classifier adds the allowance ``C_B`` given below, including its fixed
    bookkeeping allowance. With ``classify=False``, or with an empty skipped
    classifier, the classifier allowance is zero. These charges follow the
    stated logical payload conventions and do not bound process RSS.

    With ``classify``, the circulant classification is admitted before
    candidate bands are known with b = min(m, 2**n). Its work is
    ``_circulant_classification_work(n, m, b)``. Admit the active
    classifier's linear parsed, band and exact-integer workspace together
    with the SELECT tables and all other live payloads. Sequential
    classifiers share workspace capacity, while completed child outputs
    remain charged. For a supplied decomposition,

        \[B_{\rm held}=m(n+16),\qquad B_{\rm inner}=m(n+16)+64P(n+16)+\mathbf1_{\{\mathrm{classify}\land m>0\}}C_B.\]

    For dense input after decomposition, with ``classify=False``,

        \[B_{\rm held}=16D^2+m(n+16),\qquad B_{\rm inner}=16D^2+m(n+16)+64P(n+16).\]

    Here D = 2**n. Additional distinct live matrices or completed child
    outputs contribute to both B_held and B_inner. The active classifier
    allowance is

        C_B = [128L(2J)+512]V + H0,
        J = 4208 + 2n + 2 ceil(log2(b+1)) + ceil(log2(m+b+1)),  V = m+b+n+1,

    with L(b) = ``integer_object_bytes`` and H0 = ``BOOKKEEPING_BYTES``. The
    detector holds parsed term and mask tuples, a band dictionary, exact band
    coefficients, expected rational coefficients and one carry recurrence
    with its current arithmetic. There is no m*b rational table. Every finite
    binary64 band component is an integer multiple of 2**-1074 below 2**1024
    in magnitude. Division by d gives a common denominator at most
    2**(1074+n). Predicted components are below b*2**1024 and residual
    components below (b+1)*2**1024, so their squared masses over m terms and
    b bands fit a numerator width of 4196 + 2n + 2 ceil(log2(b+1)) +
    ceil(log2(m+b+1)) plus a small fixed allowance, and J supplies twelve
    extra bits. The factor 128 bounds the simultaneous integer population
    per V, L(2J) covers each integer's temporary product width and object
    representation, and 512V covers parsed tuples and list slots, Fraction
    wrappers, dictionary entries and floating complex values. The m(n+16)
    label and complex-coefficient population is charged only when it is
    materialized independently. The decomposition these callers classify is
    already in B_held, so it is not charged twice here. ``classify=False``
    omits the classifier storage and work.

    Returns:
        The admitted work, so the caller can add later stages to it.
        It is ``32*P*max(1,n)**2`` plus
        ``_circulant_classification_work(n,m,b)`` for an active classifier,
        with P, n, m and b as defined above.
        The caller's ``max_work`` stays the budget of this plan.
    """
    n, terms = decomposition.num_qubits, len(decomposition.terms)
    table = 1 << max(terms - 1, 0).bit_length()
    size = held_bytes + 64 * table * (n + 16)
    work = 32 * table * max(1, n)**2
    # An empty decomposition skips classification and has no classifier workspace.
    if classify and terms:
        bands = min(terms, 1 << n)
        # ceil(log2(k + 1)) == k.bit_length() for every integer k >= 0.
        width = 4208 + 2 * n + 2 * bands.bit_length() + (terms + bands).bit_length()
        size += (128 * integer_object_bytes(2 * width) + 512) * (terms + bands + n + 1) + BOOKKEEPING_BYTES
        work += _circulant_classification_work(n, terms, bands)
    _check_bytes(size, max_bytes, "block-encoding Pauli classification/table arrays")
    if work > integer(max_work, "max_work", 1):
        raise ValueError("block-encoding Pauli work exceeds max_work")
    return work


def plan_block_encoding(
    operator: Any,
    *,
    implementation: str = "auto",
    normalization: float | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_BLOCK_WORK,
) -> BlockEncodingPlan:
    """Choose a block encoding and compute its `alpha`, width and error bound without building it.

    An exact circulant, either a supplied `BandSpecification` or structure
    detected with zero error, goes to `banded` first. Otherwise `"auto"`
    compares the per-query CX counts of multiplexed Pauli SELECT and dense
    dilation and selects the smaller. A tie selects the smaller alpha, with
    dense dilation winning when both alpha values are equal. Automatic
    routing never introduces an approximate structure. An explicit
    `"banded"` request keeps the detector tolerance and records the
    resulting bound as `error_bound`.
    [`build_block_encoding_from_plan`][nwqlib.subroutines.block_encoding.build_block_encoding_from_plan]
    builds the returned plan's construction without comparing candidates
    again. Pass it the same `max_bytes` and `max_work`.

    A supplied `normalization` becomes `alpha` only when the plan uses
    dense dilation. Without one, dense dilation uses the optimal spectral
    normalization `||A||_2`. Pauli SELECT always uses its coefficient
    1-norm. Arithmetic failures in structural error bounds propagate rather
    than selecting another encoding without the required bound. For a
    `PauliDecomposition`, the target operator is its kept terms. The
    encoding bound does not include loss from an earlier decomposition, so
    its caller must carry any original-to-kept approximation separately.
    The plan computes the normalization without keeping the singular
    vectors, so building it later computes its own SVD for the synthesis.

    Args:
        operator (array_like | PauliDecomposition | BandSpecification): The
            operator `A`, as for `build_block_encoding`.
        implementation (str): `"auto"` (default), `"pauli_lcu"`,
            `"multiplexed_pauli"`, `"banded"` or `"dense_dilation"`.
        normalization (float | None): `alpha` to use if the plan is a dense
            dilation. Default `None`, which selects `||A||_2`.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the array bytes of each planning step, as for
            `build_block_encoding`.
        max_work (int): Default `1_000_000_000`. Limit on the work of each
            planning step, as for `build_block_encoding`.

    Returns:
        plan (BlockEncodingPlan): The chosen construction with its `alpha`,
            `num_ancillas` and `error_bound`.

    Raises:
        ValueError: For unknown implementation names, operator and
            implementation mismatches, a zero operator, a matrix that is not
            square with power-of-two dimension, a `normalization` that is not
            finite and positive or lies below `||A||_2`, a limit that is not a
            positive integer, or a planning step that exceeds `max_bytes` or
            `max_work`.
        OverflowError: If structural error-bound arithmetic overflows.
        FloatingPointError: If a structural error bound is nonfinite.
    """

    return _plan_block_encoding(operator, implementation=implementation, normalization=normalization,
        max_bytes=max_bytes, max_work=max_work)


def _detect_circulant(operator, implementation):
    """Return ``(specification, detection, detection_error)`` for the circulant route.

    A supplied ``BandSpecification`` is used as given, with zero error. For
    ``auto`` and ``banded``, a Pauli input is checked by
    ``_detect_banded_pauli_structure``, and ``auto`` keeps the result only
    when its error is exactly zero. A dense input is scanned with
    ``atol=0`` under ``auto`` and with the detector's default tolerance under
    an explicit ``banded`` request, whose error then enters ``error_bound``.
    A failed detection under ``auto`` means no circulant. Under an explicit
    ``banded`` request it raises. Other requests skip detection.
    """
    from nwqlib.subroutines.block_encoding.banded import (
        BandSpecification,
        _detect_banded_structure_with_error,
    )

    specification = None
    detection = None
    detection_error = 0.0
    if isinstance(operator, BandSpecification):
        specification = operator
        detection = "given_band_specification"
    elif isinstance(operator, PauliDecomposition) and implementation in ("auto", "banded"):
        try:
            specification, detection_error = _detect_banded_pauli_structure(operator)
        except ValueError:
            if implementation == "banded":
                raise
        else:
            detection = "detected_from_pauli_decomposition"
            if implementation == "auto" and detection_error != 0.0:
                specification = None
    elif implementation == "banded":
        array = _as_power_of_two_matrix(operator)
        specification, detection_error = _detect_banded_structure_with_error(array)
        detection = "detected_from_dense_matrix"
    elif implementation == "auto":
        array = _as_power_of_two_matrix(operator)
        try:
            specification, detection_error = _detect_banded_structure_with_error(array, atol=0.0)
        except ValueError:
            specification = None
        else:
            detection = "detected_from_dense_matrix"
    return specification, detection, detection_error


def _banded_plan(specification, *, implementation, detection, detection_error,
                 max_bytes, max_work):
    """Admit and describe the shift-LCU encoding of a circulant ``specification``.

    ``alpha`` is the band-coefficient 1-norm and the address register has
    ``ceil(log2 B)`` qubits for B bands. ``periodic_gate_counts`` records the
    census of ``build_banded_block_encoding``: one QFT pair, one UCRZ table
    per system qubit (``2**a`` CX each for a >= 1 address qubits), an
    address diagonal of ``2**a - 2`` CX when any of its phases is nonzero,
    and a PREP pair of positive-amplitude trees. The table admission uses the
    same SELECT-table byte and work allowances as ``_admit_pauli_plan``.
    """
    from nwqlib.subroutines.block_encoding.banded import _validate_band_specification

    integer(specification.num_qubits, "num_qubits", 1)
    bands = len(specification.offsets)
    ancillas = max(bands - 1, 0).bit_length()
    _check_bytes(64 * (1 << ancillas) * (specification.num_qubits + 16), max_bytes,
        "block-encoding band tables")
    if 32 * (1 << ancillas) * max(1, specification.num_qubits)**2 > max_work:
        raise ValueError("block-encoding band work exceeds max_work")
    _validate_band_specification(specification)
    dimension = 1 << specification.num_qubits
    # The address-diagonal phases of build_banded_block_encoding: the
    # coefficient phase plus half of each UCRZ angle 2 pi 2**q b / N, because
    # P(phi) = exp(i phi / 2) RZ(phi). Scaling by 2 and 1/2 is exact in
    # binary64, so these values equal the builder's bit for bit.
    correction_phases = [
        float(np.angle(coefficient))
        + sum(
            np.pi * (1 << system_index) * offset / dimension
            for system_index in range(specification.num_qubits)
        )
        for offset, coefficient in zip(
            specification.offsets,
            specification.coefficients,
            strict=True,
        )
    ]
    ucrz_law = multiplexor_resource_law("ucrz", control_qubits=ancillas)
    diagonal_law = multiplexor_resource_law("diagonal", control_qubits=ancillas)
    diagonal_active = any(phase != 0.0 for phase in correction_phases)
    return BlockEncodingPlan(
        requested_implementation=implementation,
        implementation="banded",
        alpha=float(np.sum(np.abs(np.asarray(specification.coefficients)))),
        num_ancillas=ancillas,
        system_qubits=specification.num_qubits,
        error_bound=detection_error,
        source=specification,
        decomposition=None,
        detail={
            "num_bands": bands,
            "padded_table_size": 1 << ancillas,
            "structure_detection": detection,
            "periodic_gate_counts": {
                "qft_pairs": 1,
                "ucrz_multiplexor_gates": specification.num_qubits,
                "ucrz_table_entries_per_gate": ucrz_law["table_entries"],
                "ucrz_basis_cx_per_gate": ucrz_law["basis_cx_gates"],
                "control_diagonal_gates": int(diagonal_active and ancillas > 0),
                "control_diagonal_table_entries": diagonal_law["table_entries"],
                "control_diagonal_basis_cx": (
                    diagonal_law["basis_cx_gates"] if diagonal_active else 0
                ),
                "prep_register_qubits": ancillas,
                "prep_pair_cx": 2 * direct_preparation_cx_bound(ancillas, complex_phases=False),
            },
        },
    )


def _dense_norm_work(dimension, full_svd):
    """Return the work of the spectral norm that ``_plan_block_encoding`` computes.

    A ``dense_norm`` from the caller computes a full SVD of the d-square
    matrix, whose frames the dense completion reuses, charged ``8 d**3``
    like every full SVD of the dense laws. The default
    ``np.linalg.norm(A, 2)`` computes the singular values alone, charged
    ``_linalg_laws.singular_values_work``.
    """
    return 8 * dimension**3 if full_svd else singular_values_work(dimension)


def _explicit_dense_dilation_plan(array, *, implementation, normalization, dense_norm,
                                  full_svd, work_used, max_work, system_qubits):
    """Plan the dense dilation that the caller requested explicitly for a dense matrix.

    Without a supplied ``normalization``, ``alpha = ||A||_2`` from
    ``dense_norm``, charged ``_dense_norm_work(d, full_svd)``. The entrywise
    test ``max|A_ij| <= alpha``, with the rounding slack of
    ``_require_alpha_covers``, rejects some alphas below the spectral norm
    before any construction, since every entry magnitude is at most
    ``||A||_2``. The error bound is that of ``_normalization_error_bound``.
    """
    if normalization is None and work_used + _dense_norm_work(array.shape[0], full_svd) > max_work:
        raise ValueError("block-encoding normalization exceeds max_work")
    alpha = (
        float(normalization) if normalization is not None else dense_norm(array)
    )
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            "dense-dilation normalization must be finite and positive (zero matrix is unsupported)"
        )
    _require_alpha_covers(alpha, float(np.max(np.abs(array))), "the largest entry magnitude")
    return BlockEncodingPlan(
        requested_implementation=implementation,
        implementation="dense_dilation",
        alpha=alpha,
        num_ancillas=1,
        system_qubits=system_qubits,
        error_bound=_normalization_error_bound(array, alpha, normalized_matrix(array, alpha)),
        source=array,
        decomposition=None,
        detail={},
    )


def _plan_block_encoding(operator, *, implementation, normalization=None,
                         max_bytes, max_work, dense_norm=None):
    """Select one encoding for ``operator`` and return its plan without building a circuit.

    An exact circulant routes to ``banded`` (``_detect_circulant`` and
    ``_banded_plan``). An explicit dense dilation of a dense matrix is
    planned by ``_explicit_dense_dilation_plan``. Otherwise a dense matrix is
    first Pauli-decomposed with exact-zero removal only, and ``auto``
    compares the Pauli and dense-dilation CX laws. ``build_block_encoding``
    passes ``dense_norm`` so that the SVD computing ``||A||_2`` is kept for
    the completion. Without it the norm comes from ``np.linalg.norm(A, 2)``.
    """
    from nwqlib.subroutines.block_encoding.banded import BandSpecification

    valid_names = (
        "auto",
        "pauli_lcu",
        *sorted(BLOCK_ENCODING_IMPLEMENTATIONS),
    )
    if implementation not in valid_names:
        joined = "', '".join(valid_names)
        raise ValueError(f"block encoding implementation must be one of: '{joined}'")
    integer(max_work, "max_work", 1)
    _check_bytes(0, max_bytes)
    work_used = 0
    if normalization is not None and finite_real(normalization, "normalization") <= 0:
        raise ValueError("normalization must be positive")
    if isinstance(operator, PauliDecomposition):
        # B_held: the supplied decomposition's labels and complex coefficients,
        # the m(n+16) logical population, stay live while it is classified.
        work_used = _admit_pauli_plan(operator, held_bytes=len(operator.terms) * (operator.num_qubits + 16),
            max_bytes=max_bytes, max_work=max_work, classify=implementation in ("auto", "banded"))
    elif not isinstance(operator, BandSpecification):
        d = _admit_dense_input(operator, max_bytes=max_bytes, max_work=max_work)
        work_used = 16*d*d
    full_svd_requested = dense_norm is not None
    if dense_norm is None:
        def dense_norm(matrix):
            return float(np.linalg.norm(matrix, 2))

    specification, detection, detection_error = _detect_circulant(operator, implementation)
    if specification is not None and implementation in ("auto", "banded"):
        return _banded_plan(specification, implementation=implementation, detection=detection,
            detection_error=detection_error, max_bytes=max_bytes, max_work=max_work)
    if isinstance(operator, BandSpecification):
        raise ValueError(
            f"implementation '{implementation}' requires a dense matrix input; a "
            "BandSpecification routes to the 'banded' implementation"
        )

    array = None
    if isinstance(operator, PauliDecomposition):
        decomposition = operator
        system_qubits = decomposition.num_qubits
    else:
        array = _as_power_of_two_matrix(operator)
        system_qubits = int(np.log2(array.shape[0]))
    if implementation == "dense_dilation" and array is not None:
        return _explicit_dense_dilation_plan(array, implementation=implementation,
            normalization=normalization, dense_norm=dense_norm, full_svd=full_svd_requested,
            work_used=work_used, max_work=max_work, system_qubits=system_qubits)

    if array is not None:
        d = array.shape[0]
        # The block transform has q levels over d**2 entries (32 units each).
        # The planning allowance of 192 d**2 bytes covers the source, the two
        # live complex128 levels and the half-sum temporaries, about 64 d**2
        # together. Up to d**2 Pauli terms are charged like SELECT table rows,
        # 64 (q + 16) bytes each.
        work_used += 32 * max(1,system_qubits) * d * d
        if work_used > max_work:
            raise ValueError(
                f"block-encoding Pauli decomposition exceeds max_work: it needs {work_used} work units, "
                f"including {32 * max(1, system_qubits) * d * d} for the transform of dimension {d}, "
                f"max_work={max_work}. Raise max_work to at least {work_used}.")
        _check_bytes(192*d*d + 64*d*d*(system_qubits+16), max_bytes,
            "block-encoding Pauli decomposition arrays")
        # Reuse the range-safe exact-zero block transform. Qiskit's squared
        # magnitude zero test can discard tiny terms even with atol=rtol=0.
        decomposition = PauliDecomposition(
            terms=tuple(PauliTerm(label=label, coefficient=value)
                for label, value in pauli_coefficients(array)),
            input_dimension=d, operator_dimension=d, num_qubits=system_qubits,
            atol=0.0)
        # B_held: the dense input (16 d**2) and the decomposition's labels and
        # complex coefficients (m(n+16)) stay live while the tables are planned.
        work_used += _admit_pauli_plan(decomposition,
            held_bytes=16 * d * d + len(decomposition.terms) * (system_qubits + 16),
            max_bytes=max_bytes, max_work=max_work, classify=False)
        if work_used > max_work:
            raise ValueError(
                f"block-encoding Pauli selection exceeds max_work: it needs {work_used} work units for "
                f"{len(decomposition.terms)} Pauli terms, max_work={max_work}. "
                f"Raise max_work to at least {work_used}.")
    if not decomposition.terms:
        raise ValueError(
            "pauli_lcu cannot block-encode the zero matrix (no Pauli terms, alpha would be 0)"
        )
    detail = _pauli_plan_detail(decomposition)
    pauli_alpha = float(fsum(abs(term.coefficient) for term in decomposition.terms))
    pauli_cost = _pauli_predicted_cx(detail)
    dense_cost = _dense_dilation_predicted_cx(system_qubits)
    selected = "dense_dilation" if implementation == "dense_dilation" else "multiplexed_pauli"
    dense_alpha = None if normalization is None else float(normalization)
    if implementation == "auto" and pauli_cost >= dense_cost:
        selected = "dense_dilation"
    dense_source = array
    # A Pauli input selected for dense dilation is expanded to its d-square
    # matrix (L d**2 work for L terms) while its term table stays alive, and
    # its spectral norm then costs _dense_norm_work unless a normalization is
    # given.
    if selected == "dense_dilation" and dense_source is None:
        d = decomposition.operator_dimension
        retained_bytes = 64*len(decomposition.terms)*(system_qubits+16)
        _check_bytes((384 if full_svd_requested else 192)*d*d+32*d+retained_bytes,
            max_bytes, "block-encoding Pauli dense conversion and kept table")
        work_used += len(decomposition.terms)*d*d
        norm_work = _dense_norm_work(d, full_svd_requested) if dense_alpha is None else 0
        if work_used + norm_work > max_work:
            raise ValueError("block-encoding Pauli dense conversion exceeds max_work")
        dense_source = np.asarray(decomposition.to_sparse_pauli_op().to_matrix(), dtype=complex)
    if selected == "dense_dilation" and dense_alpha is None:
        assert dense_source is not None
        d = dense_source.shape[0]
        retained_bytes = 64*len(decomposition.terms)*(system_qubits+16)
        _check_bytes((384 if full_svd_requested else 192)*d*d+32*d+retained_bytes,
            max_bytes, "block-encoding normalization and kept table")
        if work_used + _dense_norm_work(d, full_svd_requested) > max_work:
            raise ValueError("block-encoding normalization exceeds max_work")
        dense_alpha = dense_norm(dense_source)
    if selected == "dense_dilation":
        if not np.isfinite(dense_alpha) or dense_alpha <= 0.0:
            raise ValueError("dense-dilation normalization must be finite and positive")
        _require_alpha_covers(dense_alpha, float(np.max(np.abs(dense_source))), "the largest entry magnitude")
    # At equal CX cost, the smaller alpha gives the stronger encoded amplitude.
    # A supplied dense normalization can exceed the Pauli triangle bound, and then the tie
    # selects Pauli.
    if implementation == "auto" and pauli_cost == dense_cost:
        assert dense_alpha is not None
        if dense_alpha > pauli_alpha:
            selected = "multiplexed_pauli"
    reason = (
        f"auto selected {selected}: predicted per-query CX "
        f"multiplexed_pauli={pauli_cost:g}, dense_dilation={dense_cost:g}"
        if implementation == "auto"
        else f"explicit implementation '{implementation}' selected {selected}"
    )
    routing = {
        **detail,
        "error_evaluation": (
            "row/column upper bound on the normalization residual for the supplied operator"
            if selected == "dense_dilation"
            else "ideal block identity for the supplied Pauli terms; upstream pruning error excluded"
            if array is None else "ideal block identity for the supplied matrix; no coefficient pruning"
        ),
        "predicted_per_query_cx": {
            "multiplexed_pauli": pauli_cost,
            "dense_dilation": dense_cost,
        },
        "selection_reason": reason,
        "normalization_records": {
            "multiplexed_pauli": pauli_alpha,
            **({"dense_dilation": dense_alpha} if dense_alpha is not None else {}),
        },
        **({"structured_input": "pauli_decomposition"} if array is None else {}),
    }
    if selected == "dense_dilation":
        return BlockEncodingPlan(
            requested_implementation=implementation,
            implementation=selected,
            alpha=float(dense_alpha),
            num_ancillas=1,
            system_qubits=system_qubits,
            error_bound=_normalization_error_bound(
                dense_source, dense_alpha, normalized_matrix(dense_source, dense_alpha)
            ),
            source=dense_source,
            decomposition=decomposition,
            detail=routing,
        )
    return BlockEncodingPlan(
        requested_implementation=implementation,
        implementation=selected,
        alpha=pauli_alpha,
        num_ancillas=max(len(decomposition.terms) - 1, 0).bit_length(),
        system_qubits=system_qubits,
        error_bound=0.0,
        source=array,
        decomposition=decomposition,
        detail=routing,
    )


def build_block_encoding(
    operator: Any,
    *,
    implementation: str = "auto",
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_work: int = DEFAULT_MAX_BLOCK_WORK,
) -> BlockEncoding:
    """Build a block-encoding circuit of a square matrix, a Pauli sum or a circulant.

    The function chooses the construction as
    [`plan_block_encoding`][nwqlib.subroutines.block_encoding.plan_block_encoding]
    does and builds it. It computes one dense SVD and uses it for the
    normalization, an equal-cost comparison and the completion. The
    [constructions table](block_encoding.md#constructions)
    lists what each construction encodes, its `alpha` and its ancillas.

    Args:
        operator (array_like | PauliDecomposition | BandSpecification): A
            dense square matrix (power-of-two dimension), a
            `PauliDecomposition` whose kept terms define the operator, or a
            `BandSpecification` describing a periodic banded Toeplitz
            operator without forming its matrix.
        implementation (str): `"auto"` (default), `"pauli_lcu"`,
            `"banded"`, or `"dense_dilation"`. `"multiplexed_pauli"` is
            accepted as the resolved name of `"pauli_lcu"`. `"auto"` looks
            for exact banded structure first, then compares the per-query CX
            counts of the Pauli and dense-dilation constructions without
            building either one.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit in bytes on the array data that each planning or
            construction step holds at the same time, checked before the
            step allocates it. NumPy, LAPACK and Qiskit internal workspace
            and Python object overhead are not counted, except in the
            circulant classification of a Pauli input, which counts its
            Python integers, parsed tuples, Fraction wrappers and dictionary
            entries, and a fixed allowance of 65,536 bytes for scalar
            bookkeeping and ndarray headers.
        max_work (int): Default `1_000_000_000`. Limit on the work of each
            step, in NWQLib's work units, which estimate scalar operations:
            `d**3` for a product of two d-square matrices and `8 d**3` for
            a dense SVD.
            Planning, the dense SVD and construction are each checked
            against it separately.

    Returns:
        encoding (BlockEncoding): The circuit with its `alpha`,
            `num_ancillas` and `error_bound`.

    Raises:
        ValueError: For unknown implementation names, operator and
            implementation mismatches, or a step that exceeds `max_bytes` or
            `max_work`.
        OverflowError: If structural error-bound arithmetic overflows.
        FloatingPointError: If a structural error bound is nonfinite.

    Examples:
        `A = [[1, 0.5], [0.5, -1]]` has spectral norm
        `sqrt(5)/2 = 1.1180339887...`. For this one-qubit matrix `"auto"`
        chooses the one-ancilla dense dilation, whose `alpha` is that norm,
        and `alpha` times the all-zero ancilla block of the circuit's
        unitary is `A`:

        >>> import numpy as np
        >>> from qiskit.quantum_info import Operator
        >>> from nwqlib.subroutines.block_encoding import (
        ...     block_encoding_top_left, build_block_encoding)
        >>> A = np.array([[1.0, 0.5], [0.5, -1.0]])
        >>> encoding = build_block_encoding(A)
        >>> print(encoding.implementation, round(encoding.alpha, 10))
        dense_dilation 1.1180339887
        >>> U = Operator(encoding.circuit).data
        >>> block = block_encoding_top_left(U, num_ancillas=1)
        >>> print(np.allclose(encoding.alpha * block, A))
        True
    """

    selected_svd = None

    def dense_norm(matrix):
        nonlocal selected_svd
        _admit_dense_input(matrix, max_bytes=max_bytes, max_work=max_work, construction=True)
        selected_svd = np.linalg.svd(matrix)
        return float(selected_svd[1][0])

    plan = _plan_block_encoding(operator, implementation=implementation,
        max_bytes=max_bytes, max_work=max_work, dense_norm=dense_norm)
    return _build_block_encoding_from_plan(plan, max_bytes=max_bytes, max_work=max_work,
        selected_svd=selected_svd)


def build_block_encoding_from_plan(plan: BlockEncodingPlan, *, max_bytes=DEFAULT_MAX_BYTES,
                                   max_work=DEFAULT_MAX_BLOCK_WORK) -> BlockEncoding:
    """Build the circuit of a `BlockEncodingPlan` without choosing again.

    Args:
        plan (BlockEncodingPlan): A plan from `plan_block_encoding`.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Pass the value used for the plan.
        max_work (int): Default `1_000_000_000`. Pass the value used for
            the plan.

    Returns:
        encoding (BlockEncoding): The circuit of `plan.implementation`, with
            the plan's `alpha`, `num_ancillas` and `error_bound`.

    Raises:
        ValueError: If a construction step exceeds `max_bytes` or
            `max_work`.
    """
    return _build_block_encoding_from_plan(plan, max_bytes=max_bytes, max_work=max_work)


def _build_block_encoding_from_plan(plan, *, max_bytes, max_work, selected_svd=None):
    """Check the plan's byte and work laws, then build ``plan.implementation`` as planned.

    Dense dilation is checked with the completion law of
    ``_admit_dense_input``. A plan that also keeps a Pauli decomposition of
    L terms keeps it alive during the completion, so the completion bytes
    are checked together with the ``64 L (n + 16)`` bytes that
    ``_plan_block_encoding`` charges for the kept term table.

    Multiplexed Pauli and banded plans are checked with the table law of
    ``_admit_pauli_plan`` for ``P = 2**num_ancillas`` padded addresses and n
    system qubits. For each system qubit the Pauli builder reads the
    address support that ``_pauli_plan_detail`` stored in the plan, builds
    the projected table of at most P local 2x2
    factors as complex128 matrices (64 bytes each) from the labels, and
    pads it for the ``UCGate``, whose at most P matrices stay in the
    circuit. The n kept
    tables give the ``64 P n`` bytes of matrix data, and the transient
    copies of one qubit's table fit in the untuned 16 extra slots per
    address. This table charge excludes the NumPy object headers of the
    2x2 arrays. The circulant-classification charge in ``_admit_pauli_plan``
    includes its integer objects, parsed tuples, Fraction wrappers,
    dictionary entries and fixed bookkeeping allowance. ``_admit_lcu``
    checks the PREP arrays separately. The banded builder's per-qubit tables
    of P RZ angles are smaller. The work per system qubit grows as ``P (a + c)`` for
    ``a = log2 P`` address qubits and a small constant c: a pairwise
    equality test per address bit (on the packed letter codes, at
    planning), the padding, the projection, the label lookups
    and the ``UCGate`` unitarity check of each 2x2 matrix. The
    untuned ``32 P n`` per qubit of the ``32 P n**2`` law grows at least as
    fast when ``a <= 2n``, which holds for distinct Pauli words. A
    decomposition with repeated words can exceed it.

    A banded plan detected from other input carries its detection bound into
    ``error_bound``. ``selected_svd`` lets a full build reuse the SVD that
    already fixed alpha for the dense completion.
    """
    integer(max_work, "max_work", 1)
    _check_bytes(0, max_bytes)
    if plan.implementation == "dense_dilation":
        if plan.decomposition is not None:
            d = 1 << plan.system_qubits
            _check_bytes(384*d*d+32*d+64*len(plan.decomposition.terms)*(plan.system_qubits+16),
                max_bytes, "block-encoding completion and kept Pauli table")
        _admit_dense_input(plan.source, max_bytes=max_bytes, max_work=max_work, construction=True,
            selected_svd=selected_svd is not None)
    else:
        n, a = plan.system_qubits, plan.num_ancillas
        _check_bytes(64*(1 << a)*(n+16), max_bytes, "block-encoding construction tables")
        if 32*(1 << a)*max(1,n)**2 > max_work:
            raise ValueError("block-encoding construction exceeds max_work")

    if plan.implementation == "multiplexed_pauli":
        return _pauli_lcu_encoding(plan, max_bytes=max_bytes, max_work=max_work)
    if plan.implementation == "banded":
        from nwqlib.subroutines.block_encoding.banded import build_banded_block_encoding

        encoding = build_banded_block_encoding(
            plan.source,
            requested_implementation=plan.requested_implementation,
        )
        detection = plan.detail.get("structure_detection")
        if detection in (
            "detected_from_dense_matrix",
            "detected_from_pauli_decomposition",
        ):
            from_dense = detection == "detected_from_dense_matrix"
            return replace(
                encoding,
                metadata={
                    **dict(encoding.metadata),
                    "structure_detection": detection,
                    "target_operator_is_hermitian": (
                        encoding.metadata.get("target_operator_is_hermitian")
                        if plan.error_bound == 0.0 else None
                    ),
                    "preprocessing_label": (
                        "small_dense_validation" if from_dense else "stored_pauli_structure"
                    ),
                    "classical_preprocessing_note": (
                        "circulant structure detected from a dense validation-scale matrix"
                        if from_dense
                        else "circulant structure detected directly from kept Pauli terms"
                    ),
                },
                error_bound=plan.error_bound,
            )
        return replace(encoding, error_bound=plan.error_bound)
    encoding = _dense_dilation_encoding(
        plan.source,
        requested_implementation=plan.requested_implementation,
        normalization=plan.alpha,
        planned_error_bound=plan.error_bound,
        selected_svd=selected_svd,
    )
    return replace(
        encoding,
        metadata={**dict(encoding.metadata), **dict(plan.detail)},
    )


__all__ = [
    "BlockEncoding",
    "BlockEncodingPlan",
    "block_encoding_top_left",
    "build_block_encoding",
    "build_block_encoding_from_plan",
    "plan_block_encoding",
]
