"""Block-encoding records, the choice of construction, and the dense constructions.

Definition (Gilyen, Su, Low, and Wiebe, arXiv:1806.01838v1, Definition 43):
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
  number of terms. With real coefficients SELECT applies
  ``sign(c_j) P_j`` and the whole circuit is an involution
  (``_pauli_plan_detail``).
* ``banded`` encodes a circulant ``A = sum_b beta_b S^b`` as an exact LCU of
  cyclic shifts with ``alpha = sum_b |beta_b|`` and ``ceil(log2 B)``
  ancillas for ``B`` bands (``banded.py``). It never forms a dense matrix. A
  Hermitian specification is built with paired labels as an involution.
* ``dense_dilation`` uses one ancilla and the smallest zero-error
  ``alpha = ||A||_2``. The completed unitary on ``n + 1`` qubits is
  synthesized as one dense gate, whose cost grows as ``4**(n + 1)``, so it
  serves small validation instances. The automatic choice estimates its
  cost with the CX count of Shende, Bullock, and Markov,
  quant-ph/0406176v5, Table 1 (p. 14) and Eq. (19) (p. 16)
  (``_dense_dilation_predicted_cx``).

``nodes.py`` adds diagonal encodings of quadrature nodes on an address
register. ``build_block_encoding_query`` builds the controlled and adjoint
queries of a planned Pauli or banded encoding by controlling only its
SELECT core, and ``block_encoding_query_cx`` counts
them before construction.

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
from typing import Any, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

import numpy as np
from nwqlib._validation import finite_real, integer
from nwqlib._limits import ResourceLimitError, exceeds, fits_power, memory_limit, optional_limit
from nwqlib._linalg_laws import singular_values_work
from nwqlib._numerics import normalized_matrix
from nwqlib.operators.access import _check_bytes

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
from nwqlib.operators._pauli import PauliTerms, _block_transform, _coefficient_masks, _pauli_masks
from nwqlib.subroutines._serialization import array_to_json


@dataclass(frozen=True, kw_only=True)
class BlockEncoding:
    """A block-encoding circuit with its `alpha`, ancilla count and error bound.

    [`build_block_encoding`][nwqlib.subroutines.block_encoding.build_block_encoding],
    [`build_block_encoding_from_plan`][nwqlib.subroutines.block_encoding.build_block_encoding_from_plan]
    and [`build_banded_block_encoding`][nwqlib.subroutines.block_encoding.build_banded_block_encoding]
    return it, and so do `build_joint_encoding` of this package and
    `build_qsp_evolution_encoding`, `build_control_diagonal_generator_encoding`,
    `build_shared_qsvt_evolution` and `build_qubitized_evolution` of
    `nwqlib.subroutines.qsp`.
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
            module, `"diagonal_rotation_table"` or
            `"uniform_node_<kind>"` from the node builders of
            `nwqlib.subroutines.block_encoding.nodes`,
            `"joint_<construction>"` (`"joint_separate_parts"`,
            `"joint_single_a"`, `"joint_convex_contraction"` or
            `"joint_product"`) from `build_joint_encoding`,
            `"qsp_jacobi_anger_evolution"` from
            `build_qsp_evolution_encoding`,
            `"control_diagonal_generator_lcu"` from
            `build_control_diagonal_generator_encoding`,
            `"shared_qsvt_evolution"` from `build_shared_qsvt_evolution`, or
            `"qubitized_<kind>"` (`"qubitized_jacobi_anger_laurent"`,
            `"qubitized_jacobi_anger_gqet"` or
            `"qubitized_imaginary_time_gqet"`) from
            `build_qubitized_evolution`.
        metadata: JSON-like record of how the circuit was built. Where it
            repeats the subnormalization, ancilla count, error or
            construction, the fields above are the values to use. The
            block-encoding builders of this package record three structural
            facts, each True, False or None (unknown); the evolution
            builders of `nwqlib.subroutines.qsp` do not record them. The
            facts are `"target_operator_is_hermitian"`
            (A is Hermitian), `"encoded_operator_is_hermitian"` (the
            encoded block is Hermitian) and `"full_unitary_involution"` (the
            whole unitary U satisfies `U = U^dagger`, so `U @ U = I`).
            A real Pauli sum, a Hermitian band specification and the node
            constructions give an involution; a dense dilation has no
            involution proof. `"query_census"`, when present, gives the CX
            counts of the forward, adjoint, controlled and
            controlled-adjoint queries that
            [`build_block_encoding_query`][nwqlib.subroutines.block_encoding.build_block_encoding_query]
            builds from the Pauli or banded plan of this encoding, or that
            `build_uniform_node_query` or `build_diagonal_table_query` builds
            from a node recipe or a diagonal table. Passed to
            `build_block_encoding_query` itself, this built encoding is an
            opaque circuit whose controlled cost is unknown. A joint encoding
            records no census, because its structured queries come from its
            `JointEncodingPlan` (`build_joint_encoding`).
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
            LCHS's `qsp_block_encoding` backend also records `"exact_zero"`
            for the H part of `A = L + iH` when relative pruning keeps none
            of its Pauli terms. No encoding is built for that part. The record has
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
            projected address supports and CX counts, and the system qubit
            whose table carries the signs of real coefficients or, without
            system qubits, the addresses whose sign is a phase flip), the
            predicted per-query CX of both constructions, the Pauli alpha
            and, when it was computed, the dense-dilation alpha, and the
            reason for the choice. For banded it holds the band count, how
            the bands were detected and the gate counts, including those of
            the paired labels of a Hermitian specification. Every plan
            from `plan_block_encoding` holds the three structural facts described under
            `BlockEncoding.metadata`, so they are known before a circuit is
            built, and a Pauli or banded plan holds `"query_census"`, the CX
            counts of the queries `build_block_encoding_query` builds from
            this plan. When
            dense dilation was requested for a dense matrix the facts are
            all it holds. The Pauli and dense-dilation builders copy it into
            the circuit metadata.
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
    """Return the part of the construction record that all block encodings share."""

    return {
        "requested_block_encoding_implementation": requested,
        "preprocessing_label": preprocessing_label,
        "classical_preprocessing_note": preprocessing_note,
        "implementation_metadata": implementation_metadata(BLOCK_ENCODING_IMPLEMENTATIONS, resolved,
                                                           slot="block encoding"),
        "register_order": list(register_order),
    }


# Relative rounding slack of a dense-dilation alpha against a computed value
# that is at most ||A||_2 in exact arithmetic, registered in
# docs/ENGINEERING_CONSTANTS.md as the "Dense-dilation normalization
# relative slack".
_DENSE_ALPHA_RELATIVE_SLACK = 1.0e-12


def _require_alpha_covers(alpha: float, magnitude: float, name: str) -> None:
    """Reject a dense-dilation ``alpha`` that lies below ``magnitude`` by more than rounding.

    ``magnitude`` is a computed value that is at most ``||A||_2`` in exact
    arithmetic, either the largest entry magnitude or the largest singular
    value of a computed SVD. LAPACK's SVD is backward stable, so it
    returns the singular values of ``A + E`` with ``||E||_2`` a modest
    multiple of the unit roundoff times ``||A||_2``, and by Weyl's
    inequality each computed singular value lies within ``||E||_2`` of the
    exact one. A norm computed that way, and an alpha taken from it, can
    therefore fall a few units in the last place below an entry magnitude
    or below the exact norm. The test accepts an alpha within
    ``_DENSE_ALPHA_RELATIVE_SLACK`` times alpha of ``magnitude``, about 9000
    unit roundoffs, which covers that error at the dimensions a dense
    dilation serves, and rejects a larger deficit. The message keeps both
    raw values.
    """
    if alpha + _DENSE_ALPHA_RELATIVE_SLACK * alpha < magnitude:
        from nwqlib.algorithms.protocol import ApplicabilityError
        raise ApplicabilityError(f"planned dense-dilation alpha {alpha!r} is below ||A||_2: {name} is {magnitude!r}")


def _pauli_dependency_supports(decomposition: Any, negative: Any = None):
    """Return each system qubit's projected address support and whether its local table has a nonidentity entry.

    The masks are packed qubit zero first, as ``_pauli_masks`` numbers them
    (bit j of word j // 64 is system qubit j), and ``local_pauli_dependencies``
    gives the ordered support of each system qubit. With a Boolean
    ``negative`` row the second list holds the supports of the tables whose
    marked entries carry the factor -1 (``_pauli_plan_detail``), otherwise it
    is None. The third list is True for a system qubit on which some term
    has a letter other than I.

    The labels are packed in tiles of at most 4096 terms: one ASCII byte per
    letter and one bit byte per padded qubit, ``t (q + 64 W)`` bytes for a
    tile of t terms, beside the ``16 W`` bytes per term of the masks.
    """
    terms, q = decomposition.terms, decomposition.num_qubits
    words = max(1, -(-q // 64))
    x = np.zeros((len(terms), words), dtype=np.uint64)
    z = np.zeros((len(terms), words), dtype=np.uint64)
    for start in range(0, len(terms) if q else 0, 4096):
        tile = terms[start:start + 4096]
        # Column j of the reversed letters is system qubit j.
        letters = np.frombuffer("".join(term.label for term in tile).encode("ascii"),
                                dtype=np.uint8).reshape(len(tile), q)[:, ::-1]
        bits = np.zeros((len(tile), 64 * words), dtype=np.uint8)
        for masks, first, second in ((x, ord("X"), ord("Y")), (z, ord("Z"), ord("Y"))):
            bits[:, :q] = (letters == first) | (letters == second)
            masks[start:start + len(tile)] = np.packbits(bits, axis=1, bitorder="little").view("<u8")
    qubits = range(q)
    unsigned = [list(local_pauli_dependencies(x, z, q)) for q in qubits]
    signed = None if negative is None else [list(local_pauli_dependencies(x, z, q, negative)) for q in qubits]
    lettered = [bool(np.any(((x[:, q // 64] | z[:, q // 64]) >> np.uint64(q % 64)) & np.uint64(1)))
                for q in qubits]
    return unsigned, signed, lettered


# Metadata keys of the three structural facts that a block encoding of A
# carries, each True (proved by the construction from the stored values, or
# asserted by the caller of a supplied circuit), False (disproved) or None
# (unavailable). The target fact is that A, the operator alpha times the
# block approximates, is Hermitian. The encoded fact is that the ideal
# all-zero block of the circuit is Hermitian. The involution fact is that the
# whole circuit unitary U satisfies U^dagger = U, hence U**2 = I. A unitary
# involution is Hermitian and so is every compression of it, so a True
# involution with a False encoded fact is a contradiction. The three differ:
# the dense dilation of an exactly Hermitian A has a Hermitian block but no
# involution proof, and a block within epsilon of A can be Hermitian while A
# is not. QSP evolution reads the first two keys from the metadata. The third
# is the premise of a quantum walk that uses U without a Hermitian lift.
STRUCTURE_FACT_KEYS = ("target_operator_is_hermitian", "encoded_operator_is_hermitian", "full_unitary_involution")


def _pauli_structure_facts(decomposition: Any, real: bool) -> dict[str, bool | None]:
    """Return the structural facts of the multiplexed-Pauli encoding of the stored terms.

    Every Pauli string is Hermitian. Real coefficients prove the target and
    the encoded block Hermitian directly in the stored representation,
    without a 2**n square matrix, and the sign-folded SELECT of
    ``_pauli_plan_detail`` makes the whole circuit an involution. Unique
    Pauli words are linearly independent, so an imaginary coefficient then
    disproves Hermiticity, and with it the involution. Repeated words can
    cancel, so all three stay unknown. ``real`` says whether every stored
    coefficient is real.
    """
    unique = len({term.label for term in decomposition.terms}) == len(decomposition.terms)
    hermitian = True if real else False if unique else None
    return dict.fromkeys(STRUCTURE_FACT_KEYS, hermitian)


def _pauli_plan_detail(decomposition: Any) -> dict[str, Any]:
    r"""Return the gate counts of the dependency-projected Pauli SELECT.

    These counts predict what ``_pauli_lcu_encoding`` builds without
    building it. Each system qubit gets one UCG over the projected address support of
    its local Pauli factors (``projected_unitary_resource_law``). Nonzero
    coefficient phases add one diagonal on the a-qubit padded address
    register, ``2**a - 2`` CX by the Theorem 7 recursion of Shende et al.,
    quant-ph/0406176v5, the diagonal count of ``multiplexor_resource_law``,
    written inline here. The PREP pair is two
    positive-amplitude magnitude trees of ``2**a - 2`` CX each
    (``direct_preparation_cx_bound``).

    ``_pauli_dependency_supports`` computes the projected address support of
    each system qubit, which is the support that
    ``project_unitary_table_dependencies`` defines, and the plan records it
    as ``dependency_supports``. The builder reads it from the plan and appends
    the projected tables it determines (``project_local_pauli_table``).

    Real coefficients need no phase diagonal. The SELECT branch of term j is
    ``sign(c_j) P_j``, a Hermitian involution, and the sign is the exact
    factor -1 in the local table of one system qubit, ``sign_qubit``, never
    the phase ``exp(i pi)`` of a rotation angle. SELECT is then the direct
    sum of Hermitian involutions over the addresses (identity on padding),
    so with the same PREP circuit P and its reversed adjoint,

        \[U=P^\dagger\,\mathrm{SELECT}\,P,\qquad U^\dagger=U,\qquad U^2=I,\]

    for every PREP circuit, including its rounding. The all-zero block is
    ``sum_j (|c_j|/alpha) sign(c_j) P_j = A/alpha``. The signs can add
    address bits to the support of the chosen qubit; the planner picks the
    qubit whose UCG gains the fewest CX, then the fewest with one external
    control, then the lowest index. Complex coefficients keep their phases
    in the separate address diagonal, and that encoding is no involution.
    ``nontrivial_tables`` marks the qubits that receive a gate: a support of
    zero bits still has a one-qubit gate when the factor is not I.

    A sum on zero system qubits, such as the 1x1 matrix -1, has no table to
    hold the signs. SELECT is then the diagonal ``sign(c_j)`` on the
    address register, and ``sign_labels`` lists its negative addresses, each
    flipped by ``_append_phase_flip``: an X with ``a - 1`` controls between
    two H on a address qubits, or the global phase pi when the circuit has no
    qubit at all. Controls turn that phase into an exact flip on the control
    value. ``coefficient_diagonal_cx`` counts these flips.
    """

    term_count = len(decomposition.terms)
    padded_table_size = 1 << max(term_count - 1, 0).bit_length()
    control_qubits = max(term_count - 1, 0).bit_length()
    # One pass over the stored coefficients gives both the structure facts and the signs.
    coefficients = np.array([complex(term.coefficient) for term in decomposition.terms], dtype=np.complex128)
    real = not np.any(coefficients.imag)
    facts = _pauli_structure_facts(decomposition, real)
    negative = coefficients.real < 0.0 if real else np.zeros(0, dtype=bool)
    supports, signed, nontrivial = _pauli_dependency_supports(decomposition, negative if negative.any() else None)
    sign_qubit = None
    sign_labels = np.flatnonzero(negative).tolist() if not decomposition.num_qubits else []
    if signed is not None and decomposition.num_qubits:
        def cx(r):
            return projected_unitary_resource_law(r)["basis_cx_gates"]

        def added(q):
            before = (cx(len(supports[q])), cx(len(supports[q]) + 1)) if nontrivial[q] else (0, 0)
            return (cx(len(signed[q])) - before[0], cx(len(signed[q]) + 1) - before[1], q)

        sign_qubit = min(range(decomposition.num_qubits), key=added)
        supports[sign_qubit] = signed[sign_qubit]
        nontrivial[sign_qubit] = True
    effective_controls: list[int] = []
    core_cx = 0
    completion_cx = 0
    for support in supports:
        law = projected_unitary_resource_law(len(support))
        effective_controls.append(len(support))
        core_cx += law["ucg_core_cx"]
        completion_cx += law["ucg_completion_diagonal_cx"]
    prep_pair_cx = 2 * direct_preparation_cx_bound(control_qubits, complex_phases=False)
    detail = {
        "term_count": term_count,
        "padded_table_size": padded_table_size,
        "effective_control_counts": effective_controls,
        "dependency_supports": supports,
        "nontrivial_tables": nontrivial,
        "sign_qubit": sign_qubit,
        "sign_labels": sign_labels,
        "coefficient_phase_diagonal": not real,
        "ucg_core_cx": core_cx,
        "ucg_completion_diagonal_cx": completion_cx,
        "coefficient_diagonal_cx": _sign_label_cx(sign_labels, control_qubits) if real
        else max(0, padded_table_size - 2),
        "prep_pair_cx": prep_pair_cx,
        **facts,
    }
    detail["query_census"] = _query_census(shell="PREP", shell_cx=prep_pair_cx,
                                           core_cx=_pauli_core_cx(detail, 0),
                                           controlled_core_cx=_pauli_core_cx(detail, 1),
                                           shell_bound_cx=prep_pair_cx,
                                           core_bound_cx=_sign_label_cx(sign_labels, control_qubits),
                                           controlled_core_bound_cx=_sign_label_cx(sign_labels, control_qubits + 1))
    return detail


def _sign_label_cx(labels, selecting_qubits: int) -> int:
    """Return the CX of exact phase flips on these labels of m selecting qubits: an X with m - 1 controls each, none for m = 0."""
    return len(labels) * _mcx_cx(selecting_qubits - 1) if labels and selecting_qubits else 0


def _pauli_core_cx(detail: Mapping[str, Any], num_controls: int) -> int:
    """Return the CX of the Pauli SELECT of ``_pauli_select`` with k external controls.

    Each system qubit with a gate has a UCG over its r support bits, or over
    r + k bits when controlled (``_controlled_projection``), counted by
    ``projected_unitary_resource_law``. A complex coefficient diagonal covers
    the a address bits plus the k controls (``multiplexor_resource_law``),
    and so do the sign flips of a sum without system qubits
    (``_sign_label_cx``). With k = 0 this is ``ucg_core_cx +
    ucg_completion_diagonal_cx + coefficient_diagonal_cx``.
    """
    address_qubits = max(detail["padded_table_size"] - 1, 0).bit_length()
    total = sum(projected_unitary_resource_law(r + num_controls)["basis_cx_gates"]
                for r, active in zip(detail["effective_control_counts"], detail["nontrivial_tables"], strict=True)
                if active)
    if detail["coefficient_phase_diagonal"]:
        total += multiplexor_resource_law("diagonal", control_qubits=address_qubits + num_controls)["basis_cx_gates"]
    return total + _sign_label_cx(detail["sign_labels"], address_qubits + num_controls)


def _query_census(*, shell: str, shell_cx: int, core_cx: int, controlled_core_cx: int,
                  workspace_qubits: int = 0, shell_bound_cx: int = 0, core_bound_cx: int = 0,
                  controlled_core_bound_cx: int = 0) -> dict[str, Any]:
    """Return the CX of the four query sequences of a conjugated encoding ``U = B^dagger V B``, with one control.

    The shell B (``shell`` names its parts) and its adjoint are never
    controlled, so their ``shell_cx`` enters every sequence once. The
    adjoint reverses the same gates, so it has the forward count, and the
    controlled adjoint has the controlled count. ``workspace_qubits`` are
    the scratch qubits among the encoding ancillas, besides an address or
    signal register, all returned to zero on the encoded block. The
    encoding applies no reflection.

    The ``*_bound_cx`` arguments are the parts of these counts that are
    upper bounds rather than exact: the slot bound of a PREP tree or of a
    uniform superposition, and Qiskit's count for a multi-controlled X
    without ancillas, which a lowering that borrows idle qubits of the
    surrounding circuit undercuts (an X with four controls took 18 CX
    instead of 36 with two idle qubits, Qiskit 2.6.0). The rest,
    the multiplexors, diagonals, QFT pair and comparison, is the exact count
    of the pinned constructions. ``forward_bound_cx`` and
    ``controlled_bound_cx`` give these parts per sequence.
    """
    return {"shell": shell, "shell_cx": shell_cx, "core_cx": core_cx, "controlled_core_cx": controlled_core_cx,
            "forward_cx": shell_cx + core_cx, "adjoint_cx": shell_cx + core_cx,
            "controlled_cx": shell_cx + controlled_core_cx,
            "controlled_adjoint_cx": shell_cx + controlled_core_cx,
            "forward_bound_cx": shell_bound_cx + core_bound_cx,
            "controlled_bound_cx": shell_bound_cx + controlled_core_bound_cx,
            "controls": 1, "workspace_qubits": workspace_qubits, "reflections": 0}


def _pauli_preparation(plan: BlockEncodingPlan, *, max_bytes, max_work):
    """Return the LCU data and the one PREP circuit of a multiplexed-Pauli plan, whose reversal is PREP^dagger."""
    from nwqlib.subroutines.lcu.core import _build_lcu_preparation_artifact, prepare_lcu_gate_data

    decomposition = plan.decomposition
    if decomposition is None:
        raise ValueError("multiplexed Pauli plan is missing its decomposition")
    if not decomposition.terms:
        raise ValueError(
            "pauli_lcu cannot block-encode the zero matrix (no Pauli terms, alpha would be 0)"
        )
    data = prepare_lcu_gate_data(
        [term.coefficient for term in decomposition.terms],
        system_dimension=decomposition.operator_dimension,
        coefficient_atol=decomposition.atol,
        max_bytes=max_bytes, max_work=max_work,
    )
    return data, _build_lcu_preparation_artifact(data, max_bytes=max_bytes, max_work=max_work)


def block_encoding_query_cx(plan: BlockEncodingPlan, *, num_controls: int = 0) -> int | None:
    """Return the CX count of one query of a planned encoding, with optional controls, without building it.

    For `"multiplexed_pauli"` and `"banded"` plans the count is that of the
    circuit that
    [`build_block_encoding_query`][nwqlib.subroutines.block_encoding.build_block_encoding_query]
    builds: the uncontrolled PREP pair (and, for banded, the QFT pair) plus
    the SELECT core with `num_controls` controls. The adjoint reverses the
    same gates and has the same count. Multiplexor tables, diagonals and
    the QFT pair are counted exactly for Qiskit 2.6.0's constructions. A
    PREP tree is counted by its slot bound, which a uniform or basis-state
    PREP can undercut, and an X gate with c >= 1 controls by Qiskit's CX
    count without ancillas, which a lowering that borrows idle qubits of the
    surrounding circuit can undercut; the result is then an upper bound.
    The plan's `detail["query_census"]` separates these bound parts for one
    control. The count is also unknown, and the function returns None, when
    the query needs an X with more than 64 controls, for which no Qiskit
    count is tabulated. For other plans, such as a dense dilation, the
    controlled cost of the opaque fallback is unknown and the function
    returns None.

    Args:
        plan (BlockEncodingPlan): A plan from `plan_block_encoding`.
        num_controls (int): Default `0`. Number of external control qubits.

    Returns:
        cx (int | None): The CX count, or None when unknown.

    Raises:
        ValueError: For a negative `num_controls` or a plan whose stored
            bands are invalid.
    """
    k = integer(num_controls, "num_controls", 0)
    try:
        if plan.implementation == "multiplexed_pauli":
            return int(plan.detail["prep_pair_cx"]) + _pauli_core_cx(plan.detail, k)
        if plan.implementation == "banded":
            from nwqlib.subroutines.block_encoding.banded import _band_layout, _band_query_cx

            return _band_query_cx(_band_layout(plan.source), k)
    except _UnknownMCXCount:
        return None
    return None


def build_block_encoding_query(
    source: Any,
    *,
    num_controls: int = 0,
    ctrl_state: int | str | None = None,
    adjoint: bool = False,
    max_bytes: int | None = None,
    max_work: int | None = None,
):
    r"""Build one query U, U^dagger, C(U) or C(U^dagger) of a block encoding as a gate.

    A `"multiplexed_pauli"` or `"banded"` plan is built as
    `U = B^dagger V B`, where the shell B is the coefficient preparation
    PREP (and, for banded, the QFT on the system register) and the core V
    is SELECT. Only the core is controlled:

        \[C_g(U)=(I\otimes B^\dagger)\,C_g(V)\,(I\otimes B),\]

    because `B^dagger B = I` on the inactive control branch. The controlled
    core is built directly as multiplexor tables over the control and
    address qubits that equal the identity unless the controls hold
    `ctrl_state`, so every global phase of V becomes the correct relative
    phase. No PREP or QFT gate is controlled. The adjoint is the reversed
    circuit of the same gates, so the inverse branch uses the actual
    adjoint and no separately rounded angle. A plan whose
    `full_unitary_involution` is True has `U^dagger = U` for the circuit it
    builds: the sign-folded SELECT of real Pauli coefficients and the
    paired band SELECT are direct sums of exact Hermitian involutions
    between a shell and its exact reversal. Its adjoint query is therefore
    the forward query, built and checked as such. Any other plan, such as a
    dense dilation, or a built `BlockEncoding`, such as a supplied circuit,
    is an opaque fallback: the whole circuit is controlled gate by gate with
    `nwqlib.subroutines.qiskit_compat.controlled`, and
    [`block_encoding_query_cx`][nwqlib.subroutines.block_encoding.block_encoding_query_cx]
    reports its cost as unknown. Before it is controlled, the exact
    synthesis of each dense unitary in it and Qiskit's control of the
    synthesized gates are checked against `max_work` and `max_bytes`, by the
    same sizes that the library's circuit lowering and QLS planning use. The
    adjoint of a controlled query reverses the controlled gate while that
    gate still exists, so it is checked as two such constructions held
    together.

    The limits apply to each of the following construction stages on its
    own, before the stage starts, and not to the sum of the stages:

    - building the circuit of a dense-dilation plan, as
      `build_block_encoding_from_plan` does,
    - building a Pauli or banded query, its gate and its adjoint,
    - turning the circuit of the opaque fallback into a gate and reversing it,
    - the syntheses and controls of the opaque fallback.

    Each call with a dense-dilation plan builds its circuit again, including
    the SVD of the completion. A built `BlockEncoding` reuses one completion
    across queries.

    Each stage is checked against a coarse
    estimate of its dominant arrays. For a Pauli or banded plan with n
    system qubits, a ancillas, k controls and m Pauli terms or bands, the
    query needs `H + (4 + 2A) E` bytes, where
    `E = 64 * 2**(a + k) * (n + 16)` is one copy of the query tables,
    `H = m (n + 16)` the terms or bands of the plan, plus the matrix that a
    plan made from a dense matrix keeps (`16 * 4**n` bytes), and A = 1 for the
    adjoint of a plan that is not an involution: the circuit and its gate
    hold one copy each, and that adjoint adds the expanded multiplexors and
    their reversed matrices. The opaque
    fallback counts the arrays its plan keeps and `(5 + 3A) s` bytes for
    every dense matrix of its circuit, `s = 16 * 4**m` for an m-qubit
    unitary instruction: the circuit's stored copies, the copies that the
    conversion to a gate makes and, for an uncontrolled adjoint, the
    reversed copies. Qiskit's synthesis workspace, Python and gate objects
    and the memory of the process are not counted, and no work is charged
    for the reversal, except in the controlled case above.

    The qubits of the returned gate are the `num_controls` controls, then the
    encoding ancillas, then the system qubits, the order Qiskit's
    `Gate.control` uses.

    Args:
        source (BlockEncodingPlan | BlockEncoding): The encoding to query.
        num_controls (int): Default `0`. Number of external control qubits.
        ctrl_state (int | str | None): Default `None`, all controls one.
            Control value in Qiskit's little-endian convention.
        adjoint (bool): Default `False`. Build the adjoint query.
        max_bytes (int | None): Default `None`, the physical memory. Limit
            on the known array bytes of each construction stage described
            above, including the coefficient-preparation data of a Pauli plan.
        max_work (int | None): Default `None`, no limit. Limit on the
            construction work of each stage, as for
            `build_block_encoding_from_plan`, and on the syntheses and
            controls of the opaque fallback.

    Returns:
        gate (qiskit.circuit.Gate): The query on
            `num_controls + num_ancillas + system_qubits` qubits.

    Raises:
        TypeError: If `source` is neither a plan nor a `BlockEncoding`.
        ValueError: For a negative `num_controls`, a `ctrl_state` that does
            not fit the controls, or a limit the construction exceeds.
    """
    from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate

    k = integer(num_controls, "num_controls", 0)
    if isinstance(source, BlockEncodingPlan) and source.implementation in ("multiplexed_pauli", "banded"):
        adjoint = adjoint and source.detail.get("full_unitary_involution") is not True
        _native_query_requirements(source, num_controls=k, adjoint=adjoint, max_bytes=max_bytes, max_work=max_work)
        circuit = _native_query_circuit(source, num_controls=k, ctrl_state=ctrl_state,
                                        max_bytes=max_bytes, max_work=max_work)
        gate = circuit.to_gate(label=circuit.name)
    else:
        held = 0
        if isinstance(source, BlockEncodingPlan):
            # The plan keeps its source matrix (all of the array that a view of it keeps alive) and
            # any kept Pauli terms, m(n + 16) bytes of labels and coefficients, alive while the gate
            # is made.
            held = _backing_array(source.source).nbytes if isinstance(source.source, np.ndarray) else 0
            if source.decomposition is not None:
                held += len(source.decomposition.terms) * (source.system_qubits + 16)
            source = _build_block_encoding_from_plan(source, max_bytes=max_bytes, max_work=max_work)
        elif not isinstance(source, BlockEncoding):
            raise TypeError("build_block_encoding_query requires a BlockEncodingPlan or a BlockEncoding")
        _opaque_query_requirements(source, num_controls=k, adjoint=adjoint, max_bytes=max_bytes, max_work=max_work,
                                   held=held)
        gate = source.circuit.to_gate(label=source.implementation)
        if k:
            gate = controlled(gate, k, ctrl_state=_control_state(ctrl_state, k))
        else:
            _control_state(ctrl_state, 0)
    return inverse_realized_gate(gate, native_ucg=False) if adjoint else gate


def _backing_array(array: np.ndarray) -> np.ndarray:
    """Return the ndarray whose allocation ``array`` keeps alive: ``array`` itself, or the base of a view.

    Its ``nbytes`` is the retained storage, so a small view of a large
    array counts the large array, and two views of one array share one
    count when their callers deduplicate by the returned object. ``.base``
    is followed while it is an ndarray; the extent of any other owner
    beyond that array, such as a file mapping, is not counted.
    """
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


def _native_query_requirements(plan: BlockEncodingPlan, *, num_controls: int, adjoint: bool, max_bytes,
                               max_work) -> tuple[int, int]:
    """Admit one query of a Pauli or banded plan and return its ``(bytes, work)`` before anything is built.

    The bytes are the envelope of ``_admit_native_query``. The work is that
    of the ``2**(a + k)`` query tables, ``32 max(1, n)**2`` per entry, which
    is checked against ``max_work`` before the power is formed, plus, for a
    multiplexed Pauli plan, the PREP stage of its coefficients
    (``lcu.data._lcu_prep_stage_requirements``), which the PREP builder
    checks itself. ``build_block_encoding_query`` admits each such query
    here, and planning that knows the query's controls, such as LCHS
    (``algorithms.lchs.parameters._coherent_construction_work``), reads the
    same numbers with ``max_work=None``. ``adjoint`` is the query actually
    built, so it is False for an involution.
    """
    data = _admit_native_query(plan, num_controls=num_controls, adjoint=adjoint, max_bytes=max_bytes)
    n, a = plan.system_qubits, plan.num_ancillas
    unit = 32 * max(1, n)**2
    work = unit * _admitted_power(a + num_controls, 64 * (n + 16), unit, max_bytes, max_work,
                                  "block-encoding query tables")
    if plan.implementation == "multiplexed_pauli":
        from nwqlib.subroutines.lcu.data import _lcu_prep_stage_requirements
        width = max(len(plan.decomposition.terms) - 1, 0).bit_length()
        work += _lcu_prep_stage_requirements(1 << width, width)[1]
    return data, work


def _admit_native_query(plan: BlockEncodingPlan, *, num_controls: int, adjoint: bool, max_bytes) -> int:
    r"""Refuse a Pauli or banded query whose dominant arrays exceed ``max_bytes``, from plan metadata only, and return its bytes.

    With n system qubits, a ancillas, k controls, ``Q = 2**(a + k)``, m
    stored terms or bands and ``A = 1`` for an adjoint that is reversed (not
    for an involution, whose adjoint query is the forward one), the estimate is

        \[B_{\rm native}=H+(4+2A)E,\qquad E=64Q(n+16),\qquad H=m(n+16)+S.\]

    E is one copy of the query tables, one 2x2 complex128 matrix per
    expanded table entry and system qubit plus 16 such matrices per entry
    for construction temporaries. The native circuit and the gate that
    ``circuit.to_gate()`` copies from it hold two copies, and an adjoint adds
    the expanded one-qubit matrices of each multiplexor and their reversal.
    The other 2E covers the copies Qiskit keeps of those one-qubit matrices.
    H counts the labels and coefficients or band values of the stored plan,
    which holds no table, and S is the allocation of the dense matrix
    that a plan made from a matrix keeps as ``plan.source``
    (``_backing_array``). The estimate is coarse and does not count Qiskit's
    synthesis workspace or its gate objects. The test never forms
    ``2**(a + k)``, and an admitted query returns B.
    """
    n, a = plan.system_qubits, plan.num_ancillas
    stored =len(plan.decomposition.terms) if plan.implementation == "multiplexed_pauli" else plan.detail["num_bands"]
    held = stored * (n + 16)
    if isinstance(plan.source, np.ndarray):
        held += _backing_array(plan.source).nbytes
    per_entry = (6 if adjoint else 4) * 64 * (n + 16)
    _check_bytes(held, max_bytes, "block-encoding query plan")
    limit = memory_limit(max_bytes)
    bits = a + num_controls
    if not fits_power(bits, per_entry, None if limit is None else limit - held):
        source = f"max_bytes={limit}" if max_bytes is not None else f"the physical memory ({limit} bytes)"
        raise ResourceLimitError(f"block-encoding query needs {held} + {per_entry} * 2**{bits} bytes, exceeding {source}")
    return held + (per_entry << bits)


def _opaque_query_requirements(encoding: BlockEncoding, *, num_controls: int, adjoint: bool, max_bytes, max_work,
                               held: int = 0, built: bool = False) -> tuple[int, int, str | None]:
    """Admit the opaque fallback query of a built encoding and return its ``(bytes, work, unresolved)``.

    The gate copies hold ``held`` plus 5, or 8 for an uncontrolled adjoint,
    times the dense matrices of the circuit's ``unitary`` instructions and
    ``UCGate`` tables (``_opaque_matrix_bytes``), read from instruction names,
    widths and table lengths alone. With k controls,
    ``qiskit_compat.controlled`` synthesizes each distinct dense unitary
    exactly and Qiskit controls every occurrence of the synthesized gates, as
    blocks/lowering.py charges a controlled transform. The adjoint reverses
    that controlled gate while it is alive, so its charges are held twice, as
    QLS host planning holds its forward and adjoint controlled queries
    together. The charges come from one walk of the circuit
    (``qiskit_compat.dense_gatewise_charges``).

    ``build_block_encoding_query`` admits each opaque query here before it
    synthesizes anything. Reading a definition may make Qiskit build it, and
    NWQLib synthesizes nothing before the check. Planning that knows the
    query's controls passes ``built`` and ``max_work=None``
    (``joint._component_query_requirements``): the walk reads only
    definitions that are already built, and an operation whose definition
    is not built leaves its part of the synthesis unknown. Then
    ``unresolved`` names that operation, and the bytes and work are those
    of the known part, the gate copies and the syntheses outside such
    definitions. The construction makes all of these too, so their bytes
    are checked here, but the work is not the query's total. Otherwise
    ``unresolved`` is None.
    """
    from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses
    from nwqlib.subroutines.qiskit_compat import dense_gatewise_charges, unbuilt_definition

    copies = 8 if adjoint and not num_controls else 5
    data = held + copies * _opaque_matrix_bytes(encoding.circuit)
    _check_bytes(data, max_bytes, "opaque block-encoding query gate copies")
    if not num_controls:
        return data, 0, None
    widths, counts, unbuilt = dense_gatewise_charges(encoding.circuit, num_controls, built=built)
    copies = 2 if adjoint else 1
    synthesis, work = admit_dense_syntheses(copies * widths, controls=copies * (counts,),
                                            max_work=optional_limit(max_work, "max_work"), max_bytes=max_bytes,
                                            operation="opaque block-encoding query synthesis")
    reason = None if unbuilt is None else f"controlled query synthesis of {unbuilt_definition(unbuilt)}"
    return max(data, synthesis), work, reason


def _opaque_matrix_bytes(circuit) -> int:
    """Return a coarse size of the dense matrices that the top-level instructions of ``circuit`` store.

    Each m-qubit ``unitary`` instruction counts ``16 * 4**m`` bytes, and each
    ``UCGate`` (instruction name ``multiplexer``) counts 64 bytes for every
    2x2 matrix of its table. The table is the one the gate keeps, after
    Qiskit's ``mux_simp`` simplification or NWQLib's padding
    (``_multiplexors.append_uniformly_controlled_unitaries``). Only names,
    qubit counts and table lengths are read, so no gate is built and no
    matrix entry is read. Another instruction named ``multiplexer`` holds no
    such table and counts nothing. Matrices inside other gates' definitions
    are not counted.
    """
    from qiskit.circuit.library import UCGate

    total = 0
    for item in circuit.data:
        if item.name == "unitary":
            total += 16 << (2 * len(item.qubits))
        elif item.name == "multiplexer" and isinstance(item.operation, UCGate):
            total += 64 * len(item.operation.params)
    return total


def _native_query_circuit(plan: BlockEncodingPlan, *, num_controls: int, ctrl_state, max_bytes, max_work):
    """Return PREP-shell, controlled core and PREP^dagger of a native plan on [controls, ancillas, system].

    ``build_block_encoding_query`` has already admitted the arrays of the
    whole query, gate and adjoint, and the work of its ``2**(a + k)``
    tables (``_native_query_requirements``).
    """
    from qiskit import QuantumCircuit

    ctrl_state = _control_state(ctrl_state, num_controls)
    if plan.implementation == "banded":
        from nwqlib.subroutines.block_encoding.banded import _band_circuit, _band_layout

        return _band_circuit(_band_layout(plan.source), num_controls=num_controls, ctrl_state=ctrl_state)
    data, preparation = _pauli_preparation(plan, max_bytes=max_bytes, max_work=max_work)
    core = _pauli_select(plan.decomposition, plan.detail, num_controls=num_controls, ctrl_state=ctrl_state)
    circuit = QuantumCircuit(*core.qregs, name="c_multiplexed_pauli" if num_controls else "multiplexed_pauli")
    address = circuit.qubits[num_controls:num_controls + data.num_control_qubits]
    circuit.compose(preparation.circuit, qubits=address, inplace=True)
    circuit.compose(core, qubits=circuit.qubits, inplace=True)
    circuit.compose(preparation.circuit.inverse(), qubits=address, inplace=True)
    return circuit


def _admitted_power(bits: int, bytes_per_entry: int, work_per_entry: int, max_bytes, max_work, operation: str) -> int:
    """Return ``2**bits`` after admitting ``bytes_per_entry`` data bytes and ``work_per_entry`` work units per entry.

    ``max_bytes=None`` uses the physical memory and ``max_work=None`` sets
    no work limit. A refused request never forms the power.
    """
    _check_bytes(0, max_bytes)
    work = optional_limit(max_work, "max_work")
    limit = memory_limit(max_bytes)
    if not fits_power(bits, bytes_per_entry, limit):
        source = f"max_bytes={limit}" if max_bytes is not None else f"the physical memory ({limit} bytes)"
        raise ResourceLimitError(f"{operation} needs {bytes_per_entry} * 2**{bits} bytes, exceeding {source}")
    if not fits_power(bits, work_per_entry, work):
        raise ResourceLimitError(f"{operation} needs {work_per_entry} * 2**{bits} work units, "
                                 f"exceeding max_work={max_work}")
    return 1 << bits


def _control_state(ctrl_state: int | str | None, num_controls: int) -> int:
    """Return the integer control state, all ones for None, in Qiskit's little-endian convention."""
    try:
        state = (1 << num_controls) - 1 if ctrl_state is None else (
            int(ctrl_state, 2) if isinstance(ctrl_state, str) else integer(ctrl_state, "ctrl_state", 0))
    except ValueError:
        raise ValueError(f"control state {ctrl_state!r} does not fit {num_controls} controls") from None
    if state >= 1 << num_controls:
        raise ValueError(f"control state {ctrl_state!r} does not fit {num_controls} controls")
    return state


def _controlled_table(values, address_qubits: int, num_controls: int, state: int, idle) -> list:
    """Return a multiplexor table over k external controls and a address bits, ``idle`` unless the controls hold ``state``.

    The external controls are the low-order k bits of the table index and
    the address the high-order a bits, the qubit order
    ``build_block_encoding_query`` uses. With k = 0 the padded table itself.
    """
    padded = list(values) + [idle] * ((1 << address_qubits) - len(values))
    if not num_controls:
        return padded
    mask = (1 << num_controls) - 1
    return [padded[index >> num_controls] if index & mask == state else idle
            for index in range(1 << (num_controls + address_qubits))]


def _controlled_projection(projection, num_controls: int, state: int):
    """Return the projected UCG table controlled on k external low-order bits.

    The controlled table is ``U_p`` when the controls hold ``state`` and I
    otherwise, which depends on every control bit unless all ``U_p`` are I
    (then no gate is appended in either case). Its support is the k control
    bits followed by the original support shifted by k.
    """
    from nwqlib.subroutines._multiplexors import ProjectedUnitaryTable

    identity = np.eye(2, dtype=complex)
    if not num_controls or (not projection.support and np.array_equal(projection.projected_table[0], identity)):
        return projection
    mask = (1 << num_controls) - 1
    table = tuple(projection.projected_table[index >> num_controls] if index & mask == state else identity
                  for index in range(1 << (num_controls + len(projection.support))))
    return ProjectedUnitaryTable(support=tuple(range(num_controls)) + tuple(num_controls + bit for bit in projection.support),
                                 projected_table=table)


class _UnknownMCXCount(ValueError):
    """No Qiskit CX count is tabulated for an X with this many controls."""


def _mcx_cx(controls: int) -> int:
    """Return the CX of an X with ``controls`` controls (0 for a plain X), from Qiskit's count without ancillas.

    Open controls add only one-qubit X gates (``_mcx_counts``). Beyond the
    64 tabulated controls the count is unknown, and ``_UnknownMCXCount``
    refuses it. The banded and Pauli query census, and the node census, let
    that refusal stand; only ``block_encoding_query_cx``, whose result may be
    unknown, turns it into None. Callers count an X only when the circuit
    has one, so a control count is never asked for an absent gate.
    """
    from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS

    if controls > len(MCX_CX_BY_CONTROLS):
        raise _UnknownMCXCount(f"no CX count for an X with {controls} controls")
    return MCX_CX_BY_CONTROLS[controls - 1] if controls else 0


def _append_mcx(circuit, controls, state: int, target) -> None:
    """Append an X on ``target`` controlled by ``controls`` on ``state``, or a plain X without controls."""
    from qiskit.circuit.library import MCXGate

    if controls:
        circuit.append(MCXGate(len(controls), ctrl_state=state), [*controls, target])
    else:
        circuit.x(target)


def _append_phase_flip(circuit, qubits, state: int, spare) -> None:
    """Multiply the basis state ``state`` of ``qubits`` by exactly -1.

    On the first qubit t, ``H X H = Z`` gives the phase flip when the other
    qubits hold their part of ``state``, and X gates around it select
    ``t = 0``. With no qubits the flip is the global -1, applied as
    ``(Z X)**2 = -I`` on ``spare``. These gates have no rotation angle, so
    the sign is exactly -1 in the ideal circuit, unlike ``exp(i pi)`` from a
    rounded angle, and it stays a relative phase when controls are added
    to ``qubits``. The cost is one X with ``len(qubits) - 1`` controls. A
    circuit without any qubit, given ``spare=None``, can hold the -1 only as
    its global phase pi. Its controlled form is built with the controls in
    ``qubits``, so the flip on the control value stays exact.
    """
    if not qubits:
        if spare is None:
            circuit.global_phase += np.pi
            return
        for _ in range(2):
            circuit.z(spare)
            circuit.x(spare)
        return
    target, controls = qubits[0], list(qubits[1:])
    if not state & 1:
        circuit.x(target)
    circuit.h(target)
    _append_mcx(circuit, controls, state >> 1, target)
    circuit.h(target)
    if not state & 1:
        circuit.x(target)


def _pauli_select(decomposition, detail, *, num_controls: int = 0, ctrl_state: int = 0):
    """Return the Pauli SELECT on [external controls, address, system], controlled on ``ctrl_state``.

    With k = 0 it is the SELECT of ``_pauli_lcu_encoding``: the coefficient
    phase diagonal on the address register, then one projected UCG per
    system qubit with the sign of each real coefficient in the table of
    ``sign_qubit``, or, without system qubits, the exact phase flips of
    ``sign_labels``. With k external controls every table becomes a table over
    the controls and the address that is I unless the controls hold
    ``ctrl_state`` (``_controlled_table``, ``_controlled_projection``), which
    is exactly the controlled SELECT. PREP stays outside this core.
    """
    from qiskit import QuantumCircuit, QuantumRegister

    address_qubits = max(len(decomposition.terms) - 1, 0).bit_length()
    registers = [QuantumRegister(num_controls, "control")] if num_controls else []
    if address_qubits:
        registers.append(QuantumRegister(address_qubits, "lcu_control"))
    system = QuantumRegister(decomposition.num_qubits, "system")
    select = QuantumCircuit(*registers, system, name="multiplexed_pauli_select")
    selectors = list(select.qubits[:num_controls + address_qubits])
    if detail["coefficient_phase_diagonal"]:
        phases = [float(np.angle(term.coefficient)) for term in decomposition.terms]
        append_control_diagonal_phases(select, selectors,
                                       _controlled_table(phases, address_qubits, num_controls, ctrl_state, 0.0))
    for label in detail["sign_labels"]:
        _append_phase_flip(select, selectors, ctrl_state | (label << num_controls), None)
    labels = [term.label for term in decomposition.terms]
    sign_qubit = detail["sign_qubit"]
    negative = ([complex(term.coefficient).real < 0.0 for term in decomposition.terms]
                if sign_qubit is not None else None)
    # The projected address supports planned by _pauli_plan_detail.
    supports = detail["dependency_supports"]
    for system_index, target in enumerate(system):
        table = project_local_pauli_table(labels, system_index, tuple(supports[system_index]),
                                          negative if system_index == sign_qubit else None)
        append_projected_unitary_table(select, target, selectors,
                                       _controlled_projection(table, num_controls, ctrl_state))
    return select


def _pauli_lcu_encoding(
    plan: BlockEncodingPlan,
    *, max_bytes=None, max_work=None,
) -> BlockEncoding:
    """Build the multiplexed-Pauli block encoding (PREP, dependency-projected SELECT, PREP adjoint) of ``plan``.

    For ``A = sum_j c_j P_j`` with ``L`` terms, PREP prepares
    ``sum_j sqrt(|c_j| / alpha) |j>`` on ``ceil(log2 L)`` address qubits and
    SELECT applies ``exp(i arg c_j) P_j`` for address ``j``. The all-zero
    address block is then ``A / alpha`` with ``alpha = sum_j |c_j|`` (Low and
    Chuang, arXiv:1610.06546v3, Lemma 5). SELECT applies complex coefficient
    phases as one diagonal on the address register and then one exact UCG
    per system qubit (``_pauli_select``). Real coefficients keep their signs
    as the exact factor -1 in one qubit's table, which makes the circuit an
    involution (``_pauli_plan_detail``). ``term.label[-1 - q]`` is the factor on system qubit
    ``q`` because Qiskit labels list qubit 0 last. Padded addresses have zero
    PREP amplitude, identity factors and zero phase.
    """
    from nwqlib.subroutines.lcu.core import _assemble_prepared_lcu
    from qiskit import QuantumCircuit

    data, preparation = _pauli_preparation(plan, max_bytes=max_bytes, max_work=max_work)
    select = _pauli_select(plan.decomposition, plan.detail)
    control_qubits = list(select.qubits[:data.num_control_qubits])
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
    metadata["pauli_term_count"] = len(plan.decomposition.terms)
    metadata["lcu"] = lcu.to_dict()
    # The plan detail carries the structural facts of _pauli_structure_facts.
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
    ``K`` shares the singular vectors of ``A/alpha``, and in that basis each
    2x2 block row is ``[s_i, sqrt(1 - s_i^2)]`` and its partner, which are
    orthonormal, so ``U`` is exactly unitary in exact arithmetic. With
    computed factors it departs from unitarity by the rounding of the
    singular frames and of the two products, and no runtime product
    ``U U^dagger`` is formed to measure it. The ``error_bound`` bounds the
    normalization residual ``A - alpha * (A / alpha)`` in the supplied
    operator's units and does not include circuit-synthesis roundoff.

    The caller that supplies the factors is responsible for their being an
    SVD of this matrix. QLS, for example, extends the SVD of its unpadded
    matrix with identity singular vectors and singular value ``alpha`` on the padding
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
    # K and leaves a deviation from unitarity of at most s**2 - 1, about
    # twice that relative slack.
    complement = (left * np.sqrt(np.clip(1.0 - scaled_singular**2, 0.0, None))) @ right_h

    # Ancilla is circuit qubit 0 (low-order bit): index = (s << 1) | a.
    full = np.zeros((2 * dimension, 2 * dimension), dtype=complex)
    full[0::2, 0::2] = scaled
    full[1::2, 0::2] = complement
    full[0::2, 1::2] = complement
    full[1::2, 1::2] = -scaled

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
    metadata.update(_dense_structure_facts(array, scaled))
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


def _dense_structure_facts(array, scaled) -> dict[str, bool | None]:
    """Return the structural facts of the dense dilation of ``array`` with block ``scaled = A/alpha``.

    The construction owns the dense input and its normalized copy, so their
    stored entries are compared with O(d) row workspace, without making an
    adjoint matrix or expanding a structured operator or circuit. Real scalar
    normalization preserves a proved Hermitian input, including signed zero.
    The SVD dilation has no involution proof (an exactly Hermitian
    nonsingular input can give one, but the construction does not establish
    it), so that fact is unknown unless a non-Hermitian block disproves it.
    """
    target = _stored_dense_is_hermitian(array)
    encoded = True if target else _stored_dense_is_hermitian(scaled)
    return {"target_operator_is_hermitian": target, "encoded_operator_is_hermitian": encoded,
            "full_unitary_involution": False if encoded is False else None}


def _stored_dense_is_hermitian(array):
    """Return whether the stored entries satisfy ``A[i, j] == conj(A[j, i])`` exactly, in O(d²) time and O(d) workspace."""
    return all(np.array_equal(array[row, row:], array[row:, row].conj())
               for row in range(len(array)))


def _dense_dilation_predicted_cx(system_qubits: int) -> float:
    """Return the paper's QSD CX count of the dense dilation, the estimate that the automatic choice compares.

    The count is ``(23/48) 4**m - (3/2) 2**m + 4/3`` at width ``m = n + 1``,
    which includes the dilation ancilla. It is Eq. (19) (p. 16) and the row
    "QSD (l = 2, optimized)" of Table 1 (p. 14) in Shende et al.,
    quant-ph/0406176v5. That count comes from the quantum Shannon
    decomposition (QSD, Theorem 13) recursed down to two-qubit blocks with the
    optimizations of their Appendix A. The automatic choice uses this paper
    count for the dense construction. A dilation compiled to gates is
    synthesized by
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
    """Return the per-query CX count of the Pauli encoding from its ``_pauli_plan_detail`` gate counts.

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
    """Sum Pauli phases over a cyclic shift by a dynamic program over the bits with two carry states.

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


def _shift_pauli_trace_table(*, num_qubits, offsets, flips, phases):
    """Return ``_shift_pauli_trace_sum`` for every term (row) and offset (column) as an int64 array.

    The same carry dynamic program runs on whole arrays, row ``t`` with flip
    mask ``flips[t]`` and phase mask ``phases[t]`` and column ``j`` with offset
    ``offsets[j]``. Every entry has magnitude at most ``2**n``, so int64 is
    exact for the ``n <= 62`` qubits that the caller passes.
    """
    offsets = np.asarray(offsets, dtype=np.int64)[None, :]
    flips = np.asarray(flips, dtype=np.int64)[:, None]
    phases = np.asarray(phases, dtype=np.int64)[:, None]
    shape = (flips.shape[0], offsets.shape[1])
    states = [np.ones(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)]
    for qubit in range(num_qubits):
        offset_bit = (offsets >> qubit) & 1
        flip_bit = (flips >> qubit) & 1
        phase_bit = (phases >> qubit) & 1
        next_states = [np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)]
        for carry, subtotal in enumerate(states):
            for input_bit in (0, 1):
                total = input_bit + offset_bit + carry
                output_bit = total & 1
                signed = np.where(phase_bit & output_bit, -subtotal, subtotal)
                signed = np.where((input_bit ^ output_bit) == flip_bit, signed, 0)
                next_states[0] += np.where(total >> 1 == 0, signed, 0)
                next_states[1] += np.where(total >> 1 == 1, signed, 0)
        states = next_states
    return states[0] + states[1]


# Rows of the first chunk of the circulant check. A small first chunk ends
# the detection of a non-circulant input after few trace evaluations, and
# the doubling keeps the number of chunks logarithmic. The value changes
# only the order of the checks, not their outcome.
_FIRST_CIRCULANT_ROWS = 16


def _circulant_relative_tolerance(num_qubits, terms, bands):
    """Return the relative window of the circulant check of ``terms`` Pauli terms on ``num_qubits`` with ``bands`` bands.

    Untuned relative window for binary64 coefficients against their exact
    rational prediction. It grows with the qubit, term and band counts, and a
    mismatch above it rejects the circulant candidate. The exact residual is
    computed separately, and automatic routing keeps a circulant whose
    residual is at most this window times ``||A||_F``, a roundoff-level
    difference. Registered in docs/ENGINEERING_CONSTANTS.md. Revisit with a
    changed accumulation or classification algorithm.
    """
    return 64.0 * np.finfo(float).eps * max(1, num_qubits, terms, bands)


def _detect_banded_pauli_structure(
    decomposition: PauliDecomposition,
) -> tuple[Any, float]:
    """Certify a circulant from kept Pauli terms in polynomial work.

    With ``P = i**y X**f Z**z``, ``P|0> = i**y |f>``, so the candidate band at
    offset ``f`` is ``beta_f = A[f, 0] = sum_{P with flip mask f} c_P i**y``.
    The candidate circulant ``C = sum_f beta_f S**f`` has Pauli coefficients
    ``c_P(C) = 2**-n sum_f beta_f Tr(P^dagger S**f)``, evaluated exactly from
    the integer traces of ``_shift_pauli_trace_sum`` (``_shift_pauli_trace_table``).
    Every binary64 number is an integer multiple of ``2**-1074``, so the
    band and supplied coefficients are integers at one common binary scale
    and the exact rational arithmetic is integer arithmetic. Only the
    nonzero traces are accumulated. Every supplied coefficient must match its
    prediction within the relative window below, taken of the largest of the
    two and the largest supplied magnitude. The terms are predicted and
    checked in chunks, the first of ``_FIRST_CIRCULANT_ROWS`` rows and each
    later one twice the previous up to about 2**20 trace entries, so the
    first mismatching chunk ends the detection before later traces are
    evaluated.
    Pauli strings are orthogonal, so ``||M||_F**2 = 2**n sum_P |c_P(M)|**2``
    and ``sum_P |c_P(C)|**2 = sum_f |beta_f|**2``. The squared mismatch on
    supplied labels plus the candidate mass on absent labels is therefore
    ``||A - C||_F**2 / 2**n``, and the returned error evaluates
    ``||A - C||_F``, which bounds the operator-norm difference. The automatic
    choice keeps the circulant when this error is within the roundoff window
    of ``_circulant_relative_tolerance`` (``_detect_circulant``).
    """

    from nwqlib.subroutines.block_encoding.banded import BandSpecification

    dimension = decomposition.operator_dimension
    num_qubits = decomposition.num_qubits
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

    relative_tolerance = _circulant_relative_tolerance(num_qubits, len(parsed_terms), len(band_coefficients))
    # Common binary scale 2**-K of every band and supplied component.
    components = [part for value in band_coefficients.values() for part in (value.real, value.imag)]
    components += [part for term, _ in parsed_terms for part in (term.coefficient.real, term.coefficient.imag)]
    scale_bits = max(float(part).as_integer_ratio()[1].bit_length() - 1 for part in components)

    def scaled(value: complex) -> tuple[int, int]:
        pairs = (float(value.real).as_integer_ratio(), float(value.imag).as_integer_ratio())
        return tuple(numerator << (scale_bits + 1 - denominator.bit_length()) for numerator, denominator in pairs)

    offsets = list(band_coefficients)
    scaled_bands = [scaled(band_coefficients[offset]) for offset in offsets]
    flips = [masks[0] for _, masks in parsed_terms]
    phases = [masks[1] for _, masks in parsed_terms]

    def predicted(first, stop):
        """Return sum_f Tr(P_t^dagger S**f) beta_f * 2**K, an exact complex integer, for terms first..stop-1."""
        numerators = [[0, 0] for _ in range(first, stop)]
        if num_qubits <= 62:
            table = _shift_pauli_trace_table(num_qubits=num_qubits, offsets=offsets,
                                             flips=flips[first:stop], phases=phases[first:stop])
            rows, columns = np.nonzero(table)
            for row, column, trace_sum in zip(rows.tolist(), columns.tolist(), table[rows, columns].tolist()):
                numerators[row][0] += trace_sum * scaled_bands[column][0]
                numerators[row][1] += trace_sum * scaled_bands[column][1]
            return numerators
        for numerator, flip_mask, phase_mask in zip(numerators, flips[first:stop], phases[first:stop], strict=True):
            for (band_real, band_imag), offset in zip(scaled_bands, offsets, strict=True):
                trace_sum = _shift_pauli_trace_sum(num_qubits=num_qubits, offset=offset,
                                                   flip_mask=flip_mask, phase_mask=phase_mask)
                numerator[0] += trace_sum * band_real
                numerator[1] += trace_sum * band_imag
        return numerators

    # c_P(C) = numerator / (2**n 2**K). All masses below are exact integers at the scale (2**n 2**K)**2.
    denominator = dimension << scale_bits
    # A coefficient at roundoff level of the largest one, such as the residue
    # of a decomposition, is compared against that largest magnitude.
    largest = max(abs(term.coefficient) for term, _ in parsed_terms)
    supplied_residual_mass = projected_mass = 0
    rows_per_chunk = max(1, (1 << 20) // len(offsets))
    first, size = 0, min(_FIRST_CIRCULANT_ROWS, rows_per_chunk)
    while first < len(parsed_terms):
        stop = min(first + size, len(parsed_terms))
        for (term, (_flip, _phase, y_count)), (expected_real, expected_imag) in zip(
                parsed_terms[first:stop], predicted(first, stop), strict=True):
            for _ in range(y_count % 4):
                expected_real, expected_imag = -expected_imag, expected_real
            # Integer true division rounds the exact rational to nearest.
            expected = complex(expected_real / denominator, expected_imag / denominator)
            scale = max(abs(term.coefficient), abs(expected), largest)
            if abs(term.coefficient - expected) > relative_tolerance * scale:
                raise ValueError("Pauli decomposition is not a certified circulant operator")
            supplied_real, supplied_imag = scaled(term.coefficient)
            supplied_residual_mass += ((dimension * supplied_real - expected_real) ** 2
                                       + (dimension * supplied_imag - expected_imag) ** 2)
            projected_mass += expected_real**2 + expected_imag**2
        first, size = stop, min(2 * size, rows_per_chunk)
    band_mass = dimension**2 * sum(real**2 + imag**2 for real, imag in scaled_bands)
    missing_mass = band_mass - projected_mass
    if missing_mass < 0:
        raise ValueError("Pauli decomposition is not a certified circulant operator")
    residual_mass = supplied_residual_mass + missing_mass
    if residual_mass == 0:
        detection_error = 0.0
    else:
        residual_mass_upper = float(np.nextafter(residual_mass / denominator**2, np.inf))
        # ||R||_2 <= ||R||_F = sqrt(2**n * residual_mass). Round each
        # floating operation outward, since an upper input alone does
        # not make nearest-rounded sqrt and multiplication upper bounds.
        root_upper = np.nextafter(np.sqrt(residual_mass_upper), np.inf)
        dimension_root_upper = np.nextafter(2.0 ** (num_qubits / 2.0), np.inf)
        detection_error = float(np.nextafter(root_upper * dimension_root_upper, np.inf))
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
            num_qubits=num_qubits,
        ),
        detection_error,
    )


def _admit_dense_input(operator, *, max_bytes, max_work, construction=False, selected_svd=False):
    """Check the dense byte and work formulas against the limits before any conversion, SVD or completion.

    Only the shape, or the row lengths of a nested sequence, is read first,
    so an input above the caller's limits fails before any array is formed.
    The "Dense block-encoding arrays and work" row in
    docs/ENGINEERING_CONSTANTS.md registers these formulas.

    With ``construction=True`` the 256 bytes per entry of the d-square
    matrix count the main arrays of ``_dense_dilation_encoding`` as if they
    were alive together: the complex128 source, ``W``, ``V^dagger``,
    ``A/alpha``, the product ``W sqrt(1 - s**2)`` and the complement ``K``
    (6 x 16 = 96), the 2d-square dilation ``U`` and an allowance for the
    copy that its gate can hold (2 x 64 = 128), and 32 for the unpadded
    singular vectors that the QLS padded completion
    (``algorithms/qls/quantum.py``) keeps beside the padded ones. The
    ``32 d`` bytes are four float64 vectors: the singular values,
    ``s/alpha`` and two temporaries of ``sqrt(1 - s**2)``.

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
    # 256 d**2 counts the completion arrays as if alive together and
    # 192 d**2 is the planning allowance (see the docstring).
    # 32 d: four float64 vectors.
    _check_bytes((256 if construction else 192) * d * d + 32*d, max_bytes,
        "block-encoding dense arrays")
    # Completion work is 8d**3 for the SVD and d**3 for the complement
    # product, 9d**3 in total, or d**3 when the SVD is supplied. Without
    # construction, 16 units per entry cover the intake scan.
    work = (1 if selected_svd else 9) * d**3 if construction else 16 * d * d
    cap = optional_limit(max_work, "max_work")
    if exceeds(work, cap):
        stage = "completion" if construction else "intake scan"
        raise ResourceLimitError(f"block-encoding dense work exceeds max_work: {stage} work {work} for dimension {d}, max_work={cap}")
    return d


def _circulant_classification_work(q, terms, bands):
    """Return the work C(q, m, b), in logical visits, of one circulant classification of m kept terms.

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
    proxies for the limit check, not timings or equal-cost CPU operations.
    """
    if terms == 0:
        return 0
    sort_levels = max(0, (bands - 1).bit_length())
    return ((q + 32) * terms
            + 4 * (q + 1) * terms * bands
            + 16 * bands + 2 * bands * sort_levels + 32)


def _pauli_planning_requirements(n, terms):
    """Return the planning bytes ``16 m w + 2P + 32m`` and work ``m n + 2n (m + aP)`` of ``_admit_pauli_plan``."""
    a = max(terms - 1, 0).bit_length()
    words = max(1, -(-n // 64))
    return 16 * terms * words + 2 * (1 << a) + 32 * terms, terms * n + 2 * n * (terms + a * (1 << a))


def _admit_pauli_plan(decomposition, *, held_bytes, max_bytes, max_work, classify):
    r"""Check the arrays and work of planning a Pauli SELECT, and the optional circulant check, before the work.

    With m terms on n system qubits, ``w = max(1, ceil(n/64))`` mask words
    and ``P = 2**a`` padded addresses, ``_pauli_dependency_supports`` packs
    the X and Z masks into two uint64 arrays of ``16 m w`` bytes. For each
    system qubit, ``local_pauli_dependencies`` forms the P letter codes and
    compares the codes of each address pair that differs only in one bit.
    The workspace of one
    qubit, the codes, one comparison and the extracted mask words, is
    counted at ``2P + 32m`` bytes. The work is ``m n`` for the masks plus
    ``2n (m + aP)`` for the dependency tests of the unsigned and the signed
    tables. ``held_bytes`` is the data that the caller has already checked
    and that stays in memory during this phase, each array counted once.
    Planning forms no SELECT table. The builders check the tables at
    ``64 P (n + 16)`` bytes and ``32 P max(1, n)**2`` work before they form them
    (``_build_block_encoding_from_plan`` and ``_admit_native_query``).

    With ``classify``, the circulant classification adds the work
    ``_circulant_classification_work(n, m, b)`` with b = min(m, 2**n)
    candidate bands. One chunk of its trace check holds at most
    ``max(min(m b, 2**20), b)`` int64 trace entries, counted at 80 bytes
    each for the table and the arrays of the same shape that its carry
    recursion forms. The classification ends before the dependency tests
    start, so the larger of the two stages is checked. Its exact rational
    band and coefficient values are Python objects, not arrays, and are not
    counted.

    Returns:
        The checked work, so the caller can add later stages to it. The
        caller's ``max_work`` (None: no limit) stays the budget of this
        plan.
    """
    n, terms = decomposition.num_qubits, len(decomposition.terms)
    planning_bytes, work = _pauli_planning_requirements(n, terms)
    # An empty decomposition skips classification.
    if classify and terms:
        bands = min(terms, 1 << n)
        work += _circulant_classification_work(n, terms, bands)
        planning_bytes = max(planning_bytes, 80 * max(min(terms * bands, 1 << 20), bands))
    _check_bytes(held_bytes + planning_bytes, max_bytes, "block-encoding Pauli planning arrays")
    if exceeds(work, optional_limit(max_work, "max_work")):
        raise ResourceLimitError(f"block-encoding Pauli planning work {work} exceeds max_work={max_work}")
    return work


def plan_block_encoding(
    operator: Any,
    *,
    implementation: str = "auto",
    normalization: float | None = None,
    max_bytes: int | None = None,
    max_work: int | None = None,
) -> BlockEncodingPlan:
    """Choose a block encoding and compute its `alpha`, width and error bound without building it.

    A circulant goes to `banded` first: a supplied `BandSpecification`, or
    structure that the detector finds within a roundoff window of the input,
    whose detection residual becomes `error_bound`. Otherwise `"auto"`
    compares the per-query CX counts of multiplexed Pauli SELECT and dense
    dilation and selects the smaller. A tie selects the smaller alpha, with
    dense dilation winning when both alpha values are equal. An explicit
    `"banded"` request keeps the detector's default tolerance and records
    the resulting bound as `error_bound`.
    [`build_block_encoding_from_plan`][nwqlib.subroutines.block_encoding.build_block_encoding_from_plan]
    builds the returned plan's construction without comparing the
    constructions again. Pass it the same `max_bytes` and `max_work`.

    A supplied `normalization` becomes `alpha` only when the plan uses
    dense dilation. Without one, dense dilation uses the optimal spectral
    normalization `||A||_2`. Pauli SELECT always uses its coefficient
    1-norm. Arithmetic failures in structural error bounds propagate rather
    than selecting another encoding without the required bound. For a
    `PauliDecomposition`, the target operator is its kept terms. The
    encoding bound does not include loss from an earlier decomposition, so
    the caller must account separately for the error between the original
    operator and its kept terms.
    A dense-dilation plan computes the normalization from the singular
    values alone and keeps no singular vectors. Building it later computes
    a full SVD for the completion, so planning and then building without a
    `normalization` costs one values-only SVD more than
    `build_block_encoding`, about `4 d**3 / 3` counted units over the
    `9 d**3` of the completion. Use `build_block_encoding` when the circuit
    will be built. For a dense dilation, a supplied `normalization` avoids
    the norm computation and becomes `alpha` as given.

    Args:
        operator (array_like | PauliDecomposition | BandSpecification): The
            operator `A`, as for `build_block_encoding`.
        implementation (str): `"auto"` (default), `"pauli_lcu"`,
            `"multiplexed_pauli"`, `"banded"` or `"dense_dilation"`.
        normalization (float | None): `alpha` to use if the plan is a dense
            dilation. Default `None`, which selects `||A||_2`.
        max_bytes (int | None): Default `None`, the physical memory. Limit
            on the array bytes of each planning step, as for
            `build_block_encoding`.
        max_work (int | None): Default `None`, no limit. Limit on the work
            of each planning step, as for `build_block_encoding`.

    Returns:
        plan (BlockEncodingPlan): The chosen construction with its `alpha`,
            `num_ancillas` and `error_bound`.

    Raises:
        ValueError: For unknown implementation names, operator and
            implementation mismatches, a zero operator, a matrix that is not
            square with power-of-two dimension, a `normalization` that is not
            finite and positive or lies below `||A||_2`, a limit that is
            neither None nor a positive integer, or a planning step that
            exceeds `max_bytes` or `max_work`.
        OverflowError: If structural error-bound arithmetic overflows.
        FloatingPointError: If a structural error bound is nonfinite.
    """

    return _plan_block_encoding(operator, implementation=implementation, normalization=normalization,
        max_bytes=max_bytes, max_work=max_work)


def _detect_circulant(operator, implementation):
    """Return ``(specification, detection, detection_error, hermitian_input)`` for the circulant route.

    A supplied ``BandSpecification`` is used as given, with zero error. For
    ``auto`` and ``banded``, a Pauli input is checked by
    ``_detect_banded_pauli_structure``, and ``auto`` keeps the result when its
    error is at most the roundoff window ``_circulant_relative_tolerance``
    times ``||A||_F``. A dense input is scanned with the roundoff tolerance
    ``64 eps max(1, n) max|A|`` under ``auto`` and with the detector's
    default tolerance under an explicit ``banded`` request. Either way a
    nonzero detection error enters ``error_bound``.
    A failed detection under ``auto`` means no circulant. Under an explicit
    ``banded`` request it raises. Other requests skip detection.

    ``hermitian_input`` states that the stored input proves Hermiticity, by
    entrywise equality of a dense matrix or real coefficients of a Pauli sum.
    Such a circulant is replaced by its Hermitian projection, after the
    routing decision above and within its error bound
    (``banded._hermitian_band_projection``). Any other input keeps its
    detected bands, so a non-Hermitian circulant stays unpaired.
    """
    from nwqlib.subroutines.block_encoding.banded import (
        BandSpecification,
        _detect_banded_structure_with_error,
        _hermitian_band_projection,
    )

    specification = None
    detection = None
    detection_error = 0.0
    hermitian_input = False
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
            # A float circulant rarely decomposes to an exactly zero residual, so
            # automatic routing keeps one whose residual is roundoff.
            window = _circulant_relative_tolerance(operator.num_qubits, len(operator.terms),
                                                   len(specification.offsets)) * float(np.sqrt(
                operator.operator_dimension * fsum(abs(term.coefficient) ** 2 for term in operator.terms)))
            if implementation == "auto" and detection_error > window:
                specification = None
            else:
                hermitian_input = all(term.coefficient.imag == 0 for term in operator.terms)
    elif implementation == "banded":
        array = _as_power_of_two_matrix(operator)
        specification, detection_error = _detect_banded_structure_with_error(array)
        detection = "detected_from_dense_matrix"
        hermitian_input = _stored_dense_is_hermitian(array)
    elif implementation == "auto":
        array = _as_power_of_two_matrix(operator)
        try:
            # Roundoff-level entry differences, the dense counterpart of the Pauli window.
            atol = (64.0 * np.finfo(float).eps * max(1, array.shape[0].bit_length() - 1)
                    * float(np.max(np.abs(array))))
            specification, detection_error = _detect_banded_structure_with_error(array, atol=atol)
        except ValueError:
            specification = None
        else:
            detection = "detected_from_dense_matrix"
            hermitian_input = _stored_dense_is_hermitian(array)
    if hermitian_input:
        specification, detection_error = _hermitian_band_projection(specification, detection_error)
    return specification, detection, detection_error, hermitian_input


def _banded_plan(specification, *, implementation, detection, detection_error, hermitian_input,
                 max_bytes, max_work):
    """Check against the limits and describe the shift-LCU encoding of a circulant ``specification``.

    ``alpha`` is the band-coefficient 1-norm and the address register has
    ``ceil(log2 B)`` qubits for B bands. ``periodic_gate_counts`` records the
    gate counts of ``build_banded_block_encoding`` (``banded._band_gate_counts``):
    the QFT pair, one UCRZ table per system qubit (``2**a`` CX each for
    a >= 1 address qubits), an address diagonal of ``2**a - 2`` CX when any
    of its phases is nonzero, a PREP pair of positive-amplitude trees and,
    for a Hermitian specification, the X gates of the pair swap J, the half
    shift and the sign flips. The plan and the builder read the same
    ``banded._band_layout``, so alpha, the phases and the counts agree bit
    for bit. Planning forms Python lists of the B band values and of the
    n x B RZ angles, a few times each, counted as ``8 n B`` work and no array
    bytes. The builder checks its tables before it forms them
    (``_build_block_encoding_from_plan``).
    """
    from nwqlib.subroutines.block_encoding.banded import (
        _band_gate_counts, _band_layout, _band_query_census, _band_structure_facts)

    integer(specification.num_qubits, "num_qubits", 1)
    bands = len(specification.offsets)
    ancillas = max(bands - 1, 0).bit_length()
    _check_bytes(0, max_bytes)
    if exceeds(need := 8 * specification.num_qubits * bands, max_work):
        raise ResourceLimitError(f"block-encoding band planning work exceeds max_work: it needs {need} work units, "
                                 f"max_work={max_work}")
    layout = _band_layout(specification)
    return BlockEncodingPlan(
        requested_implementation=implementation,
        implementation="banded",
        alpha=float(np.sum(np.asarray(layout.magnitudes, dtype=float))),
        num_ancillas=layout.address_qubits,
        system_qubits=specification.num_qubits,
        error_bound=detection_error,
        source=specification,
        decomposition=None,
        detail={
            "num_bands": bands,
            "padded_table_size": 1 << ancillas,
            "structure_detection": detection,
            "periodic_gate_counts": _band_gate_counts(layout),
            **_band_structure_facts(layout, detection_error, hermitian_input=hermitian_input),
            "query_census": _band_query_census(layout),
        },
    )


def _dense_norm_work(dimension, full_svd):
    """Return the work of the spectral norm that ``_plan_block_encoding`` computes.

    A ``dense_norm`` from the caller computes a full SVD of the d-square
    matrix, whose singular vectors the dense completion reuses, counted at
    ``8 d**3`` like every full SVD in the dense work formulas. The default
    ``np.linalg.norm(A, 2)`` computes the singular values alone, counted at
    ``_linalg_laws.singular_values_work``.
    """
    return 8 * dimension**3 if full_svd else singular_values_work(dimension)


def _explicit_dense_dilation_plan(array, *, implementation, normalization, dense_norm,
                                  full_svd, work_used, max_work, system_qubits):
    """Plan the dense dilation that the caller requested explicitly for a dense matrix.

    Without a supplied ``normalization``, ``alpha = ||A||_2`` from
    ``dense_norm``, counted as ``_dense_norm_work(d, full_svd)``. The entrywise
    test ``max|A_ij| <= alpha``, with the rounding slack of
    ``_require_alpha_covers``, rejects some alphas below the spectral norm
    before any construction, since every entry magnitude is at most
    ``||A||_2``. The error bound is that of ``_normalization_error_bound``.
    """
    if normalization is None and exceeds(need := work_used + _dense_norm_work(array.shape[0], full_svd), max_work):
        raise ResourceLimitError(f"block-encoding normalization exceeds max_work: it needs {need} work units, "
                                 f"max_work={max_work}")
    alpha = (
        float(normalization) if normalization is not None else dense_norm(array)
    )
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            "dense-dilation normalization must be finite and positive (zero matrix is unsupported)"
        )
    _require_alpha_covers(alpha, float(np.max(np.abs(array))), "the largest entry magnitude")
    scaled = normalized_matrix(array, alpha)
    return BlockEncodingPlan(
        requested_implementation=implementation,
        implementation="dense_dilation",
        alpha=alpha,
        num_ancillas=1,
        system_qubits=system_qubits,
        error_bound=_normalization_error_bound(array, alpha, scaled),
        source=array,
        decomposition=None,
        detail=_dense_structure_facts(array, scaled),
    )


def _plan_block_encoding(operator, *, implementation, normalization=None,
                         max_bytes, max_work, dense_norm=None, term_count=None, source_matrix=None):
    """Select one encoding for ``operator`` and return its plan without building a circuit.

    A circulant within roundoff routes to ``banded`` (``_detect_circulant`` and
    ``_banded_plan``). An explicit dense dilation of a dense matrix is
    planned by ``_explicit_dense_dilation_plan``. Otherwise a dense matrix is
    first Pauli-transformed with exact-zero removal only, and ``auto``
    compares the Pauli and dense-dilation CX formulas. The PREP pair of m
    Pauli terms alone costs 2*(2**a - 2) CX with a = bit_length(m - 1),
    which exceeds the dense dilation when m > d**2/2, so ``auto`` then
    selects the dense dilation from the term count, without packing the
    words or forming the labels, the terms or their gate counts.
    ``build_block_encoding`` passes ``dense_norm`` so that the SVD computing
    ``||A||_2`` is kept for the completion. Without it the norm comes from
    ``np.linalg.norm(A, 2)``.

    Two private arguments let a caller that already holds the Pauli words of
    its operator (compiled LCHS children) skip repeated work, with the same
    selection, records and error bound. ``term_count`` is the number of
    nonzero Pauli words of a matrix input. When it settles an ``auto`` dense
    choice, the matrix is not transformed again. ``source_matrix`` is the
    matrix that a Pauli input decomposes exactly. When ``auto`` selects the
    dense dilation, it is used instead of reconstructing the matrix from the
    terms.
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
    max_work = optional_limit(max_work, "max_work")
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

    specification, detection, detection_error, hermitian_input = _detect_circulant(operator, implementation)
    if specification is not None and implementation in ("auto", "banded"):
        return _banded_plan(specification, implementation=implementation, detection=detection,
            detection_error=detection_error, hermitian_input=hermitian_input, max_bytes=max_bytes, max_work=max_work)
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

    dense_cost = _dense_dilation_predicted_cx(system_qubits)
    if array is not None:
        d = array.shape[0]
        coefficients = None
        if not (implementation == "auto" and term_count is not None and 2 * direct_preparation_cx_bound(
                max(term_count - 1, 0).bit_length(), complex_phases=False) > dense_cost):
            # The block transform has q levels over d**2 entries (32 units
            # each). Its 64 d**2 bytes cover the source, the two live
            # complex128 levels and the half-sum temporaries.
            work_used += 32 * max(1,system_qubits) * d * d
            if exceeds(work_used, max_work):
                raise ResourceLimitError(
                    f"block-encoding Pauli decomposition exceeds max_work: it needs {work_used} work units, "
                    f"including {32 * max(1, system_qubits) * d * d} for the transform of dimension {d}, "
                    f"max_work={max_work}. Raise max_work to at least {work_used}.")
            _check_bytes(64*d*d, max_bytes, "block-encoding Pauli transform")
            # The range-safe exact-zero block transform. Qiskit's squared
            # magnitude zero test can discard tiny terms even with atol=rtol=0.
            coefficients = _block_transform(array)
            term_count = 0 if coefficients is None else int(np.count_nonzero(coefficients))
        terms = term_count
        prep_pair_cx = 2 * direct_preparation_cx_bound(max(terms - 1, 0).bit_length(), complex_phases=False)
        if implementation == "auto" and prep_pair_cx > dense_cost:
            decomposition = None
        else:
            # _coefficient_masks checks the packed masks of the m nonzero words.
            x, z, values = _coefficient_masks(coefficients, system_qubits, max_bytes=max_bytes)
            decomposition = PauliDecomposition(
                terms=tuple(PauliTerm(label=label, coefficient=value) for label, value in
                    PauliTerms._snapshot(system_qubits, x, z, values).labels(max_bytes=max_bytes)),
                input_dimension=d, operator_dimension=d, num_qubits=system_qubits,
                atol=0.0)
            del x, z, values
        del coefficients
    if decomposition is not None:
        if not decomposition.terms:
            raise ValueError(
                "pauli_lcu cannot block-encode the zero matrix (no Pauli terms, alpha would be 0)"
            )
        if array is not None:
            # B_held: the dense input (16 d**2) and the decomposition's labels and
            # complex coefficients (m(n+16)) stay live while the tables are planned.
            work_used += _admit_pauli_plan(decomposition,
                held_bytes=16 * d * d + len(decomposition.terms) * (system_qubits + 16),
                max_bytes=max_bytes, max_work=max_work, classify=False)
            if exceeds(work_used, max_work):
                raise ResourceLimitError(
                    f"block-encoding Pauli selection exceeds max_work: it needs {work_used} work units for "
                    f"{len(decomposition.terms)} Pauli terms, max_work={max_work}. "
                    f"Raise max_work to at least {work_used}.")
        detail = _pauli_plan_detail(decomposition)
        pauli_alpha = float(fsum(abs(term.coefficient) for term in decomposition.terms))
        pauli_cost = _pauli_predicted_cx(detail)
    else:
        # Every other term of the Pauli CX count is nonnegative, so this
        # PREP pair alone settles the comparison.
        detail = {"term_count": terms, "prep_pair_cx": prep_pair_cx, "query_census": None}
        pauli_alpha = pauli_cost = None
    selected = "dense_dilation" if implementation == "dense_dilation" else "multiplexed_pauli"
    dense_alpha = None if normalization is None else float(normalization)
    if implementation == "auto" and (pauli_cost is None or pauli_cost >= dense_cost):
        selected = "dense_dilation"
    dense_source = array
    # A Pauli input selected for dense dilation is expanded to its d-square
    # matrix (L d**2 work for L terms) while its term table stays alive, and
    # its spectral norm then costs _dense_norm_work unless a normalization is
    # given. A caller's source_matrix is that matrix already.
    if selected == "dense_dilation" and dense_source is None and source_matrix is not None:
        dense_source = np.asarray(source_matrix, dtype=complex)
    if selected == "dense_dilation" and dense_source is None:
        d = decomposition.operator_dimension
        kept_terms = len(decomposition.terms)*(system_qubits+16)
        _check_bytes((384 if full_svd_requested else 192)*d*d+32*d+kept_terms,
            max_bytes, "block-encoding Pauli dense conversion and kept terms")
        work_used += len(decomposition.terms)*d*d
        norm_work = _dense_norm_work(d, full_svd_requested) if dense_alpha is None else 0
        if exceeds(work_used + norm_work, max_work):
            raise ResourceLimitError(f"block-encoding Pauli dense conversion exceeds max_work: it needs "
                                     f"{work_used + norm_work} work units, max_work={max_work}")
        dense_source = np.asarray(decomposition.to_sparse_pauli_op().to_matrix(), dtype=complex)
    if selected == "dense_dilation" and dense_alpha is None:
        assert dense_source is not None
        d = dense_source.shape[0]
        kept_terms = 0 if decomposition is None else len(decomposition.terms)*(system_qubits+16)
        _check_bytes((384 if full_svd_requested else 192)*d*d+32*d+kept_terms,
            max_bytes, "block-encoding normalization and kept terms")
        if exceeds(work_used + _dense_norm_work(d, full_svd_requested), max_work):
            raise ResourceLimitError(f"block-encoding normalization exceeds max_work: it needs "
                                     f"{work_used + _dense_norm_work(d, full_svd_requested)} work units, "
                                     f"max_work={max_work}")
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
        if implementation == "auto" and pauli_cost is not None
        else f"auto selected {selected}: the PREP pair of the {detail['term_count']}-term Pauli encoding "
        f"alone needs {detail['prep_pair_cx']} CX, above dense_dilation={dense_cost:g}"
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
            **({"multiplexed_pauli": pauli_cost} if pauli_cost is not None else {}),
            "dense_dilation": dense_cost,
        },
        "selection_reason": reason,
        "normalization_records": {
            **({"multiplexed_pauli": pauli_alpha} if pauli_alpha is not None else {}),
            **({"dense_dilation": dense_alpha} if dense_alpha is not None else {}),
        },
        **({"structured_input": "pauli_decomposition"} if array is None else {}),
    }
    if selected == "dense_dilation":
        scaled = normalized_matrix(dense_source, dense_alpha)
        # The Pauli gate counts stay as the record of the comparison, while
        # the structural facts and query census describe the selected circuit.
        routing.pop("query_census")
        routing.update(_dense_structure_facts(dense_source, scaled))
        return BlockEncodingPlan(
            requested_implementation=implementation,
            implementation=selected,
            alpha=float(dense_alpha),
            num_ancillas=1,
            system_qubits=system_qubits,
            error_bound=_normalization_error_bound(dense_source, dense_alpha, scaled),
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
    max_bytes: int | None = None,
    max_work: int | None = None,
) -> BlockEncoding:
    """Build a block-encoding circuit of a square matrix, a Pauli sum or a circulant.

    The function chooses the construction as
    [`plan_block_encoding`][nwqlib.subroutines.block_encoding.plan_block_encoding]
    does and builds it. It computes one dense SVD only when it needs the
    dense normalization, that is, when the dense dilation is requested or,
    under `"auto"`, costs no more than the Pauli construction, and it uses
    that SVD for the normalization, the equal-cost comparison and the
    completion. A banded plan and a strictly cheaper Pauli plan need no
    dense SVD. The
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
            accepted as the resolved name of `"pauli_lcu"`. `"auto"` first
            looks for a circulant within a roundoff window of the input, as
            `plan_block_encoding` does, and records the detection residual
            as `error_bound`. Otherwise it compares the per-query CX counts
            of the Pauli and dense-dilation constructions without building
            either one.
        max_bytes (int | None): Default `None`, the physical memory. Limit
            in bytes on the array data that each planning or construction
            step holds at the same time, checked before the step allocates
            it. NumPy, LAPACK and Qiskit internal workspace and Python
            objects are not counted.
        max_work (int | None): Default `None`, no limit. Limit on the work of each
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


def build_block_encoding_from_plan(plan: BlockEncodingPlan, *, max_bytes=None,
                                   max_work=None) -> BlockEncoding:
    """Build the circuit of a `BlockEncodingPlan` without choosing again.

    A dense-dilation plan is completed with a full SVD of its source,
    counted as `8 D**3 + D**3` work units for dimension D.

    Args:
        plan (BlockEncodingPlan): A plan from `plan_block_encoding`.
        max_bytes (int | None): Default `None`, the physical memory. Pass
            the value used for the plan.
        max_work (int | None): Default `None`, no limit. Pass the value
            used for the plan.

    Returns:
        encoding (BlockEncoding): The circuit of `plan.implementation`, with
            the plan's `alpha`, `num_ancillas` and `error_bound`.

    Raises:
        ValueError: If a construction step exceeds `max_bytes` or
            `max_work`.
    """
    return _build_block_encoding_from_plan(plan, max_bytes=max_bytes, max_work=max_work)


def _build_block_encoding_from_plan(plan, *, max_bytes, max_work, selected_svd=None):
    """Check the plan's byte and work formulas, then build ``plan.implementation`` as planned.

    Dense dilation is checked with the completion formula of
    ``_admit_dense_input``. A plan that also keeps a Pauli decomposition of
    L terms keeps it alive during the completion, so the completion bytes
    are checked together with the ``L (n + 16)`` bytes of its labels and
    coefficients.

    Multiplexed Pauli and banded plans are checked with the table formula
    ``64 P (n + 16)`` bytes and ``32 P max(1, n)**2`` work for
    ``P = 2**num_ancillas`` padded addresses and n system qubits, before any
    table is formed. For each system qubit the Pauli builder reads the
    address support that ``_pauli_plan_detail`` recorded in the plan, builds
    the projected table of at most P local 2x2
    factors as complex128 matrices (64 bytes each) from the labels, and
    pads it for the ``UCGate``, whose at most P matrices stay in the
    circuit. The n kept
    tables give the ``64 P n`` bytes of matrix data, and the transient
    copies of one qubit's table fit in the untuned 16 extra matrices per
    address. This table count excludes the NumPy object headers of the
    2x2 arrays. ``_admit_lcu`` checks the PREP arrays separately. The banded builder's per-qubit tables
    of P RZ angles are smaller. The work per system qubit grows as ``P (a + c)`` for
    ``a = log2 P`` address qubits and a small constant c: a pairwise
    equality test per address bit (on the packed letter codes, at
    planning), the padding, the projection, the label lookups
    and the ``UCGate`` unitarity check of each 2x2 matrix. The
    untuned ``32 P n`` per qubit of the ``32 P n**2`` formula grows at least as
    fast when ``a <= 2n``, which holds for distinct Pauli words. A
    decomposition with repeated words can exceed it.

    A banded plan detected from other input carries its detection bound into
    ``error_bound``. ``selected_svd`` lets a full build reuse the SVD that
    already fixed alpha for the dense completion.
    """
    max_work = optional_limit(max_work, "max_work")
    _check_bytes(0, max_bytes)
    if plan.implementation == "dense_dilation":
        if plan.decomposition is not None:
            d = 1 << plan.system_qubits
            _check_bytes(256*d*d+32*d+len(plan.decomposition.terms)*(plan.system_qubits+16),
                max_bytes, "block-encoding completion and kept Pauli terms")
        _admit_dense_input(plan.source, max_bytes=max_bytes, max_work=max_work, construction=True,
            selected_svd=selected_svd is not None)
    else:
        n, a = plan.system_qubits, plan.num_ancillas
        _check_bytes(64*(1 << a)*(n+16), max_bytes, "block-encoding construction tables")
        if exceeds(need := 32*(1 << a)*max(1,n)**2, max_work):
            raise ResourceLimitError(f"block-encoding construction exceeds max_work: it needs {need} work units, "
                                     f"max_work={max_work}")

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
                    "target_operator_is_hermitian": plan.detail["target_operator_is_hermitian"],
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
    "block_encoding_query_cx",
    "block_encoding_top_left",
    "build_block_encoding",
    "build_block_encoding_from_plan",
    "build_block_encoding_query",
    "plan_block_encoding",
]
