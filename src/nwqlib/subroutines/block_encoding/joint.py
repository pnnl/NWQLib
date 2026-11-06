r"""Joint LCHS generators, convex contractions and block-encoding products, built as native queries.

The transformations of ``nwqlib.subroutines.qsp.evolution`` act on a block
encoding ``U`` with ``||X - alpha <0|U|0>|| <= delta``. This module builds
the encodings of the joint LCHS generator and of the imaginary-time
contraction from planned child encodings, and gives one query function,
`build_access_query`, for every source those transformations accept. Each
construction below states the identities it uses.

Separate parts. With a node diagonal ``D_K`` on the node
register, an optional application-time diagonal ``D_Q`` with entries
``q_a = t_a / T_e`` on the application register, and centered children
``L_0``, ``H_0``,

    \[S = D_Q\otimes(D_K\otimes L_0 + I\otimes H_0),\qquad
      \alpha_S=\lambda_Q(\lambda_K\alpha_L+\alpha_H).\]

``lambda_K`` is the normalization of the node encoding, which covers its
padding completion (``UniformNodeRecipe.alpha``), not the physical radius.
The two branches ``D_K (x) U_L`` and ``I (x) U_H`` are combined by a real
RY on one qubit with ``cos^2(theta/2) = lambda_K alpha_L / (lambda_K
alpha_L + alpha_H)``, and ``U_Q`` is tensored on. A branch whose child is
absent (an exact zero) is omitted, so no combine qubit is used. Every
factor acts on its own registers and the combine shell is real, so the
whole unitary is a Hermitian involution exactly when the node, time and
child encodings are. The three facts (Hermitian target, Hermitian encoded
block, full involution) are propagated separately. With node and time
residuals ``delta_K``, ``delta_Q`` and target norm bounds
``B_X = alpha_X + delta_X``, the triangle inequality and the tensor
product give

    \[\delta_{\rm hom}\le\lambda_K\delta_L+\delta_K B_L+\delta_H,\qquad
      \delta_S\le\lambda_Q\delta_{\rm hom}+\delta_Q(\lambda_K B_L+B_H),\]

plus the rounding of the combine weights, ``2 alpha |cos^2(theta/2) -
w_L/alpha|`` for the stored angle.

Single A. For an encoding ``U_A`` of ``A_0 = L_0 + i H_0`` and a
complex diagonal ``C = D_Q (x) diag(c)`` with ``|c| <= r``, ``B = U_C (x)
U_A`` and

    \[U_S=H_h\begin{pmatrix}0&B\\B^\dagger&0\end{pmatrix}H_h
        =H_h D_B X_h D_B^\dagger H_h,\qquad D_B={\rm diag}(B,I)_h.\]

The block is ``(B_00 + B_00^dagger)/2 = (d_L (x) L_0 + d_H (x) H_0) /
(r alpha_A)`` for ``c = d_L - i d_H``, so ``alpha_S = r alpha_A``. One
Hermitianization forms the joint block and ``U_S`` is a full involution
for every ``U_A``. External controls act only on ``X_h``, because
``D_B D_B^dagger = I`` on the inactive branch. With ``|c_hat - c| <=
delta_C``, ``delta_S <= r delta_A + delta_C (alpha_A + delta_A)``.

Convex contraction. For an encoding of ``L_0`` and ``alpha >=
alpha_L + delta_L``, ``X_a = (1 - q_a) I - q_a L_0 / alpha`` is encoded
with normalization one by an application-multiplexed RY on one combine
qubit, amplitudes ``sqrt(1-q_a)`` and ``sqrt(q_a)``, the identity on the
first branch and ``-(c_s (x) U_L)`` on the second, where ``c_s =
alpha_L/alpha`` is a one-qubit scale encoding when ``alpha > alpha_L``.
Then ``e^{-t_a L} = kappa_a e^{-beta(1 - X_a)}`` with ``beta = T_e alpha``.
The second branch is a Hermitian involution when ``U_L`` is, so the whole
unitary inherits that fact. ``delta_X <= max_a q_a (delta_L/alpha +
|c_hat_s - c_s|) + 2 max_a |q_hat_a - q_a|``.

Product. Two encodings with independent ancilla registers on the
same system give ``<0|U_1 U_2|0> = B_1 B_2``, ``alpha = alpha_1 alpha_2``
and ``err <= e_1 N_2 + (N_1 + e_1) e_2`` for target norm bounds ``N_i``.

Scaled branch (B1–B4 of the finite-bank relations derived in
`nwqlib.algorithms.lchs.sampled_qsp`). One branch of a sampled LCHS pair evolves
``U_x = exp(-i t_x G_x)`` with ``G_x = k L_0 + H_0``. Its query
``V_x = P(theta)^dagger SELECT P(theta)`` with ``SELECT = |0><0| (x)
sign(k) U_L + |1><1| (x) U_H`` and ``cos^2(theta/2) = |k| alpha_L / a_x``
encodes ``G_x`` with ``a_x = |k| alpha_L + alpha_H``, the separate-parts
normalization of a length-one node table without address padding; the
sign is an exact phase flip and a single present part has no combine
qubit. For a bank time ``t_b >= t_x a_x`` and ``r = t_x a_x / t_b``, the
scale flag ``D(r) = [[r, s], [s, -r]] = R_Y(2 arccos r) Z``, ``s =
sqrt(1 - r^2)``, gives ``D(r) (x) V_x`` the block ``r B_x`` with
normalization one, and substituting ``r`` and factoring ``t_x/t_b``

    \[\|r\,B_x - t_x G_x/t_b\|\le (t_x/t_b)\,\delta_x,\qquad
      \delta_x\le|k|\delta_L+\delta_H+2a_x|\cos^2(\hat\theta/2)-|k|\alpha_L/a_x|.\]

A realized amplitude ``r_hat = cos(phi_hat/2)`` of the stored angle adds
``|r_hat - r|``, because the block of a unitary is a contraction. D(r) is a
real Hermitian involution on its own qubit, so the three structural facts
are those of the present children. A zero branch (``t_x a_x = 0``) is the
normalization-one zero block ``D(0) (x) I``, which a bank polynomial maps to
its value at zero.

Every residual above that comes from a stored binary64 angle is a
numerical estimate, not an enclosure. Query construction follows the
structured-control rule of the child owners: only the core of a native
child is controlled (``build_block_encoding_query``), and the gates of the
combine shells here are never controlled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.subroutines.block_encoding.core import (
    STRUCTURE_FACT_KEYS,
    BlockEncoding,
    BlockEncodingPlan,
    _append_mcx,
    _append_phase_flip,
    _control_state,
    _mcx_cx,
    _UnknownMCXCount,
    _build_block_encoding_from_plan,
    block_encoding_query_cx,
    build_block_encoding_query,
)
from nwqlib.subroutines.block_encoding.nodes import (
    UniformNodeRecipe,
    _normalization_error,
    _rotation_table,
    _uniform_values,
    angle_table_cx,
    build_diagonal_table_query,
    build_uniform_node_query,
    uniform_node_query_cx,
)


@dataclass(frozen=True, kw_only=True)
class DiagonalTable:
    """A real diagonal on an address register, encoded as the angle table of `build_diagonal_table_query`.

    Attributes:
        values: Real diagonal entries in their physical units. Entries beyond
            `len(values)` up to the next power of two are 0.
        normalization: `lambda >= max |values|`, the encoding normalization.
    """

    values: tuple[float, ...]
    normalization: float


@dataclass(frozen=True, kw_only=True)
class ComplexDiagonalTable:
    """A complex diagonal ``diag(c)`` on an address register with ``|c| <= radius``, for the single-A generator.

    The encoding is the angle table of ``|c| / radius`` followed by the phase
    diagonal ``exp(i arg c)`` on the address register, so its block is
    ``diag(c) / radius``. Whether its whole unitary is an involution is not
    recorded.

    Attributes:
        values: Complex entries; entries beyond `len(values)` up to the next
            power of two are 0.
        radius: `r >= max |values|`, the encoding normalization.
    """

    values: tuple[complex, ...]
    radius: float


@dataclass(frozen=True, kw_only=True)
class AccessSummary:
    """The normalization, error, widths and structural facts of a block-encoding source, known without a circuit.

    Attributes:
        alpha: Normalization in the units of the target.
        error_bound: Bound on `||X - alpha <0|U|0>||`, or None when unknown.
        num_ancillas: Ancillas returned to zero on the encoded block.
        system_qubits: Width of the register the target acts on.
        target_operator_is_hermitian: True, False or None (unknown).
        encoded_operator_is_hermitian: True, False or None (unknown).
        full_unitary_involution: Whether `U = U^dagger`: True, False or None.
        implementation: Name of the construction.
        error_bound_evidence: What the error bound covers.
    """

    alpha: float
    error_bound: float | None
    num_ancillas: int
    system_qubits: int
    target_operator_is_hermitian: bool | None
    encoded_operator_is_hermitian: bool | None
    full_unitary_involution: bool | None
    implementation: str
    error_bound_evidence: str = ""


@dataclass(frozen=True, kw_only=True)
class JointEncodingPlan:
    """A composite block encoding chosen before any circuit exists: a joint generator, a contraction or a product.

    `build_access_query` and `build_joint_encoding` build exactly this
    construction from `components`, without choosing again.

    Attributes:
        construction: `"separate_parts"` (`plan_joint_generator`),
            `"single_a"` (`plan_single_a_generator`), `"convex_contraction"`
            (`plan_convex_contraction`) or `"product"`
            (`plan_block_encoding_product`).
        alpha: Normalization of the encoded target.
        error_bound: Operator-norm bound on `||X - alpha <0|U|0>||`, or None
            when a component bound is unknown. Residuals of stored binary64
            angles are numerical estimates.
        num_ancillas: Ancillas returned to zero on the encoded block.
        system_qubits: Encoded system width (address registers included).
        target_operator_is_hermitian: True, False or None (unknown).
        encoded_operator_is_hermitian: True, False or None (unknown).
        full_unitary_involution: Whether the whole unitary is a Hermitian
            involution: True, False or None.
        layout: Register widths and their order, low to high: ancillas
            then system.
        error_terms: The parts of `error_bound` and their evidence.
        query_census: Child-query calls of one forward query and of one
            query with one external control, as `(component, controls,
            adjoint, count)` rows, and CX upper bounds of both when every
            component has a CX law (`forward_cx`, `controlled_cx`, None
            otherwise). The bounds inherit the child laws: PREP slot bounds
            and Qiskit's multi-controlled X counts without ancillas, which a
            lowering that borrows idle qubits can undercut.
        components: The planned children, tables and recipes that the
            builders read.
    """

    construction: Literal["separate_parts", "single_a", "convex_contraction", "product"]
    alpha: float
    error_bound: float | None
    num_ancillas: int
    system_qubits: int
    target_operator_is_hermitian: bool | None
    encoded_operator_is_hermitian: bool | None
    full_unitary_involution: bool | None
    layout: Mapping[str, Any]
    error_terms: Mapping[str, Any]
    query_census: Mapping[str, Any]
    components: Mapping[str, Any] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the scalar fields, layout, error terms and census as JSON-like data, without the components."""
        return {
            "construction": self.construction, "alpha": self.alpha, "error_bound": self.error_bound,
            "num_ancillas": self.num_ancillas, "system_qubits": self.system_qubits,
            **{name: getattr(self, name) for name in STRUCTURE_FACT_KEYS},
            "layout": dict(self.layout), "error_terms": dict(self.error_terms),
            "query_census": {key: (list(map(list, value)) if key.endswith("_calls") else value)
                             for key, value in self.query_census.items()},
        }


Source = Any


@dataclass(frozen=True, kw_only=True)
class ScaledBranchAccess:
    """The normalization-one query ``D(r) (x) V_x`` of one sampled-pair branch (module docstring, Scaled branch).

    Planned per row from the Plan's centered children; nothing is built until
    `build_access_query` is called. Its target is ``t_x G_x / t_b``.

    Attributes:
        l_child: Encoding of the centered dissipative part ``L_0``, or None
            for an exact zero.
        h_child: Encoding of the centered Hamiltonian part ``H_0``, or None.
        node: The kernel node k of the branch; ``k = 0`` omits ``L_0``.
        scale: ``r = t_x a_x / t_b`` in [0, 1]; 0 is the zero block
            ``D(0) (x) I``, which applies no child.
        system_qubits: Width of the children's system.
        angles: Optional ``(scale angle, combine angle)`` used in place of
            the stored ``2 arccos r`` and ``theta``, for example Qiskit
            parameters of a row template. The summary always uses the
            stored values.
    """

    l_child: Source | None
    h_child: Source | None
    node: float
    scale: float
    system_qubits: int
    angles: tuple | None = field(default=None, compare=False, repr=False)


def _branch_parts(access: ScaledBranchAccess):
    """Return the present ``(name, child, summary, weight)`` parts, the combine angle and its weight residual."""
    if access.scale == 0.0:
        return (), None, 0.0
    parts = []
    if access.l_child is not None and access.node != 0.0:
        summary = access_summary(access.l_child)
        parts.append(("l_child", access.l_child, summary, abs(access.node) * summary.alpha))
    if access.h_child is not None:
        summary = access_summary(access.h_child)
        parts.append(("h_child", access.h_child, summary, summary.alpha))
    if not parts:
        raise ValueError("a scaled branch with a positive scale needs a nonzero part")
    if len(parts) == 1:
        return tuple(parts), None, 0.0
    w_l, w_h = parts[0][3], parts[1][3]
    total = math.fsum((w_l, w_h))
    theta = 2.0 * math.atan2(math.sqrt(w_h), math.sqrt(w_l))
    return tuple(parts), theta, 2.0 * total * abs(math.cos(theta / 2.0) ** 2 - w_l / total)


def _scaled_summary(access: ScaledBranchAccess) -> AccessSummary:
    scale = finite_real(access.scale, "scale")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("a branch scale r = t_x a_x / t_b must lie in [0, 1]")
    _, residual = _rotation_table(np.array([scale]), 1.0)
    parts, _, combine = _branch_parts(access)
    if any(summary.system_qubits != access.system_qubits for *_, summary, _ in parts):
        raise ValueError("the children of a scaled branch must act on its system width")
    if not parts:
        return AccessSummary(alpha=1.0, error_bound=float(np.nextafter(residual, np.inf)), num_ancillas=1,
                             system_qubits=access.system_qubits, target_operator_is_hermitian=True,
                             encoded_operator_is_hermitian=True, full_unitary_involution=True,
                             implementation="scaled_zero_branch",
                             error_bound_evidence="binary64 residual of the stored scale angle; a numerical estimate")
    alpha = math.fsum(weight for *_, weight in parts)
    child = [(abs(access.node) if name == "l_child" else 1.0, summary.error_bound)
             for name, _, summary, _ in parts]
    error = None
    if all(bound is not None for _, bound in child):
        delta = math.fsum([factor * bound for factor, bound in child] + [combine])
        error = float(np.nextafter(scale * delta / alpha + residual, np.inf))
    return AccessSummary(
        alpha=1.0, error_bound=error, num_ancillas=1 + int(len(parts) == 2)
        + max(summary.num_ancillas for _, _, summary, _ in parts), system_qubits=access.system_qubits,
        target_operator_is_hermitian=_combined_fact(summary.target_operator_is_hermitian for *_, summary, _ in parts),
        encoded_operator_is_hermitian=_combined_fact(
            summary.encoded_operator_is_hermitian for *_, summary, _ in parts),
        full_unitary_involution=_all_true(summary.full_unitary_involution for *_, summary, _ in parts),
        implementation="scaled_branch",
        error_bound_evidence="r*delta_x/a_x + |r_hat - r| (B2, B3); delta_x composes the child bounds with the "
                             "combine-weight residual; the stored-angle residuals are numerical estimates")


def _table_width(length: int) -> int:
    return (length - 1).bit_length()


def _supplied(source: Source) -> Source:
    """Return the `BlockEncoding` of an operator input from `ingest_block_encoding`, else the source itself.

    A Plan keeps such an operator input, which the archive saves with its
    circuit, and the encoding is read from it where it is queried.
    """
    representation = getattr(getattr(getattr(source, "manifest", None), "reference", None), "representation", None)
    return source.supplied_block_encoding().encoding if representation == "block_encoding" else source


def _num_ancillas(source: Source) -> int:
    """Return the ancilla count that `access_summary` gives, without evaluating a table's error bound."""
    source = _supplied(source)
    if isinstance(source, UniformNodeRecipe):
        return source.encoding_ancillas
    if isinstance(source, (DiagonalTable, ComplexDiagonalTable)):
        return 1
    return access_summary(source).num_ancillas


def access_summary(source: Source) -> AccessSummary:
    """Return the normalization, error bound, widths and structural facts of a source, without building a circuit.

    Args:
        source: A `BlockEncodingPlan`, a built `BlockEncoding` (a supplied
            circuit, whose facts are the caller's assertions in its
            metadata), an operator input from `ingest_block_encoding` (its
            supplied encoding), a `UniformNodeRecipe`, a `DiagonalTable`, a
            `ComplexDiagonalTable`, a `JointEncodingPlan` or a
            `ScaledBranchAccess`.

    Returns:
        summary (AccessSummary): The record above.

    Raises:
        TypeError: For any other source.
        ValueError: For an `"exact_zero"` `BlockEncodingPlan`, which has no
            encoding; omit that part instead.
    """
    source = _supplied(source)
    if isinstance(source, ScaledBranchAccess):
        return _scaled_summary(source)
    if isinstance(source, JointEncodingPlan):
        return AccessSummary(alpha=source.alpha, error_bound=source.error_bound, num_ancillas=source.num_ancillas,
                             system_qubits=source.system_qubits,
                             **{name: getattr(source, name) for name in STRUCTURE_FACT_KEYS},
                             implementation="joint_" + source.construction,
                             error_bound_evidence=str(source.error_terms.get("evidence", "")))
    if isinstance(source, BlockEncodingPlan):
        if source.implementation == "exact_zero":
            raise ValueError("an exact-zero part has no encoding; omit it instead")
        return AccessSummary(alpha=source.alpha, error_bound=source.error_bound, num_ancillas=source.num_ancillas,
                             system_qubits=source.system_qubits,
                             **{name: source.detail.get(name) for name in STRUCTURE_FACT_KEYS},
                             implementation=source.implementation, error_bound_evidence="plan error bound")
    if isinstance(source, BlockEncoding):
        return AccessSummary(alpha=source.alpha, error_bound=source.error_bound, num_ancillas=source.num_ancillas,
                             system_qubits=source.system_qubits,
                             **{name: source.metadata.get(name) for name in STRUCTURE_FACT_KEYS},
                             implementation=source.implementation,
                             error_bound_evidence=str(source.metadata.get("error_bound_evidence", "")))
    if isinstance(source, UniformNodeRecipe):
        if source.kind == "rotation_table":
            _, residual = _rotation_table(_uniform_values(source), source.alpha)
            error = float(np.nextafter(residual, np.inf))
            evidence = "binary64 residual of the stored angles; a numerical estimate"
        else:
            extent = source.half_width if source.plan is None else int(source.plan.alpha)
            error = _normalization_error(source.spacing, extent, source.alpha)
            evidence = "|h N - alpha| of the exact block j/N, rounded up"
        return AccessSummary(alpha=source.alpha, error_bound=error, num_ancillas=source.encoding_ancillas,
                             system_qubits=source.address_qubits, target_operator_is_hermitian=True,
                             encoded_operator_is_hermitian=True, full_unitary_involution=True,
                             implementation="uniform_node_" + source.kind, error_bound_evidence=evidence)
    if isinstance(source, DiagonalTable):
        _, residual = _rotation_table(np.asarray(source.values, float), source.normalization)
        return AccessSummary(alpha=source.normalization, error_bound=float(np.nextafter(residual, np.inf)),
                             num_ancillas=1, system_qubits=_table_width(len(source.values)),
                             target_operator_is_hermitian=True, encoded_operator_is_hermitian=True,
                             full_unitary_involution=True, implementation="diagonal_rotation_table",
                             error_bound_evidence="binary64 residual of the stored angles; a numerical estimate")
    if isinstance(source, ComplexDiagonalTable):
        _, _, residual = _complex_table(source)
        return AccessSummary(alpha=source.radius, error_bound=float(np.nextafter(residual, np.inf)), num_ancillas=1,
                             system_qubits=_table_width(len(source.values)),
                             target_operator_is_hermitian=bool(np.all(np.asarray(source.values).imag == 0)),
                             encoded_operator_is_hermitian=None, full_unitary_involution=None,
                             implementation="complex_diagonal_table",
                             error_bound_evidence="binary64 residual of the stored angles and phases; a numerical "
                                                  "estimate")
    raise TypeError(f"unsupported block-encoding source {type(source).__name__}")


def _complex_table(table: ComplexDiagonalTable):
    """Return the magnitude angles, the phase table and the binary64 residual ``max |c - r cos(theta/2) e^{i phi}|``."""
    values = np.asarray(table.values, dtype=complex)
    radius = finite_real(table.radius, "radius")
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("complex diagonal values must be a nonempty finite one-dimensional sequence")
    if radius <= 0 or float(np.max(np.abs(values))) > radius:
        raise ValueError("radius must be positive and at least the largest value magnitude")
    padded = np.zeros(1 << _table_width(values.size), complex)
    padded[:values.size] = values
    magnitudes = np.abs(padded)
    angles = 2.0 * np.arccos(magnitudes / radius)
    phases = np.where(magnitudes > 0, np.angle(padded), 0.0)
    realized = radius * np.cos(angles / 2.0) * np.exp(1j * phases)
    return angles, phases, float(np.max(np.abs(realized - padded)))


def _combined_fact(values) -> bool | None:
    """Combine one structural fact over the summands of a real combination.

    All True gives True. With nonzero real weights, exactly one known
    non-Hermitian summand among Hermitian ones keeps its anti-Hermitian
    part, so the sum is not Hermitian; two or more unknown or False
    summands could cancel, so the result is unknown.
    """
    values = tuple(values)
    if all(value is True for value in values):
        return True
    if sum(value is False for value in values) == 1 and all(type(value) is bool for value in values):
        return False
    return None


def _all_true(values) -> bool | None:
    """True when every factor has the property, False when one is known not to, otherwise unknown."""
    values = tuple(values)
    if all(value is True for value in values):
        return True
    return False if any(value is False for value in values) else None


def _component_cx(component, num_controls: int) -> int | None:
    """CX upper bound of one query of a component with k controls, or None when no law is known."""
    k = num_controls
    try:
        if isinstance(component, BlockEncodingPlan):
            return block_encoding_query_cx(component, num_controls=k)
        if isinstance(component, UniformNodeRecipe):
            return uniform_node_query_cx(component, k)
        if isinstance(component, (DiagonalTable, ComplexDiagonalTable)):
            r = _table_width(len(component.values))
            cx = angle_table_cx(r, k)
            if isinstance(component, ComplexDiagonalTable):
                from nwqlib.subroutines._multiplexors import multiplexor_resource_law
                cx += multiplexor_resource_law("diagonal", control_qubits=r + k)["basis_cx_gates"]
            return cx
        if isinstance(component, JointEncodingPlan):
            if k <= 1:
                return component.query_census["controlled_cx" if k else "forward_cx"]
            return None
    except _UnknownMCXCount:
        return None
    return None


def _census(rows, own_cx: int, controlled_rows, controlled_own_cx: int | None) -> dict[str, Any]:
    """Return the census of one forward query and one singly controlled query from their component rows."""
    def total(entries, own):
        if own is None:
            return None
        parts = [_component_cx(component, controls) for component, _, controls, _, count in entries for _ in
                 range(count)]
        return None if any(part is None for part in parts) else own + sum(parts)
    return {
        "forward_calls": tuple((name, controls, adjoint, count) for _, name, controls, adjoint, count in rows),
        "controlled_calls": tuple((name, controls, adjoint, count)
                                  for _, name, controls, adjoint, count in controlled_rows),
        "forward_cx": total(rows, own_cx),
        "controlled_cx": total(controlled_rows, controlled_own_cx),
        "cx_qualification": "child CX laws; PREP slot bounds and Qiskit multi-controlled X counts without "
                            "ancillas are upper bounds that a lowering borrowing idle qubits can undercut",
    }


def _node_source(nodes):
    if isinstance(nodes, (UniformNodeRecipe, DiagonalTable)):
        return nodes
    raise TypeError("nodes must be a UniformNodeRecipe or a DiagonalTable")


def _time_table(time_factors):
    if time_factors is None:
        return None
    values = np.asarray(time_factors, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("time_factors must be a nonempty finite one-dimensional sequence")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("time factors q_a = t_a / T_e must lie in [0, 1]")
    top = float(np.max(values))
    if top == 0:
        raise ValueError("time factors are all zero; the generator vanishes")
    return DiagonalTable(values=tuple(float(v) for v in values), normalization=top)


def plan_joint_generator(*, nodes=None, l_child: Source | None = None, h_child: Source | None = None,
                         time_factors=None) -> JointEncodingPlan:
    r"""Plan the joint LCHS generator ``S = D_Q (x) (D_K (x) L_0 + I (x) H_0)`` from separate encodings of its parts.

    Each address slot ``nu = (a, j)`` carries ``q_a k_j L_0 + q_a H_0``, so
    ``exp(-i T_e S)`` applies ``exp(-i t_a (k_j L_0 + H_0))`` on slot nu.
    The normalization is
    ``alpha_S = lambda_Q (lambda_K alpha_L + alpha_H)`` with the
    actual normalizations of the node and time encodings, and the error
    bound is the one derived in
    [Joint generators](../../algorithms/lchs_derivations.md#joint-generators).
    A part given as None is an exact zero and gets no branch.

    Args:
        nodes (UniformNodeRecipe | DiagonalTable | None): Encoding of the kernel nodes
            ``k_j``, required with `l_child`. Its normalization, which
            covers the padding completion, is ``lambda_K``.
        l_child: Encoding of the centered dissipative part ``L_0``, or None.
        h_child: Encoding of the centered Hamiltonian part ``H_0``, or None.
        time_factors (Sequence[float] | None): Optional ``q_a = t_a / T_e`` in [0, 1] per application
            label, for a constant source; padding labels take 0.

    Returns:
        plan (JointEncodingPlan): The `"separate_parts"` plan. Ancilla
        order: combine qubit (two parts only), time signal, node signal and
        work, shared child ancillas. System order: node register, application
        register, child system.

    Raises:
        ValueError: When both parts are absent, a part is an
            `"exact_zero"` plan (pass None instead), the child widths
            differ, `nodes` is missing with `l_child`, or the time factors
            are invalid or all zero.
        TypeError: For an unsupported source type.
    """
    if l_child is None and h_child is None:
        raise ValueError("joint generator needs at least one nonzero part")
    if l_child is not None and nodes is None:
        raise ValueError("the dissipative part needs its node encoding")
    node = _node_source(nodes) if l_child is not None else None
    time = _time_table(time_factors)
    parts = {name: (child, access_summary(child)) for name, child in (("l_child", l_child), ("h_child", h_child))
             if child is not None}
    widths = {summary.system_qubits for _, summary in parts.values()}
    if len(widths) != 1:
        raise ValueError("L and H encodings must share the system width")
    child_system = widths.pop()
    node_summary = access_summary(node) if node is not None else None
    time_summary = access_summary(time) if time is not None else None
    lam_k = node_summary.alpha if node_summary else 0.0
    weights = {}
    if "l_child" in parts:
        weights["l_child"] = lam_k * parts["l_child"][1].alpha
    if "h_child" in parts:
        weights["h_child"] = parts["h_child"][1].alpha
    lam_q = time_summary.alpha if time_summary else 1.0
    homogeneous_alpha = math.fsum(weights.values())
    alpha = lam_q * homogeneous_alpha
    theta = (2.0 * math.atan2(math.sqrt(weights["h_child"]), math.sqrt(weights["l_child"]))
             if len(weights) == 2 else None)

    def bound(summary):
        return None if summary.error_bound is None else summary.alpha + summary.error_bound

    errors = [s.error_bound for s in (node_summary, time_summary) if s is not None]
    errors += [summary.error_bound for _, summary in parts.values()]
    if any(value is None for value in errors):
        delta_s = None
        terms = {}
    else:
        delta_hom = 0.0
        norm_hom = 0.0
        if "l_child" in parts:
            summary = parts["l_child"][1]
            delta_hom += lam_k * summary.error_bound + node_summary.error_bound * bound(summary)
            norm_hom += lam_k * bound(summary)
        if "h_child" in parts:
            summary = parts["h_child"][1]
            delta_hom += summary.error_bound
            norm_hom += bound(summary)
        combine = 0.0
        if theta is not None:
            combine = 2.0 * homogeneous_alpha * abs(math.cos(theta / 2.0) ** 2 - weights["l_child"] / homogeneous_alpha)
        delta_hom += combine
        delta_s = lam_q * delta_hom + (time_summary.error_bound * norm_hom if time_summary else 0.0)
        terms = {"homogeneous": delta_hom, "combine_weight_residual": combine,
                 "time_factor": time_summary.error_bound * norm_hom if time_summary else 0.0}
    terms["evidence"] = ("triangle-inequality composition of the child bounds with the node and time residuals of "
                         "their stored binary64 angles and the combine-weight residual; the residuals are numerical "
                         "estimates")
    summaries = [s for s in (node_summary, time_summary) if s is not None] + [s for _, s in parts.values()]
    node_ancillas = node_summary.num_ancillas if node_summary else 0
    child_ancillas = max(summary.num_ancillas for _, summary in parts.values())
    layout = {"combine": int(len(parts) == 2), "time_signal": int(time is not None), "node_ancillas": node_ancillas,
              "child_ancillas": child_ancillas, "node_qubits": node_summary.system_qubits if node_summary else 0,
              "application_qubits": time_summary.system_qubits if time_summary else 0,
              "child_system_qubits": child_system}
    num_ancillas = layout["combine"] + layout["time_signal"] + node_ancillas + child_ancillas
    system = layout["node_qubits"] + layout["application_qubits"] + child_system
    components = {"nodes": node, "time": time, **{name: child for name, (child, _) in parts.items()},
                  "theta": theta, "weights": dict(weights)}
    extra = int(len(parts) == 2)
    rows = []
    if time is not None:
        rows.append((time, "time", 0, False, 1))
    if node is not None:
        rows.append((node, "nodes", extra, False, 1))
    for name, (child, _) in parts.items():
        rows.append((child, name, extra, False, 1))
    controlled = [(component, name, controls + 1, adjoint, count) for component, name, controls, adjoint, count
                  in rows]
    facts = {
        "target_operator_is_hermitian": _combined_fact(s.target_operator_is_hermitian for _, s in parts.values()),
        "encoded_operator_is_hermitian": _combined_fact(
            s.encoded_operator_is_hermitian for _, s in parts.values()),
        "full_unitary_involution": _all_true(s.full_unitary_involution for s in summaries),
    }
    return JointEncodingPlan(construction="separate_parts", alpha=alpha, error_bound=delta_s,
                             num_ancillas=num_ancillas, system_qubits=system, **facts, layout=layout,
                             error_terms=terms, query_census=_census(rows, 0, controlled, 0), components=components)


def plan_single_a_generator(*, a_child: Source, node_coefficients, radius: float,
                            time_factors=None) -> JointEncodingPlan:
    r"""Plan the single-A joint generator from one encoding of ``A_0 = L_0 + i H_0``.

    With ``c_j = d_L[j] - i d_H[j]``, for the kernel ``c_j = k_j - i``, and
    ``|c_j| <= r``, the block is ``S / (r alpha_A)`` with
    ``S = D_Q (x) (d_L (x) L_0 + d_H (x) H_0)``. One Hermitianization helper
    h forms the joint block; no second lift is needed and ``U_S`` is a full
    involution. ``r`` must bound every stored entry, the padding completion
    included, which is the physical ``sqrt(R^2 + 1)`` only when padding
    entries are 0.

    Args:
        a_child: Encoding of ``A_0``, of any structure.
        node_coefficients (Sequence[complex]): Complex entries ``c_j`` per node label.
        radius: ``r >= max |c_j|``.
        time_factors (Sequence[float] | None): Optional ``q_a`` in [0, 1] per application label.

    Returns:
        plan (JointEncodingPlan): The `"single_a"` plan, ``alpha_S = lambda_Q r
        alpha_A``. Ancilla order: h, time signal, coefficient signal, child
        ancillas. System order: node register, application register, child
        system.

    Raises:
        ValueError: For invalid coefficients, radius or time factors, or an
            `"exact_zero"` child plan.
    """
    table = ComplexDiagonalTable(values=tuple(complex(v) for v in node_coefficients), radius=float(radius))
    time = _time_table(time_factors)
    child = access_summary(a_child)
    table_summary = access_summary(table)
    time_summary = access_summary(time) if time is not None else None
    lam_q = time_summary.alpha if time_summary else 1.0
    r = table.radius
    if child.error_bound is None:
        delta = None
    else:
        delta_c = table_summary.error_bound * lam_q + (time_summary.error_bound * r if time_summary else 0.0)
        delta = lam_q * r * child.error_bound + delta_c * (child.alpha + child.error_bound)
    layout = {"hermitian_helper": 1, "time_signal": int(time is not None), "coefficient_signal": 1,
              "child_ancillas": child.num_ancillas, "node_qubits": table_summary.system_qubits,
              "application_qubits": time_summary.system_qubits if time_summary else 0,
              "child_system_qubits": child.system_qubits}
    rows = [(table, "coefficients", 1, adjoint, 1) for adjoint in (False, True)]
    rows += [(a_child, "a_child", 1, adjoint, 1) for adjoint in (False, True)]
    if time is not None:
        rows += [(time, "time", 1, adjoint, 1) for adjoint in (False, True)]
    # D_B and D_B^dagger stay singly controlled on h; one external control adds one control to X_h only.
    own, controlled_own = 0, _mcx_cx(1)
    return JointEncodingPlan(
        construction="single_a", alpha=lam_q * r * child.alpha, error_bound=delta,
        num_ancillas=2 + int(time is not None) + child.num_ancillas,
        system_qubits=layout["node_qubits"] + layout["application_qubits"] + child.system_qubits,
        target_operator_is_hermitian=True, encoded_operator_is_hermitian=True, full_unitary_involution=True,
        layout=layout,
        error_terms={"evidence": "r*delta_A plus the coefficient-table residual times alpha_A+delta_A; the table "
                                 "residual is a numerical estimate"},
        query_census=_census(rows, own, rows, controlled_own),
        components={"a_child": a_child, "coefficients": table, "time": time})


def plan_convex_contraction(*, l_child: Source, alpha: float, time_factors=None) -> JointEncodingPlan:
    r"""Plan the encoding of ``X_a = (1 - q_a) I - q_a L_0 / alpha`` with normalization one.

    For the homogeneous case (no time factors) this is ``X = -L_0 / alpha``.
    The imaginary-time filter ``exp[-beta (1 - X_a)]`` with
    ``beta = T_e alpha`` then gives ``exp(-t_a L) = kappa_a exp[-beta(1 -
    X_a)]``, ``kappa_a = exp[-t_a (c - alpha)]``, for ``L = c I + L_0``.
    ``alpha`` must cover both the target and the implemented block, so it
    must be at least ``alpha_L + delta_L``; when it exceeds ``alpha_L`` a
    one-qubit scale encoding of ``alpha_L / alpha`` is tensored on.

    Args:
        l_child: Encoding of the centered dissipative part ``L_0``.
        alpha: Contraction normalization, at least ``alpha_L + delta_L``.
        time_factors (Sequence[float] | None): Optional ``q_a`` in [0, 1] per application label.

    Returns:
        plan (JointEncodingPlan): The `"convex_contraction"` plan. Ancilla
        order: combine qubit (with time factors), scale qubit (when
        ``alpha > alpha_L``), child ancillas. System order: application
        register, child system.

    Raises:
        ValueError: When the child is an `"exact_zero"` plan or its error is
            unknown, ``alpha`` does not cover ``alpha_L + delta_L``, or the
            time factors are invalid.
    """
    child = access_summary(l_child)
    alpha = finite_real(alpha, "alpha")
    if child.error_bound is None:
        raise ValueError("the contraction normalization needs a known child error bound")
    if alpha < child.alpha + child.error_bound:
        raise ValueError(f"alpha={alpha} must cover alpha_L + delta_L = {child.alpha + child.error_bound}")
    time = None
    angles = None
    q_residual = 0.0
    q_max = 1.0
    if time_factors is not None:
        q = np.asarray(_time_table(time_factors).values)
        padded = np.zeros(1 << _table_width(q.size))
        padded[:q.size] = q
        angles = 2.0 * np.arctan2(np.sqrt(padded), np.sqrt(1.0 - padded))
        q_residual = float(np.max(np.abs(np.sin(angles / 2.0) ** 2 - padded)))
        q_max = float(np.max(q))
        time = tuple(float(v) for v in padded)
    scale = child.alpha / alpha
    scale_table = DiagonalTable(values=(scale,), normalization=1.0) if scale < 1.0 else None
    scale_residual = access_summary(scale_table).error_bound if scale_table else 0.0
    delta = q_max * (child.error_bound / alpha + scale_residual) + 2.0 * q_residual
    layout = {"combine": int(time is not None), "scale": int(scale_table is not None),
              "child_ancillas": child.num_ancillas,
              "application_qubits": _table_width(len(time)) if time else 0,
              "child_system_qubits": child.system_qubits}
    extra = int(time is not None)
    rows = [(l_child, "l_child", extra, False, 1)]
    if scale_table is not None:
        rows.append((scale_table, "scale", extra, False, 1))
    controlled = [(c, name, k + 1, adj, n) for c, name, k, adj, n in rows]
    # The exact -1 on the active control value is one X with (flip qubits - 1) controls
    # (``_append_phase_flip``): none without an external control, one with it and the combine qubit.
    from nwqlib.subroutines._multiplexors import multiplexor_resource_law
    own = 0 if time is None else 2 * multiplexor_resource_law(
        "ucry", control_qubits=layout["application_qubits"])["basis_cx_gates"]
    controlled_own = own + _mcx_cx(extra)
    return JointEncodingPlan(
        construction="convex_contraction", alpha=1.0, error_bound=delta,
        num_ancillas=layout["combine"] + layout["scale"] + child.num_ancillas,
        system_qubits=layout["application_qubits"] + child.system_qubits,
        target_operator_is_hermitian=child.target_operator_is_hermitian,
        encoded_operator_is_hermitian=child.encoded_operator_is_hermitian,
        full_unitary_involution=child.full_unitary_involution, layout=layout,
        error_terms={"child": q_max * child.error_bound / alpha, "scale_residual": q_max * scale_residual,
                     "time_weight_residual": 2.0 * q_residual, "contraction_alpha": alpha,
                     "max_time_factor": q_max,
                     "evidence": "delta_X <= max_a q_a delta_L / alpha plus binary64 residuals of the stored scale "
                                 "and convex-weight angles (numerical estimates)"},
        query_census=_census(rows, own, controlled, controlled_own),
        components={"l_child": l_child, "scale": scale_table, "time": time, "angles": angles})


def plan_block_encoding_product(first: Source, second: Source, *,
                                target_norm_bounds: tuple[float, float] | None = None) -> JointEncodingPlan:
    r"""Plan the product of two encodings on one system with independent success ancillas.

    The block is ``B_1 B_2`` (``second`` acts first) and the normalization
    is ``alpha_1 alpha_2``. With target norm bounds ``N_i``, the default
    ``alpha_i + e_i``, the error is at most ``e_1 N_2 + (N_1 + e_1) e_2``;
    two contractions or unitaries with ``N_i = 1`` give
    ``e_1 + e_2 + e_1 e_2``. That the product of the two targets equals a
    wanted operator, such as ``exp(-tL) exp(-itH) = exp(-tA)`` for
    ``[L, H] = 0``, is the caller's premise.

    Args:
        first: Encoding applied second (left factor).
        second: Encoding applied first (right factor).
        target_norm_bounds: Optional ``(N_1, N_2)``.

    Returns:
        plan (JointEncodingPlan): The `"product"` plan with ancillas of
        `first` then of `second`.

    Raises:
        ValueError: When a factor is an `"exact_zero"` plan or the system
            widths differ.
    """
    one, two = access_summary(first), access_summary(second)
    if one.system_qubits != two.system_qubits:
        raise ValueError("product factors must act on the same system width")
    if one.error_bound is None or two.error_bound is None:
        delta = None
    else:
        n1, n2 = (one.alpha + one.error_bound, two.alpha + two.error_bound) if target_norm_bounds is None else (
            finite_real(target_norm_bounds[0], "N_1"), finite_real(target_norm_bounds[1], "N_2"))
        delta = one.error_bound * n2 + (n1 + one.error_bound) * two.error_bound
    rows = [(first, "first", 0, False, 1), (second, "second", 0, False, 1)]
    controlled = [(c, name, 1, adj, n) for c, name, _, adj, n in rows]
    return JointEncodingPlan(
        construction="product", alpha=one.alpha * two.alpha, error_bound=delta,
        num_ancillas=one.num_ancillas + two.num_ancillas, system_qubits=one.system_qubits,
        target_operator_is_hermitian=None, encoded_operator_is_hermitian=None, full_unitary_involution=None,
        layout={"first_ancillas": one.num_ancillas, "second_ancillas": two.num_ancillas,
                "system_qubits": one.system_qubits},
        error_terms={"evidence": "product bound e_1 N_2 + (N_1 + e_1) e_2"},
        query_census=_census(rows, 0, controlled, 0), components={"first": first, "second": second})


def _complex_table_query(table: ComplexDiagonalTable, *, num_controls, ctrl_state, adjoint, max_bytes, max_work):
    """Return C(Phi D) or its adjoint: the magnitude table D, then the phase diagonal Phi on the address."""
    from qiskit import QuantumCircuit
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases

    k = integer(num_controls, "num_controls", 0)
    state = _control_state(ctrl_state, k)
    angles, phases, _ = _complex_table(table)
    r = _table_width(len(table.values))
    _check_bytes(64 * (1 << (r + k)), max_bytes, "complex diagonal phase table")
    magnitude = build_diagonal_table_query(list(np.cos(angles / 2.0) * table.radius), table.radius,
                                           num_controls=k, ctrl_state=state, max_bytes=max_bytes, max_work=max_work)
    circuit = QuantumCircuit(k + 1 + r, name="complex_diagonal_table")
    qubits = list(circuit.qubits)
    controls, signal, address = qubits[:k], qubits[k], qubits[k + 1:]
    sign = -1.0 if adjoint else 1.0
    mask = (1 << k) - 1
    table_phases = [sign * phases[index >> k] if index & mask == state else 0.0 for index in range(1 << (r + k))]
    if adjoint:
        append_control_diagonal_phases(circuit, [*controls, *address], table_phases)
        circuit.append(magnitude, [*controls, signal, *address])
    else:
        circuit.append(magnitude, [*controls, signal, *address])
        append_control_diagonal_phases(circuit, [*controls, *address], table_phases)
    return circuit.to_gate()


def _component_query(component, *, num_controls, ctrl_state, adjoint, max_bytes, max_work):
    """One query of a child, table, recipe or nested plan on [controls, ancillas, system]."""
    component = _supplied(component)
    if isinstance(component, (BlockEncodingPlan, BlockEncoding)):
        return build_block_encoding_query(component, num_controls=num_controls, ctrl_state=ctrl_state,
                                          adjoint=adjoint, max_bytes=max_bytes, max_work=max_work)
    if isinstance(component, UniformNodeRecipe):
        return build_uniform_node_query(component, num_controls=num_controls, ctrl_state=ctrl_state,
                                        adjoint=adjoint, max_bytes=max_bytes, max_work=max_work)
    if isinstance(component, DiagonalTable):
        return build_diagonal_table_query(list(component.values), component.normalization,
                                          num_controls=num_controls, ctrl_state=ctrl_state, adjoint=adjoint,
                                          max_bytes=max_bytes, max_work=max_work)
    if isinstance(component, ComplexDiagonalTable):
        return _complex_table_query(component, num_controls=num_controls, ctrl_state=ctrl_state, adjoint=adjoint,
                                    max_bytes=max_bytes, max_work=max_work)
    if isinstance(component, JointEncodingPlan):
        return _joint_query(component, num_controls=num_controls, ctrl_state=ctrl_state, adjoint=adjoint,
                            max_bytes=max_bytes, max_work=max_work)
    if isinstance(component, ScaledBranchAccess):
        return _scaled_branch_query(component, num_controls=num_controls, ctrl_state=ctrl_state, adjoint=adjoint,
                                    max_bytes=max_bytes, max_work=max_work)
    raise TypeError(f"unsupported block-encoding source {type(component).__name__}")


def _component_query_requirements(component, *, num_controls, adjoint, max_bytes):
    """Return ``(bytes, work, unresolved)`` of the queries that one query of ``component`` builds, as ``_component_query`` does.

    Each table query is checked against ``max_bytes`` by the same law its
    builder applies (``core._native_query_requirements``,
    ``nodes._uniform_node_query_requirements``,
    ``nodes._admit_materialization``), with no work limit. The bytes are
    the largest of those queries, since each builder checks its own query,
    and the work is their sum. A nested plan builds each child query of its
    census once per call, with the plan's external controls added to the
    children that take them: a child of census row i with ``c_0`` controls
    in the forward query and ``c_1`` in the singly controlled one has
    ``c_0 + (c_1 - c_0) k`` controls in a query with k controls, because a
    construction passes a child either all or none of its external controls
    (``_joint_query``). A scaled branch query calls each present child
    once, with the combine qubit as one more control when both parts are
    present (``_scaled_branch_query``). A built encoding, such as a supplied
    one, is queried through the opaque fallback
    (``core._opaque_query_requirements``), whose controlled synthesis is read
    from the definitions that are already built, so planning builds none. A
    definition that is not built leaves its part of that synthesis unknown:
    ``unresolved`` names the first such operation, and the bytes and work
    then cover the known part only, so the work is not a total. A dense-dilation
    child adds nothing here: its synthesis and completion are counted where
    they are planned, as LCHS does in ``_dense_child_syntheses``.
    """
    from nwqlib.subroutines.block_encoding.core import _native_query_requirements, _opaque_query_requirements
    from nwqlib.subroutines.block_encoding.nodes import _admit_materialization, _uniform_node_query_requirements

    component, k = _supplied(component), num_controls
    if isinstance(component, BlockEncodingPlan):
        if component.implementation not in ("multiplexed_pauli", "banded"):
            return 0, 0, None
        adjoint = adjoint and component.detail.get("full_unitary_involution") is not True
        return (*_native_query_requirements(component, num_controls=k, adjoint=adjoint, max_bytes=max_bytes,
                                            max_work=None), None)
    if isinstance(component, BlockEncoding):
        return _opaque_query_requirements(component, num_controls=k, adjoint=adjoint, max_bytes=max_bytes,
                                          max_work=None, built=True)
    if isinstance(component, UniformNodeRecipe):
        return (*_uniform_node_query_requirements(component, num_controls=k, adjoint=adjoint, max_bytes=max_bytes,
                                                  max_work=None)[:2], None)
    if isinstance(component, (DiagonalTable, ComplexDiagonalTable)):
        length = len(component.values)
        data, work = _admit_materialization("table", max_bytes=max_bytes, max_work=None,
                                            address_bits=(length - 1).bit_length(), controls=k,
                                            adjoint=adjoint and isinstance(component, DiagonalTable), length=length)
        if isinstance(component, ComplexDiagonalTable):
            # The magnitude table is built forward, after the phase table of 2**(r + k) entries is checked.
            data = max(data, 64 * (1 << (_table_width(length) + k)))
        return data, work, None
    if isinstance(component, JointEncodingPlan):
        census = component.query_census
        data = work = 0
        unresolved = None
        for (name, forward, child_adjoint, count), (_, controlled, _, _) in zip(
                census["forward_calls"], census["controlled_calls"], strict=True):
            child_data, child_work, child_unresolved = _component_query_requirements(
                component.components[name], num_controls=forward + (controlled - forward) * k,
                adjoint=child_adjoint != adjoint, max_bytes=max_bytes)
            data, work = max(data, child_data), work + count * child_work
            unresolved = unresolved or child_unresolved
        return data, work, unresolved
    if isinstance(component, ScaledBranchAccess):
        parts = _branch_parts(component)[0]
        data = work = 0
        unresolved = None
        for _, child, _, _ in parts:
            child_data, child_work, child_unresolved = _component_query_requirements(
                child, num_controls=int(len(parts) == 2) + k, adjoint=adjoint, max_bytes=max_bytes)
            data, work = max(data, child_data), work + child_work
            unresolved = unresolved or child_unresolved
        return data, work, unresolved
    raise TypeError(f"no query requirements for block-encoding source {type(component).__name__}")


def _joint_query(plan: JointEncodingPlan, *, num_controls, ctrl_state, adjoint, max_bytes, max_work):
    from qiskit import QuantumCircuit

    k = integer(num_controls, "num_controls", 0)
    state = _control_state(ctrl_state, k)
    circuit = QuantumCircuit(k + plan.num_ancillas + plan.system_qubits, name="joint_" + plan.construction)
    controls, rest = list(circuit.qubits[:k]), list(circuit.qubits[k:])

    def query(component, extra=(), extra_state=0, *, external=True, dagger=adjoint):
        """Return a query controlled by ``extra``, then by the external controls when ``external``."""
        wires = [*extra, *(controls if external else ())]
        value = extra_state | ((state << len(extra)) if external else 0)
        gate = _component_query(component, num_controls=len(wires), ctrl_state=value, adjoint=dagger,
                                max_bytes=max_bytes, max_work=max_work)
        return gate, wires

    builder = {"separate_parts": _separate_parts, "single_a": _single_a, "convex_contraction": _convex,
               "product": _product}[plan.construction]
    builder(plan, circuit, controls, state, rest, query, adjoint)
    return circuit.to_gate(label=circuit.name)


def _take(qubits, count):
    return qubits[:count], qubits[count:]


def _separate_parts(plan, circuit, controls, state, rest, query, adjoint):
    # U = P^dagger SELECT P with the real combine P, so U^dagger = P^dagger SELECT^dagger P keeps the shell. The
    # factors of a branch act on disjoint registers, so only the child queries change in the adjoint.
    layout, parts = plan.layout, plan.components
    combine, rest = _take(rest, layout["combine"])
    time_signal, rest = _take(rest, layout["time_signal"])
    node_ancillas, rest = _take(rest, layout["node_ancillas"])
    child_ancillas, rest = _take(rest, layout["child_ancillas"])
    node_address, rest = _take(rest, layout["node_qubits"])
    application, child_system = _take(rest, layout["application_qubits"])
    if parts["time"] is not None:
        gate, wires = query(parts["time"])
        circuit.append(gate, [*wires, *time_signal, *application])
    branches = [name for name in ("l_child", "h_child") if parts.get(name) is not None]
    if parts["theta"] is not None:
        circuit.ry(parts["theta"], combine[0])
    for value, name in enumerate(branches):
        extra = tuple(combine)
        child = parts[name]
        width = _num_ancillas(child)
        if name == "l_child":
            gate, wires = query(parts["nodes"], extra, value)
            circuit.append(gate, [*wires, *node_ancillas, *node_address])
        gate, wires = query(child, extra, value)
        circuit.append(gate, [*wires, *child_ancillas[:width], *child_system])
    if parts["theta"] is not None:
        circuit.ry(-parts["theta"], combine[0])


def _single_a(plan, circuit, controls, state, rest, query, adjoint):
    # U_S = H_h D_B X_h D_B^dagger H_h is Hermitian, and its reversed sequence with every gate inverted is the
    # same sequence, so the adjoint query is the forward construction. D_B = C_{h=0}(B) never carries the
    # external controls: on their inactive branch D_B D_B^dagger = I.
    layout, parts = plan.layout, plan.components
    helper, rest = _take(rest, 1)
    time_signal, rest = _take(rest, layout["time_signal"])
    coefficient_signal, rest = _take(rest, 1)
    child_ancillas, rest = _take(rest, layout["child_ancillas"])
    node_address, rest = _take(rest, layout["node_qubits"])
    application, child_system = _take(rest, layout["application_qubits"])
    factors = [(parts["coefficients"], [*coefficient_signal, *node_address]),
               (parts["a_child"], [*child_ancillas, *child_system])]
    if parts["time"] is not None:
        factors.append((parts["time"], [*time_signal, *application]))
    circuit.h(helper[0])
    for dagger in (True, False):
        for component, wires in factors:
            gate, ctl = query(component, tuple(helper), 0, external=False, dagger=dagger)
            circuit.append(gate, [*ctl, *wires])
        if dagger:
            _append_mcx(circuit, list(controls), state, helper[0])
    circuit.h(helper[0])


def _convex(plan, circuit, controls, state, rest, query, adjoint):
    # U = P^dagger SELECT P with SELECT = |0><0| (x) I + |1><1| (x) (-(c_s (x) U_L)); the adjoint keeps the
    # shell and uses the adjoint child query. The factors of the branch commute.
    from nwqlib.subroutines._multiplexors import append_uniformly_controlled_ry

    layout, parts = plan.layout, plan.components
    combine, rest = _take(rest, layout["combine"])
    scale, rest = _take(rest, layout["scale"])
    child_ancillas, rest = _take(rest, layout["child_ancillas"])
    application, child_system = _take(rest, layout["application_qubits"])
    angles = parts["angles"]
    if angles is not None:
        append_uniformly_controlled_ry(circuit, combine[0], application, list(angles))
    extra = tuple(combine)
    extra_state = 1 if combine else 0
    # The branch sign -1 is an exact phase flip on the active control value; without any control it is
    # (Z X)^2 = -I on a spare qubit, not a rounded global angle pi.
    flip = [*extra, *controls]
    spare = (child_system or scale or child_ancillas)[0]
    _append_phase_flip(circuit, flip, extra_state | (state << len(extra)), spare)
    if parts["scale"] is not None:
        gate, wires = query(parts["scale"], extra, extra_state)
        circuit.append(gate, [*wires, *scale])
    gate, wires = query(parts["l_child"], extra, extra_state)
    circuit.append(gate, [*wires, *child_ancillas, *child_system])
    if angles is not None:
        append_uniformly_controlled_ry(circuit, combine[0], application, list(-angles))


def _product(plan, circuit, controls, state, rest, query, adjoint):
    layout, parts = plan.layout, plan.components
    first_ancillas, rest = _take(rest, layout["first_ancillas"])
    second_ancillas, system = _take(rest, layout["second_ancillas"])
    order = (("second", second_ancillas), ("first", first_ancillas))
    for name, ancillas in (reversed(order) if adjoint else order):
        gate, wires = query(parts[name])
        circuit.append(gate, [*wires, *ancillas, *system])


def _scaled_branch_query(access: ScaledBranchAccess, *, num_controls, ctrl_state, adjoint, max_bytes, max_work):
    """Build ``D(r) (x) V_x`` or its adjoint on [controls, scale flag, combine, child ancillas, system].

    D has no shell, so the external controls control its phase flip and RY.
    In V the real combine RY is never controlled, because it cancels on the
    inactive branch; the children cores and the sign flip of ``k < 0`` carry
    the combine value and the external controls, which keeps the inactive
    branch the identity. The adjoint reverses D, keeps the combine shell and
    uses the adjoint children queries (``V^dagger = P^dagger SELECT^dagger P``).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    k = integer(num_controls, "num_controls", 0)
    state = _control_state(ctrl_state, k)
    summary = _scaled_summary(access)
    parts, theta, _ = _branch_parts(access)
    scale_angle = float(_rotation_table(np.array([access.scale]), 1.0)[0][0])
    if access.angles is not None:
        scale_angle, theta = access.angles[0], access.angles[1] if theta is not None else None
    circuit = QuantumCircuit(k + summary.num_ancillas + summary.system_qubits, name="scaled_branch")
    controls, rest = list(circuit.qubits[:k]), list(circuit.qubits[k:])
    flag, rest = _take(rest, 1)
    combine, rest = _take(rest, int(len(parts) == 2))
    child_ancillas, system = _take(rest, summary.num_ancillas - 1 - len(combine))
    sign = -1.0 if adjoint else 1.0

    def scale_flag():
        # D = R_Y(phi) Z: Z on the flag, as an exact phase flip carrying the controls, then the controlled R_Y.
        rotation = RYGate(sign * scale_angle)
        rotation = rotation.control(k, ctrl_state=state, annotated=False) if k else rotation
        if not adjoint:
            _append_phase_flip(circuit, [flag[0], *controls], 1 | (state << 1), flag[0])
        circuit.append(rotation, [*controls, flag[0]])
        if adjoint:
            _append_phase_flip(circuit, [flag[0], *controls], 1 | (state << 1), flag[0])

    def child(component, extra, extra_state):
        wires = [*extra, *controls]
        gate = _component_query(component, num_controls=len(wires), ctrl_state=extra_state | (state << len(extra)),
                                adjoint=adjoint, max_bytes=max_bytes, max_work=max_work)
        width = _num_ancillas(component)
        circuit.append(gate, [*wires, *child_ancillas[:width], *system])

    if adjoint:
        scale_flag()
    if theta is not None:
        circuit.ry(theta, combine[0])
    for value, (name, component, _, _) in enumerate(parts):
        extra = tuple(combine)
        if name == "l_child" and access.node < 0:
            # sign(k) = -1 on the L branch: an exact flip of that branch's control value.
            _append_phase_flip(circuit, [*extra, *controls], (value if extra else 0) | (state << len(extra)),
                               flag[0])
        child(component, extra, value if extra else 0)
    if theta is not None:
        circuit.ry(-theta, combine[0])
    if not adjoint:
        scale_flag()
    return circuit.to_gate(label=circuit.name)


def build_access_query(source: Source, *, num_controls: int = 0, ctrl_state: int | str | None = None,
                       adjoint: bool = False, max_bytes: int | None = None,
                       max_work: int | None = None):
    r"""Build one query U, U^dagger, C(U) or C(U^dagger) of any source that `access_summary` accepts, as a gate.

    Native plans, node recipes and tables use their own query builders,
    which control only their cores. A `JointEncodingPlan` is assembled from
    those queries: its combine shells (the real RY, the convex-weight RY
    multiplexor, the Hadamards on the Hermitianization helper) are never
    controlled, because they cancel on the inactive control branch, and its
    exact signs are parameter-free phase flips. The adjoint uses the
    adjoint queries of the children in reversed order. A built
    `BlockEncoding` takes the opaque fallback of `build_block_encoding_query`.

    Byte limit: each child query checks its own dominant arrays, such as
    its multiplexor tables, against `max_bytes` before it builds them. The
    child gates that the composite holds together are not summed, so the
    composite can hold a few times that amount.

    Args:
        source: The source to query.
        num_controls (int): Default `0`. Number of external controls.
        ctrl_state (int | str | None): Default `None`, all controls one.
        adjoint (bool): Default `False`. Build the adjoint.
        max_bytes (int | None): Default `None`, the physical memory. Byte
            limit of each child query.
        max_work (int | None): Default `None`, no limit. Construction work limit,
            applied to each child construction.

    Returns:
        gate (qiskit.circuit.Gate): The query on the controls, the
            ancillas and the system qubits, in that order.

    Raises:
        TypeError: For an unsupported source.
        ValueError: For an invalid control state or an exceeded limit.
    """
    return _component_query(source, num_controls=num_controls, ctrl_state=ctrl_state, adjoint=adjoint,
                            max_bytes=max_bytes, max_work=max_work)


def _complete_dense_children(source, *, max_bytes, max_work):
    """Return a joint plan whose dense-dilation children are replaced by their built encodings.

    Each query of a dense-dilation plan builds its completion again, a full
    SVD and the completed unitary (`build_block_encoding_query`). An emitter
    that queries one joint plan in several forms calls this once, so each
    such child is completed once and every query takes the opaque fallback
    of the built `BlockEncoding`. Other sources are returned unchanged. The
    returned plan exists only for that emission and is not saved.
    """
    if not isinstance(source, JointEncodingPlan):
        return source
    built = {name: _build_block_encoding_from_plan(child, max_bytes=max_bytes, max_work=max_work)
             for name, child in source.components.items()
             if isinstance(child, BlockEncodingPlan) and child.implementation == "dense_dilation"}
    return replace(source, components=MappingProxyType({**source.components, **built})) if built else source


def build_joint_encoding(plan: JointEncodingPlan, *, max_bytes: int | None = None,
                         max_work: int | None = None) -> BlockEncoding:
    """Build the uncontrolled circuit of a composite plan as a `BlockEncoding`.

    The built encoding is one opaque circuit. Passed to `build_access_query`
    or to the evolution builders of `nwqlib.subroutines.qsp`, it is
    controlled gate by gate, including the combine shells that a structured
    query leaves uncontrolled. Structured controlled queries, and
    `JointEncodingPlan.query_census` that counts them, come from passing the
    plan itself to `build_access_query` and to those builders.

    Args:
        plan (JointEncodingPlan): A plan from this module.
        max_bytes (int | None): Default `None`, the physical memory. As for
            `build_access_query`.
        max_work (int | None): Default `None`, no limit. As for `build_access_query`.

    Returns:
        encoding (BlockEncoding): The circuit with the plan's normalization,
        widths, error bound and three structural facts in its metadata, and
        the plan's layout and error terms.
    """
    from qiskit import QuantumCircuit

    gate = build_access_query(plan, max_bytes=max_bytes, max_work=max_work)
    circuit = QuantumCircuit(gate.num_qubits, name=gate.label or "joint")
    circuit.append(gate, circuit.qubits)
    record = plan.to_dict()
    return BlockEncoding(circuit=circuit, alpha=plan.alpha, num_ancillas=plan.num_ancillas,
                         system_qubits=plan.system_qubits, error_bound=plan.error_bound,
                         implementation="joint_" + plan.construction,
                         metadata={**{name: getattr(plan, name) for name in STRUCTURE_FACT_KEYS},
                                   "layout": record["layout"], "error_terms": record["error_terms"],
                                   "error_bound_evidence": str(plan.error_terms.get("evidence", ""))})


__all__ = [
    "AccessSummary",
    "ComplexDiagonalTable",
    "DiagonalTable",
    "JointEncodingPlan",
    "ScaledBranchAccess",
    "access_summary",
    "build_access_query",
    "build_joint_encoding",
    "plan_block_encoding_product",
    "plan_convex_contraction",
    "plan_joint_generator",
    "plan_single_a_generator",
]
