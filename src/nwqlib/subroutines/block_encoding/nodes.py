r"""Diagonal block encodings of quadrature nodes on an address register.

A joint LCHS generator multiplies the child encoding on each address label
nu by a node value, so it needs a block encoding of a real diagonal
``diag(v_nu)`` on the address register. This module gives four
constructions. Each circuit is a full involution: its whole unitary U
satisfies ``U = U^dagger`` and ``U**2 = I``, so a quantum walk can use it
directly.

Angle table. For values ``x_nu = v_nu / lambda``
in [-1, 1] and one signal qubit z,

    \[D=\bigoplus_\nu R_Y(2\arccos x_\nu)\,Z
       =\bigoplus_\nu\begin{pmatrix}x_\nu&\sqrt{1-x_\nu^2}\\
       \sqrt{1-x_\nu^2}&-x_\nu\end{pmatrix},\qquad D^2=I,\]

since ``Z R_Y(t) Z = R_Y(-t)`` for every angle t. The identity holds for the
stored angles ``theta_nu = fl(2 arccos fl(v_nu / lambda))`` too, but their
block value ``cos(theta_nu / 2)`` need not equal the intended
``v_nu / lambda``: the target residual is
``max_nu |v_nu - lambda cos(theta_nu / 2)|``, padding included, which
contains both the rounded division and the rounded angle. Even ``x = 0``
moves, because the stored ``fl(pi)`` lies about 1.2e-16 below pi. The
table builders evaluate this residual from the angles they store and
report it as the error bound, a binary64 estimate rather than an
enclosure. ``x = 1, 0, -1`` give ``Z``, ``X`` and ``-Z`` up to that
residual.

Uniform nodes ``k_j = h j``, ``-J <= j <= J``, ``M = 2J + 1``, use one of
two label mappings, applied when the outer coefficient PREP is built, so
no circuit permutes labels:

* two's complement on ``r = ceil(log2(2J + 1))`` bits, label ``j mod 2**r``;
* signed magnitude on ``p + 1`` bits, ``p = ceil(log2(J + 1))``, label
  ``|j| + 2**p [j < 0]``. Physical zero is label 0; label ``2**p``, the
  negative zero, is padding.

For J >= 1 both widths are r, because 2J + 1 is odd. Padding labels have
zero amplitude in the outer PREP and SELECT keeps the label, so the encoded
diagonal can take any bounded value there (its completion). The physical
radius ``R = hJ`` is the kernel cutoff, while the encoding normalization
``alpha = h N`` covers the completion too, with N the completion extent, and
the two can differ.

* ``rotation_table``: the angle table on the two's-complement labels with the values
  ``h j`` and completion 0; ``N = J``.
* ``comparator``: with ``Uni`` the uniform superposition of
  ``l = 0, ..., J-1`` and C the reversible comparison
  ``f = [l < |j|]`` on signed-magnitude labels, ``B = C Uni`` and
  ``K = Z_sign C_{f=0} X_z``. The two factors of K commute and square to
  I, so ``U = B^dagger K B`` is a full involution. On the all-zero work
  and signal block, ``<0|U|0>(j) = (-1)^sign min(|j|, J) / J``, the
  probability of ``f = 1`` being ``min(|j|, J)/J``; physical labels give
  ``j / J`` and padding the clipped value. ``N = J``.
* ``twos_complement_pauli``:
  ``j = -1/2 I + 2**(r-2) Z_{r-1} - sum_{l<r-1} 2**(l-1) Z_l``, ``r + 1``
  terms, ``N = 2**(r-1)``; the most negative label ``-2**(r-1)`` is
  the reason N exceeds J.
* ``signed_magnitude_pauli``:
  ``j = (2**p - 1)/2 Z_s - sum_{l<p} 2**(l-1) Z_s Z_l``, ``p + 1`` terms,
  ``N = 2**p - 1``. The physical zero is one label; the other zero is
  padding.

The two Pauli recipes are real Pauli sums, so they use the multiplexed-Pauli
construction of ``core`` and its sign-folded involution. Their
coefficients are the exact dyadic numbers above for the integer diagonal
j, and the spacing h enters only alpha, so the block is ``j / N`` with no
rounding of h. The comparator's probability mechanism is that of Low and
Somma, arXiv:2508.19238v2, Lemma 18, here realized as ``B^dagger K B``.

Normalizations. The comparator and Pauli blocks are exactly ``j / N`` in
the ideal circuit, so with the stored ``alpha = fl(h N)`` the target
``h j`` is missed by ``|j| |h N - alpha| / N <= |h N - alpha|``, which
their encodings report, computed exactly from the integer ratio of h and
rounded up. The products ``h J`` and ``h N`` are formed exactly in integers,
so a finite ``h N`` is never lost to an intermediate ``float(N)`` or
``2.0**p`` overflow. A recipe whose ``h N`` overflows binary64 is not
offered, while a recipe of smaller N still is. The integer Pauli plan
stores its coefficients and its alpha in binary64: ``2**(p-1)`` and
``2**p`` exist for ``p <= 1023``, while ``(2**p - 1)/2`` is exact only for
``p <= 53``. Beyond these the Pauli recipe is not offered, rather than a
recipe whose coefficients are not those above.

``uniform_node_recipes`` returns the offered recipes as data with their
widths, normalizations and gate counts, without building a circuit, so a
caller can compare them and build only the selected one with
``build_uniform_node_query``.

Size limits. The label and value grids and the angle tables grow with
``2**r`` and, under k external controls, ``2**(r + k)``. Each public
function that forms them takes ``max_bytes`` and ``max_work`` and refuses
before any conversion, grid, table or circuit
(``_admit_materialization`` states the byte and work formulas). The
comparator needs no grid, and the Pauli recipes pass the limits to the
construction of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Mapping

import numpy as np

from nwqlib._limits import ResourceLimitError, exceeds, optional_limit
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _array_input
from nwqlib.subroutines._multiplexors import append_uniformly_controlled_ry, multiplexor_resource_law
from nwqlib.subroutines.block_encoding.core import (
    BlockEncoding,
    BlockEncodingPlan,
    STRUCTURE_FACT_KEYS,
    _admitted_power,
    _append_mcx,
    _append_phase_flip,
    _control_state,
    _controlled_table,
    _mcx_cx,
    _pauli_planning_requirements,
    _query_census,
    _UnknownMCXCount,
    block_encoding_query_cx,
    build_block_encoding_query,
)


@dataclass(frozen=True, kw_only=True)
class UniformNodeRecipe:
    """One way to encode the uniform nodes ``k_j = h j`` as a diagonal, described without building a circuit.

    Attributes:
        kind: `"rotation_table"`, `"comparator"`, `"twos_complement_pauli"` or
            `"signed_magnitude_pauli"` (module docstring).
        half_width: J, the number of positive nodes; `j` runs over `-J..J`.
        spacing: h, the node spacing in the units of k.
        mapping: Label mapping of the address register, `"twos_complement"`
            or `"signed_magnitude"`.
        address_qubits: r, the width of the address register on which the
            diagonal acts.
        encoding_ancillas: Signal and work qubits that start and end in
            zero on the encoded block.
        alpha: Encoding normalization `h N` in the units of k, rounded to
            the nearest binary64 value, with N the completion extent: J for
            the rotation table and the comparator, `2**p` for the
            two's-complement and `2**p - 1` for the signed-magnitude Pauli
            recipe, with `p = r - 1`. The block is `diag(v)/alpha`, where `v` is the node value
            on a physical label and the completion on padding.
        physical_radius: R = hJ, the largest physical node magnitude, rounded
            to the nearest binary64 value.
        pauli_terms: Number of Pauli terms for the two Pauli recipes, else 0.
        query_census: CX counts of the forward, adjoint, controlled (one
            control) and controlled-adjoint queries that
            `build_uniform_node_query` builds from this recipe, of the
            never-controlled shell and of the core, the work qubits and the
            reflections (none).
        plan: The multiplexed-Pauli `BlockEncodingPlan` of the integer
            diagonal for the two Pauli recipes, else None.
    """

    kind: Literal["rotation_table", "comparator", "twos_complement_pauli", "signed_magnitude_pauli"]
    half_width: int
    spacing: float
    mapping: Literal["twos_complement", "signed_magnitude"]
    address_qubits: int
    encoding_ancillas: int
    alpha: float
    physical_radius: float
    pauli_terms: int
    query_census: Mapping[str, Any]
    plan: BlockEncodingPlan | None = None


def _scaled(spacing: float, count: int) -> float | None:
    """Return ``h N`` rounded to the nearest binary64 value, or None when it is not finite.

    The product is formed exactly from the integer ratio of h, and Python's
    integer division rounds it once.
    """
    try:
        return float(Fraction(spacing) * count)
    except OverflowError:
        return None


def _normalization_error(spacing: float, count: int, alpha: float) -> float:
    """Return ``|h N - alpha|`` rounded up: the target error of an exact block ``j / N`` scaled by the stored alpha."""
    exact = abs(Fraction(spacing) * count - Fraction(alpha))
    bound = float(exact)
    return bound if Fraction(bound) >= exact else float(np.nextafter(bound, np.inf))


def _node_pauli_plan(half_width: int, mapping: str, *, max_bytes: int, max_work: int):
    """Plan the multiplexed-Pauli encoding of the integer diagonal j on r bits (two's-complement or signed-magnitude labels).

    The coefficients are the exact dyadic numbers of the module docstring.
    Qiskit labels list qubit 0 last; qubit r - 1 is the two's-complement
    top bit or the signed-magnitude sign.
    """
    from nwqlib.subroutines.block_encoding.core import plan_block_encoding
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    r = (2 * half_width).bit_length()

    def z(*qubits):
        return "".join("Z" if r - 1 - index in qubits else "I" for index in range(r))

    if mapping == "twos_complement":
        terms = [("I" * r, -0.5), (z(r - 1), float(2 ** (r - 2)))]
        terms += [(z(bit), -float(2.0 ** (bit - 1))) for bit in range(r - 1)]
    else:
        p = r - 1
        terms = [(z(p), (2.0**p - 1.0) / 2.0)]
        terms += [(z(p, bit), -float(2.0 ** (bit - 1))) for bit in range(p)]
    decomposition = PauliDecomposition(
        terms=tuple(PauliTerm(label=label, coefficient=coefficient) for label, coefficient in terms),
        input_dimension=1 << r, operator_dimension=1 << r, num_qubits=r, atol=0.0)
    return plan_block_encoding(decomposition, implementation="multiplexed_pauli", max_bytes=max_bytes,
                               max_work=max_work)


def _uniform_preparation_slots(count: int) -> int:
    """Elementary gate slots of Qiskit's UniformSuperpositionGate over ``0, ..., count-1``, ``count >= 2``.

    A power of two is n H gates on its n qubits. Otherwise the bound sums
    ``_preparation_laws._uniform_superposition_gate_slots``, ``17n - 23``.
    """
    from nwqlib._preparation_laws import _uniform_superposition_gate_slots

    n = (count - 1).bit_length()
    return n if count & (count - 1) == 0 else sum(_uniform_superposition_gate_slots(n).values())


def _uniform_preparation_cx(count: int) -> int:
    """CX bound of Qiskit's UniformSuperpositionGate over ``0, ..., count-1``: none for a power of two, else ``3n - 5`` on n qubits.

    The power-of-two case is n H gates; the other bound is
    ``_preparation_laws._uniform_superposition_gate_slots``.
    """
    from nwqlib._preparation_laws import _uniform_superposition_gate_slots

    if count & (count - 1) == 0:
        return 0
    return _uniform_superposition_gate_slots((count - 1).bit_length())["cx"]


def _table_census(address_qubits: int) -> dict[str, Any]:
    """Return the query census of the angle table on a address qubits.

    The core is the RY multiplexor, ``2**a`` CX (``multiplexor_resource_law``),
    and the Z on the signal; one control adds a bit to the multiplexor and
    turns Z into a phase flip, an X with one control.
    """
    def ucry(controls):
        return multiplexor_resource_law("ucry", control_qubits=controls)["basis_cx_gates"]

    return _query_census(shell="none", shell_cx=0, core_cx=ucry(address_qubits),
                         controlled_core_cx=ucry(address_qubits + 1) + _mcx_cx(1),
                         controlled_core_bound_cx=_mcx_cx(1))


def _comparator_cx(magnitude_bits: int) -> int:
    """CX of the comparison C on p magnitude bits: 2p MAJ gates of two CX and one Toffoli (6 CX), and one CX to the flag."""
    return 16 * magnitude_bits + 1


def angle_table_cx(address_qubits: int, num_controls: int) -> int:
    r"""CX of one angle-table query (`_diagonal_table_circuit`) on r address qubits with k external controls.

    The circuit is a phase flip on ``[signal, controls]`` and an RY
    multiplexor with ``r + k`` controls, so with ``U(s)`` the UCRY law
    (``2**s`` for ``s > 0``, `multiplexor_resource_law`) and ``M(k)`` the
    CX of an X with k controls (`_mcx_cx`),

        \[C_{\rm table}(k)=U(r+k)+M(k).\]

    The same circuit serves a `DiagonalTable` query and the
    ``"rotation_table"`` node recipe (M7.3 of the M7 mathematics handoff).
    Reversal has the same count, and open controls add only X gates.

    Raises:
        _UnknownMCXCount: Beyond the tabulated MCX control counts.
    """
    ucry = multiplexor_resource_law("ucry", control_qubits=address_qubits + num_controls)["basis_cx_gates"]
    return ucry + _mcx_cx(num_controls)


def uniform_node_query_cx(recipe: UniformNodeRecipe, num_controls: int) -> int | None:
    r"""CX upper bound of one query of a uniform node recipe with k external controls, or None.

    The bound holds for the forward and the adjoint query and for every
    control state (reversal keeps the count, open controls add X gates).

    * ``"rotation_table"``: `angle_table_cx` with ``r = address_qubits``.
    * ``"comparator"``: the conjugating shell ``Uni^dagger C K C Uni`` is
      never controlled; for ``J > 1`` it is two uniform preparations and two
      comparisons, ``2 C_Uni(J) + 2(16p + 1)`` with ``p = r - 1``, and the
      ``J = 1`` builder has no shell. The core K is a phase flip on the sign
      plus the k controls and an X on the signal controlled by the test bit
      plus those controls, so (M7.4)

          \[C_{\rm cmp}(k)=C_{\rm shell}+M(k+1)+M(k).\]

      ``k = 0`` and ``k = 1`` give the stored census.
    * Pauli recipes: `block_encoding_query_cx` of their integer plan.

    These are the qualified upper bounds of the pinned constructions, with
    Qiskit's multi-controlled X counts without ancillas and the PREP slot
    bounds; a lowering that borrows idle qubits can undercut them.

    Args:
        recipe (UniformNodeRecipe): From `uniform_node_recipes`.
        num_controls (int): k >= 0.

    Returns:
        cx (int | None): The bound, None beyond the tabulated MCX counts.
    """
    k = integer(num_controls, "num_controls", 0)
    try:
        if recipe.plan is not None:
            return block_encoding_query_cx(recipe.plan, num_controls=k)
        if recipe.kind == "rotation_table":
            return angle_table_cx(recipe.address_qubits, k)
        return recipe.query_census["shell_cx"] + _mcx_cx(k + 1) + _mcx_cx(k)
    except _UnknownMCXCount:
        return None


def _admit_materialization(kind: str, *, max_bytes, max_work, address_bits: int, controls: int = 0,
                           adjoint: bool = False, length: int = 0, half_width: int = 0, terms: int = 0) -> tuple[int, int]:
    r"""Refuse a node materialization whose known data bytes B or nominal work W exceed the limits, before it starts, and return ``(B, W)``.

    All sizes are Python integers; ``_admitted_power`` admits the largest
    power of two from bit lengths before forming it. With r address bits,
    k external controls, ``P = 2**r``, ``s = r + k``, ``Q = 2**s`` table
    entries, ``M = 2J + 1`` nodes, L explicit values and ``A = 1`` for an
    adjoint (its reversed copy coexists with the forward gate):

    * ``labels``: ``B = 40M``, ``W = 6M``. Signed magnitude holds j, |j|,
      the shifted sign, the result (eight bytes each) and a Boolean mask.
    * ``values``: ``B = 64P``, ``W = 16P``: at most seven eight-byte P
      arrays (labels, magnitude, sign, products and result).
    * ``table`` (explicit values):
      ``B = (1+A)[16L + 64P + 192Q + 8(s+2)]``,
      ``W = (1+A)[16L + 16P + 16Q(s+2) + 8(k+1)]``.
    * ``rotation_table`` (uniform nodes):
      ``B = (1+A)[128P + 192Q + 8(s+2)]``,
      ``W = (1+A)[32P + 16Q(s+2) + 8(k+1)]``.

      For both tables, 16L covers the converted values and their checks and
      64P the padded target, the quotients, the stored angles and the
      residual pass (128P adds the integer node grid of
      ``uniform_node_values``). 192Q is 24 eight-byte slots per expanded
      entry: the expanded angle table, the Walsh transform input, output
      and half-size butterfly copies, Gray-code indices, the transformed
      angles and control schedule, and the gate parameters of the forward
      and reversed circuits. ``gray_code_rotation_angles`` makes s butterfly
      levels over Q entries, hence ``16Q(s+2)`` work, which also covers at
      most Q RY and Q CX appends.
    * ``comparator``: ``B = 64(1+A)(p+k+16)``,
      ``W = (1+A)[G + 2U + 4k + 2 + 8(p+k+2)]`` with ``p = J.bit_length()``,
      ``G = 16p + 8`` instructions (4 for J = 1) and U the slots of the
      uniform superposition (``_uniform_preparation_slots``, 0 for J = 1),
      counted twice for Uni and its inverse; 4k + 2 open-control X gates;
      the lists of the MAJ chain, registers and controls. No grid of J
      values exists.
    * ``pauli`` with m terms, ``a = ceil(log2 m)``, ``T = 2**a`` and
      ``T_k = 2**(a + k)``: the stored plan holds its terms,
      ``H = m(r + 16)``, and ``B = H + 64(4 + 2A) T_k (r + 16)``, the envelope that
      ``core._admit_native_query`` checks for the same query, with A = 0
      for these integer diagonal plans, which are involutions (the query
      tables and their gate copy, for a reversed adjoint the realized multiplexors
      and their reversal, and Qiskit's stored copies of the realized,
      reversed and zero-support one-qubit matrices), with H counted once;
      ``W = (1+A)[32 T_k max(1, r)**2 + 16T(a + 1)]``, the query tables of
      ``core._native_query_circuit`` and the PREP intake of
      ``lcu.data._lcu_prep_stage_requirements``. The child query receives
      the same ``max_bytes``.

    Each formed power of two adds its byte length, about ``(s + 8) // 8``.
    B counts known numerical and index payload only: Python object headers,
    circuit objects and Qiskit's synthesis workspace are excluded, so it is
    no bound on process memory. W is a nominal count of construction
    steps, not a runtime. The label, value and node-table grids hold int64
    labels below ``2**r``, and every table length must index a NumPy array,
    so those kinds also need ``r <= 62`` and ``s <= 62``. Planning that knows
    a query's controls reads B and W with ``max_work=None``
    (``joint._component_query_requirements``).
    """
    _check_bytes(0, max_bytes)
    max_work = optional_limit(max_work, "max_work")
    operation = f"uniform node {kind}" if kind != "table" else "diagonal angle table"
    copies = 2 if adjoint else 1
    r, k = address_bits, controls
    if kind == "labels":
        nodes = 2 * half_width + 1
        data, work, bits = 40 * nodes, 6 * nodes, r
    elif kind == "values":
        size = _admitted_power(r, 64, 16, max_bytes, max_work, operation)
        data, work, bits = 64 * size + (r + 8) // 8, 16 * size, r
    elif kind in ("table", "rotation_table"):
        s = r + k
        if s > 62:
            raise ValueError(f"{operation} needs 2**{s} entries, beyond the int64 index range")
        entries = _admitted_power(s, 192 * copies, 16 * (s + 2) * copies, max_bytes, max_work, operation)
        size = 1 << r
        if kind == "table":
            data = copies * (16 * length + 64 * size + 192 * entries + 8 * (s + 2))
            work = copies * (16 * length + 16 * size + 16 * entries * (s + 2) + 8 * (k + 1))
        else:
            data = copies * (128 * size + 192 * entries + 8 * (s + 2))
            work = copies * (32 * size + 16 * entries * (s + 2) + 8 * (k + 1))
        data, bits = data + 3 * ((s + 8) // 8), r
    elif kind == "comparator":
        p = half_width.bit_length()
        slots = 0 if half_width == 1 else _uniform_preparation_slots(half_width)
        instructions = 4 if half_width == 1 else 16 * p + 8
        data = 64 * copies * (p + k + 16) + (p + 8) // 8 + (k + 8) // 8
        work = copies * (instructions + 2 * slots + 4 * k + 2 + 8 * (p + k + 2))
        bits = 0
    else:
        a = (terms - 1).bit_length()
        table = 1 << a
        held = terms * (r + 16)
        expanded = _admitted_power(a + k, 128 * (copies + 1) * (r + 16), 32 * copies * max(1, r)**2, max_bytes,
                                   max_work, operation)
        data = held + 128 * (copies + 1) * expanded * (r + 16) + (a + k + 8) // 8
        work = copies * (32 * expanded * max(1, r)**2 + 16 * table * (a + 1))
        bits = 0
    _check_bytes(data, max_bytes, operation)
    if exceeds(work, max_work):
        raise ResourceLimitError(f"{operation} needs {work} work units, exceeding max_work={max_work}")
    if bits > 62:
        raise ValueError(f"{operation} needs {bits}-bit labels, beyond the int64 range")
    return data, work


def uniform_node_recipes(half_width: int, spacing: float, *, max_bytes: int | None = None,
                         max_work: int | None = None) -> tuple[UniformNodeRecipe, ...]:
    """Return the diagonal encodings of the uniform nodes `k_j = h j`, `-J <= j <= J`, without building a circuit.

    The recipes are `"rotation_table"`, `"comparator"`,
    `"twos_complement_pauli"` and `"signed_magnitude_pauli"`.
    [Node encodings](../../algorithms/lchs_derivations.md#node-encodings)
    gives their formulas, label mappings and completions. Each record holds the
    address width, the work qubits, the normalization `alpha` (which can
    exceed the physical radius `hJ`) and the CX counts of one query with
    and without one control. The counts are exact for the multiplexors and
    comparisons and use the slot bounds of PREP trees and of a uniform
    superposition whose count is not a power of two. A caller compares them
    with its own query counts and builds only the selected recipe with
    `build_uniform_node_query`.

    The rotation table and the comparator are always returned, in that
    order; their normalization is the physical radius. A Pauli recipe
    follows when it is available: its normalization `h N` is a finite
    binary64 value, its integer plan has the exact binary64 coefficients of
    those Pauli forms (`p <= 1023` for the two's-complement recipe,
    `p <= 53` for the signed-magnitude one, `p = J.bit_length()`). The limits
    never remove a recipe: a Pauli plan that exceeds them raises. The work is
    O(log J) plus planning the Pauli encodings of about log2(J) terms on r
    qubits. Each plan keeps the `m(r + 16)` bytes of its m terms in the
    returned recipe and is planned with the arrays and work that
    `plan_block_encoding` checks for a Pauli decomposition.

    Args:
        half_width (int): J, at least 1.
        spacing (float): h, finite and positive, in the units of k.
        max_bytes (int | None): Default `None`, the physical memory.
            Limit on the data held by the Pauli plans.
        max_work (int | None): Default `None`, no limit. Limit on the total
            planning work of the Pauli plans.

    Returns:
        recipes (tuple[UniformNodeRecipe, ...]): The available recipes in
            the order above.

    Raises:
        ValueError: If `half_width` is not a positive integer, `spacing` is
            not finite and positive, `hJ` exceeds the binary64 range, a
            limit is neither None nor a positive integer, or a Pauli plan
            exceeds a limit.
    """
    J = integer(half_width, "half_width", 1)
    h = finite_real(spacing, "spacing")
    if h <= 0.0:
        raise ValueError("spacing must be positive")
    _check_bytes(0, max_bytes)
    work_limit = optional_limit(max_work, "max_work")
    radius = _scaled(h, J)
    if radius is None:
        raise ValueError(f"the physical radius h J of spacing {h!r} and half_width {J} exceeds the binary64 range")
    r = (2 * J).bit_length()
    p = r - 1
    table = _table_census(r)
    # K is an X on the signal with one control (the flag or m_0) and the sign
    # phase flip; one external control adds one control to each.
    core = {"core_cx": _mcx_cx(1), "controlled_core_cx": _mcx_cx(2) + _mcx_cx(1),
            "core_bound_cx": _mcx_cx(1), "controlled_core_bound_cx": _mcx_cx(2) + _mcx_cx(1)}
    if J == 1:
        comparator = _query_census(shell="none", shell_cx=0, **core)
        comparator_ancillas = 1
    else:
        uniform = 2 * _uniform_preparation_cx(J)
        comparator = _query_census(shell="uniform superposition and comparison",
                                   shell_cx=uniform + 2 * _comparator_cx(p), shell_bound_cx=uniform,
                                   workspace_qubits=p + 2, **core)
        comparator_ancillas = p + 3
    recipes = [
        UniformNodeRecipe(kind="rotation_table", half_width=J, spacing=h, mapping="twos_complement",
                          address_qubits=r, encoding_ancillas=1, alpha=radius, physical_radius=radius,
                          pauli_terms=0, query_census=table),
        UniformNodeRecipe(kind="comparator", half_width=J, spacing=h, mapping="signed_magnitude",
                          address_qubits=r, encoding_ancillas=comparator_ancillas, alpha=radius,
                          physical_radius=radius, pauli_terms=0, query_census=comparator),
    ]
    held = work = 0
    for kind, mapping, extent, terms, exact in (
            ("twos_complement_pauli", "twos_complement", 1 << p, r + 1, p <= 1023),
            ("signed_magnitude_pauli", "signed_magnitude", (1 << p) - 1, r, p <= 53)):
        alpha = _scaled(h, extent)
        data = terms * (r + 16)
        arrays, planning = _pauli_planning_requirements(r, terms)
        if alpha is None or not exact:
            continue
        _check_bytes(held + data + arrays, max_bytes, f"uniform node {kind} plans")
        if exceeds(work + planning, work_limit):
            raise ResourceLimitError(f"uniform node {kind} plans need {work + planning} work units, "
                             f"exceeding max_work={work_limit}")
        # The plan encodes the integer diagonal, so its alpha is exactly N.
        plan = _node_pauli_plan(J, mapping, max_bytes=max_bytes,
                                max_work=None if work_limit is None else work_limit - work)
        held, work = held + data, work + planning
        recipes.append(UniformNodeRecipe(
            kind=kind, half_width=J, spacing=h, mapping=mapping, address_qubits=r,
            encoding_ancillas=plan.num_ancillas, alpha=alpha, physical_radius=radius,
            pauli_terms=len(plan.decomposition.terms), query_census=dict(plan.detail["query_census"]), plan=plan))
    return tuple(recipes)


def _node_integers(recipe: UniformNodeRecipe) -> np.ndarray:
    """Return the signed integer j that each address label represents under the recipe's mapping."""
    labels = np.arange(1 << recipe.address_qubits, dtype=np.int64)
    if recipe.mapping == "twos_complement":
        return np.where(labels >= 1 << (recipe.address_qubits - 1), labels - (1 << recipe.address_qubits), labels)
    p = recipe.address_qubits - 1
    magnitude = labels & ((1 << p) - 1)
    return np.where(labels >> p, -magnitude, magnitude)


def uniform_node_labels(recipe: UniformNodeRecipe, *, max_bytes: int | None = None,
                        max_work: int | None = None) -> np.ndarray:
    """Return the address label of each physical node `j = -J, ..., J`, in that order.

    Two's complement gives `j mod 2**r`; signed magnitude gives
    `|j| + 2**p [j < 0]`. The outer coefficient PREP places the amplitude
    of node j on this label.

    Args:
        recipe (UniformNodeRecipe): A recipe from `uniform_node_recipes`.
        max_bytes (int | None): Default `None`, the physical memory.
            Limit on the arrays, `40 (2J + 1)` bytes.
        max_work (int | None): Default `None`, no limit. Limit on the work,
            `6 (2J + 1)` units.

    Returns:
        labels (np.ndarray): `2J + 1` int64 labels.

    Raises:
        ValueError: If the arrays or the work exceed the limits, or a label
            needs more than 62 bits.
    """
    _admit_materialization("labels", max_bytes=max_bytes, max_work=max_work, address_bits=recipe.address_qubits,
                           half_width=recipe.half_width)
    j = np.arange(-recipe.half_width, recipe.half_width + 1, dtype=np.int64)
    if recipe.mapping == "twos_complement":
        return j % (1 << recipe.address_qubits)
    return np.abs(j) + ((j < 0) << (recipe.address_qubits - 1))


def _uniform_values(recipe: UniformNodeRecipe) -> np.ndarray:
    """Return ``h j`` on the physical labels and the recipe's completion elsewhere, in binary64."""
    J, h = recipe.half_width, recipe.spacing
    if recipe.kind == "comparator":
        labels = np.arange(1 << recipe.address_qubits)
        magnitude = labels & ((1 << (recipe.address_qubits - 1)) - 1)
        sign = np.where(labels >> (recipe.address_qubits - 1), -1.0, 1.0)
        return h * sign * np.minimum(magnitude, J)
    j = _node_integers(recipe)
    if recipe.kind == "rotation_table":
        return np.where(np.abs(j) <= J, h * j, 0.0)
    return h * j.astype(float)


def uniform_node_values(recipe: UniformNodeRecipe, *, max_bytes: int | None = None,
                        max_work: int | None = None) -> np.ndarray:
    """Return the encoded diagonal on every address label in the units of k: `h j` on physical labels, the completion elsewhere.

    The block of the recipe's circuit is this array divided by `alpha`. The
    completions are 0 for the rotation table, `h (-1)^sign min(|j|, J)` for
    the comparator and `h j` over the whole label range for the two Pauli
    recipes. The array has `2**r` entries, about four times the node count.

    Args:
        recipe (UniformNodeRecipe): A recipe from `uniform_node_recipes`.
        max_bytes (int | None): Default `None`, the physical memory.
            Limit on the arrays, `64 * 2**r` bytes.
        max_work (int | None): Default `None`, no limit. Limit on the work,
            `16 * 2**r` units.

    Returns:
        values (np.ndarray): `2**r` binary64 values.

    Raises:
        ValueError: If the arrays or the work exceed the limits, or a label
            needs more than 62 bits.
    """
    _admit_materialization("values", max_bytes=max_bytes, max_work=max_work, address_bits=recipe.address_qubits)
    return _uniform_values(recipe)


def _rotation_table(values: np.ndarray, normalization: float) -> tuple[np.ndarray, float]:
    """Return the stored angles of the angle table of ``values`` and the binary64 value of its target residual.

    With ``v`` padded by zeros to a power of two, the angles
    ``theta = fl(2 arccos fl(v / lambda))`` are computed once, here, for
    both the circuit and the residual ``max |v - lambda cos(theta / 2)|``
    (module docstring). ``|v| <= lambda`` keeps every quotient in [-1, 1].
    """
    target = np.zeros(1 << (len(values) - 1).bit_length())
    target[:len(values)] = values
    angles = 2.0 * np.arccos(target / normalization)
    return angles, float(np.max(np.abs(target - normalization * np.cos(angles / 2.0))))


def _diagonal_table_circuit(angles: np.ndarray, *, num_controls: int, ctrl_state: int):
    """Return the angle table D on [external controls, signal, address], its core controlled on ``ctrl_state``.

    The circuit applies Z to the signal z and then the RY multiplexor of the
    stored angles. With k controls, Z becomes a phase flip on the signal and
    controls and the multiplexor table is zero unless the controls hold
    ``ctrl_state``, which is ``C(R_Y Z) = C(R_Y) C(Z)``. The table has no
    shell.
    """
    from qiskit import QuantumCircuit, QuantumRegister

    address_qubits = (len(angles) - 1).bit_length()
    registers = [QuantumRegister(num_controls, "control")] if num_controls else []
    signal = QuantumRegister(1, "signal")
    address = QuantumRegister(address_qubits, "address")
    circuit = QuantumCircuit(*registers, signal, address,
                             name="c_diagonal_rotation_table" if num_controls else "diagonal_rotation_table")
    external = list(circuit.qubits[:num_controls])
    _append_phase_flip(circuit, [signal[0], *external], 1 | (ctrl_state << 1), signal[0])
    append_uniformly_controlled_ry(circuit, signal[0], [*external, *address],
                                   _controlled_table(angles.tolist(), address_qubits, num_controls, ctrl_state, 0.0))
    return circuit


def _diagonal_table(values, normalization, *, num_controls, ctrl_state, adjoint, max_bytes, max_work):
    """Admit, convert and check a real diagonal; return its stored angles, target residual, normalization and control state."""
    k = integer(num_controls, "num_controls", 0)
    if type(values) is np.ndarray:
        if values.ndim != 1:
            raise ValueError("diagonal values must be a nonempty finite one-dimensional array")
        length = values.size
    elif type(values) in (list, tuple):
        length = len(values)
    else:
        raise TypeError("diagonal values must be an ordinary ndarray, list or tuple")
    if not length:
        raise ValueError("diagonal values must be a nonempty finite one-dimensional array")
    _admit_materialization("table", max_bytes=max_bytes, max_work=max_work, address_bits=(length - 1).bit_length(),
                           controls=k, adjoint=adjoint, length=length)
    state = _control_state(ctrl_state, k)
    v = _array_input(values, ndim=1, max_bytes=max_bytes)
    if v.dtype.kind == "c":
        raise ValueError("diagonal values must be real")
    lam = finite_real(normalization, "normalization")
    if not np.all(np.isfinite(v)):
        raise ValueError("diagonal values must be a nonempty finite one-dimensional array")
    if lam <= 0.0 or float(np.max(np.abs(v))) > lam:
        raise ValueError("normalization must be positive and at least the largest value magnitude")
    angles, residual = _rotation_table(v, lam)
    return angles, residual, lam, state


def build_diagonal_table_query(values, normalization: float, *, num_controls: int = 0,
                               ctrl_state: int | str | None = None, adjoint: bool = False,
                               max_bytes: int | None = None, max_work: int | None = None):
    """Build the angle-table encoding of a real diagonal, with optional controls, as a gate.

    The gate is `D = (+)_nu R_Y(theta_nu) Z` with the stored angles
    `theta_nu = fl(2 arccos fl(v_nu/lambda))` on one signal qubit,
    multiplexed by the address register; values beyond `len(values)` up to
    the next power of two are 0. D is a full involution and its block with
    the signal in zero is `diag(cos(theta/2))`, which differs from
    `v / lambda` by the residual that `build_diagonal_table_encoding`
    reports. Controls are added to every gate, since the table has no
    uncontrolled shell.

    Args:
        values (np.ndarray | list | tuple): Real diagonal `v` in its physical
            units, one-dimensional.
        normalization (float): `lambda >= max |v|`, the encoding normalization.
        num_controls (int): Default `0`. Number of external controls.
        ctrl_state (int | str | None): Default `None`, all controls one.
        adjoint (bool): Default `False`. Build the adjoint, the reversed
            circuit, which equals D up to rounding.
        max_bytes (int | None): Default `None`, the physical memory.
            Limit on the arrays of the values and of the table, about 192
            bytes for each of its `2**(r + k)` entries with
            `r = ceil(log2 len(values))` address qubits and k controls, twice
            that for the adjoint, checked before the values are converted.
        max_work (int | None): Default `None`, no limit. Limit on the construction
            work, about `16 (r + k + 2)` units per table entry, twice that
            for the adjoint.

    Returns:
        gate (qiskit.circuit.Gate): The query on the controls, the signal
            and `ceil(log2 len(values))` address qubits, in that order.

    Raises:
        TypeError: If `values` is not an ndarray, list or tuple of numbers.
        ValueError: For values that are empty, complex, not finite or above
            `normalization` in magnitude, a control state that does not fit,
            or a table that exceeds the limits.
    """
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate

    angles, _, _, state = _diagonal_table(values, normalization, num_controls=num_controls, ctrl_state=ctrl_state,
                                          adjoint=adjoint, max_bytes=max_bytes, max_work=max_work)
    gate = _diagonal_table_circuit(angles, num_controls=num_controls, ctrl_state=state).to_gate()
    return inverse_realized_gate(gate, native_ucg=False) if adjoint else gate


def build_diagonal_table_encoding(values, normalization: float, *, max_bytes: int | None = None,
                                  max_work: int | None = None) -> BlockEncoding:
    """Build the angle-table block encoding of a real diagonal.

    The circuit is that of `build_diagonal_table_query` without controls,
    with the signal as its one ancilla and the address register as its
    system register. Its `alpha` is `normalization`. Its `error_bound` is
    the target residual `max |v - lambda cos(theta/2)|` over the given
    values `v` (0 on padding) and the stored angles `theta`, evaluated in
    binary64 and rounded up one step; it contains the rounding of the
    division and of the angles, and it is a numerical estimate, not an
    enclosure. The metadata records the three structural facts (Hermitian
    target and block, and a full involution) and, under
    `"error_bound_evidence"`, what the error bound covers.

    Args:
        values (np.ndarray | list | tuple): Real diagonal `v` in its physical
            units, one-dimensional.
        normalization (float): `lambda >= max |v|`.
        max_bytes (int | None): Default `None`, the physical memory.
            As for `build_diagonal_table_query` without controls.
        max_work (int | None): Default `None`, no limit. As for
            `build_diagonal_table_query`.

    Returns:
        encoding (BlockEncoding): The circuit with one ancilla.

    Raises:
        TypeError: As for `build_diagonal_table_query`.
        ValueError: As for `build_diagonal_table_query`.
    """
    angles, residual, lam, _ = _diagonal_table(values, normalization, num_controls=0, ctrl_state=None,
                                               adjoint=False, max_bytes=max_bytes, max_work=max_work)
    circuit = _diagonal_table_circuit(angles, num_controls=0, ctrl_state=0)
    address = (len(angles) - 1).bit_length()
    return BlockEncoding(circuit=circuit, alpha=lam, num_ancillas=1, system_qubits=address,
                         error_bound=float(np.nextafter(residual, np.inf)), implementation="diagonal_rotation_table",
                         metadata={**dict.fromkeys(STRUCTURE_FACT_KEYS, True), "query_census": _table_census(address),
                                   "error_bound_evidence": "binary64 evaluation of max |v - alpha cos(theta/2)| over "
                                                           "the given values and the stored angles, rounded up one "
                                                           "step; a numerical estimate, not an enclosure"})


def _comparator_circuit(half_width: int, *, num_controls: int, ctrl_state: int):
    """Return the comparator encoding on [external controls, z, f, c, l, address], its core controlled on ``ctrl_state``.

    The address holds p magnitude bits and the sign as its top bit. For
    J >= 2 the work register l has p bits, of which Uni uses the low
    ``ceil(log2 J)``. The comparison C writes ``f = [l < m]`` as the carry
    out of ``m + (2**p - 1 - l)``: X on l, the MAJ chain of Cuccaro, Draper,
    Kutin and Moulton, arXiv:quant-ph/0410184v1, Fig. 3, on
    ``(carry_i, m_i, l_i)`` with ``carry_0 = c`` and ``carry_i = l_{i-1}``,
    one CX from ``l_{p-1}`` (now the carry out) to f, the reversed chain and
    X on l. C restores l, m and c, so it is its own inverse, and
    ``U = Uni^dagger C K C Uni``. With J = 1, p = 1 and no work register,
    ``f = m_0`` and ``U = K`` with K's open control on ``m_0``.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UniformSuperpositionGate

    p = half_width.bit_length()
    k = num_controls
    registers = [QuantumRegister(k, "control")] if k else []
    signal = QuantumRegister(1, "signal")
    registers.append(signal)
    if half_width > 1:
        flag, carry, work = QuantumRegister(1, "flag"), QuantumRegister(1, "carry"), QuantumRegister(p, "work")
        registers += [flag, carry, work]
    address = QuantumRegister(p + 1, "address")
    circuit = QuantumCircuit(*registers, address, name="c_node_comparator" if k else "node_comparator")
    external = list(circuit.qubits[:k])
    magnitude, sign = list(address[:p]), address[p]

    def compare():
        circuit.x(work)
        chain = [(carry[0] if bit == 0 else work[bit - 1], magnitude[bit], work[bit]) for bit in range(p)]
        for c_in, b, a in chain:
            circuit.cx(a, b)
            circuit.cx(a, c_in)
            circuit.ccx(c_in, b, a)
        circuit.cx(work[p - 1], flag[0])
        for c_in, b, a in reversed(chain):
            circuit.ccx(c_in, b, a)
            circuit.cx(a, c_in)
            circuit.cx(a, b)
        circuit.x(work)

    if half_width > 1:
        uniform = UniformSuperpositionGate(half_width, num_qubits=(half_width - 1).bit_length())
        circuit.append(uniform, work[:uniform.num_qubits])
        compare()
        test = flag[0]
    else:
        test = magnitude[0]
    # K = Z_sign C_{test=0} X_z, controlled on the external controls.
    _append_phase_flip(circuit, [sign, *external], 1 | (ctrl_state << 1), sign)
    _append_mcx(circuit, [*external, test], ctrl_state, signal[0])
    if half_width > 1:
        compare()
        circuit.append(uniform.inverse(), work[:uniform.num_qubits])
    return circuit


def _uniform_node_query_requirements(recipe, *, num_controls, adjoint, max_bytes, max_work):
    """Admit one query of a recipe and return ``(B, W, adjoint)``, with the adjoint flag of the query actually built.

    A Pauli recipe's law (``_admit_materialization("pauli")``) is the
    query that ``build_block_encoding_query`` builds from its plan, so B and
    W count that query once.
    """
    k = integer(num_controls, "num_controls", 0)
    limits = {"max_bytes": max_bytes, "max_work": max_work}
    if recipe.plan is not None:
        # The node and core check the same H + (4 + 2A) E, so the child gets the full max_bytes. The
        # integer diagonal plan is an involution, whose adjoint query is the forward one (A = 0).
        adjoint = adjoint and recipe.plan.detail.get("full_unitary_involution") is not True
        return (*_admit_materialization("pauli", **limits, address_bits=recipe.address_qubits, controls=k,
                                        adjoint=adjoint, terms=recipe.pauli_terms), adjoint)
    if recipe.kind == "rotation_table":
        return (*_admit_materialization("rotation_table", **limits, address_bits=recipe.address_qubits, controls=k,
                                        adjoint=adjoint), adjoint)
    return (*_admit_materialization("comparator", **limits, address_bits=recipe.address_qubits, controls=k,
                                    adjoint=adjoint, half_width=recipe.half_width), adjoint)


def _uniform_node_query(recipe, *, num_controls, ctrl_state, adjoint, max_bytes, max_work):
    """Admit and build one query of a recipe; return the gate and, for the rotation table, its target residual."""
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate

    k = integer(num_controls, "num_controls", 0)
    *_, adjoint = _uniform_node_query_requirements(recipe, num_controls=k, adjoint=adjoint, max_bytes=max_bytes,
                                                   max_work=max_work)
    if recipe.plan is not None:
        return build_block_encoding_query(recipe.plan, num_controls=k, ctrl_state=ctrl_state, adjoint=adjoint,
                                          max_bytes=max_bytes, max_work=max_work), None
    residual = None
    if recipe.kind == "rotation_table":
        state = _control_state(ctrl_state, k)
        angles, residual = _rotation_table(_uniform_values(recipe), recipe.alpha)
        circuit = _diagonal_table_circuit(angles, num_controls=k, ctrl_state=state)
    else:
        circuit = _comparator_circuit(recipe.half_width, num_controls=k, ctrl_state=_control_state(ctrl_state, k))
    gate = circuit.to_gate()
    return (inverse_realized_gate(gate, native_ucg=False) if adjoint else gate), residual


def build_uniform_node_query(recipe: UniformNodeRecipe, *, num_controls: int = 0,
                             ctrl_state: int | str | None = None, adjoint: bool = False,
                             max_bytes: int | None = None, max_work: int | None = None):
    """Build one query of a uniform-node recipe, with optional controls, as a gate.

    The rotation table is the angle table of `uniform_node_values(recipe)`
    and `recipe.alpha`, built like `build_diagonal_table_query`. The
    comparator `B^dagger K B` controls only the involution K, never its
    uniform superposition or comparison B
    ([definition](../../algorithms/lchs_derivations.md#node-encodings)). The
    two Pauli recipes use
    [`build_block_encoding_query`][nwqlib.subroutines.block_encoding.build_block_encoding_query]
    of their plan, which controls only SELECT. The block of the query with
    every ancilla zero and the controls active is
    `uniform_node_values(recipe) / recipe.alpha`, up to the error bound that
    `build_uniform_node_encoding` reports, and the inactive control branch
    is the identity on all qubits.

    Args:
        recipe (UniformNodeRecipe): A recipe from `uniform_node_recipes`.
        num_controls (int): Default `0`. Number of external controls.
        ctrl_state (int | str | None): Default `None`, all controls one.
        adjoint (bool): Default `False`. Build the adjoint, the reversed
            circuit.
        max_bytes (int | None): Default `None`, the physical memory.
            Limit on the known arrays: the node grid and the table over
            `2**(r + k)` entries for the rotation table, a few lists of
            O(log J + k) entries for the comparator, and for the Pauli
            recipes the terms of the stored plan plus four times the bytes of one copy
            of the query tables: the tables, their gate copy and the copies
            Qiskit keeps of the realized, reversed and zero-support one-qubit
            matrices, as `build_block_encoding_query` counts them. A Pauli
            recipe's plan is an involution, so its adjoint query is the
            forward query and needs the same bytes. All are checked before
            anything is built.
        max_work (int | None): Default `None`, no limit. Limit on the construction
            work.

    Returns:
        gate (qiskit.circuit.Gate): The query on the controls,
            `recipe.encoding_ancillas` ancillas and `recipe.address_qubits`
            address qubits, in that order.

    Raises:
        ValueError: If `ctrl_state` does not fit `num_controls`, or the
            construction exceeds the limits.
    """
    return _uniform_node_query(recipe, num_controls=num_controls, ctrl_state=ctrl_state, adjoint=adjoint,
                               max_bytes=max_bytes, max_work=max_work)[0]


def build_uniform_node_encoding(recipe: UniformNodeRecipe, *, max_bytes: int | None = None,
                                max_work: int | None = None) -> BlockEncoding:
    """Build the uncontrolled circuit of a uniform-node recipe as a block encoding of `h j`.

    The ancillas are the recipe's signal and work qubits (or the Pauli
    address register) and the system register is the node address. The
    `alpha` is `recipe.alpha`. The target is `h j` on the physical labels
    and the recipe's completion on padding, and the error bound covers
    what the stored data change in it:

    * rotation table: the target residual
      `max |v - alpha cos(theta/2)|` of the stored angles, as for
      `build_diagonal_table_encoding`, over the binary64 values
      `v = fl(h j)`, whose own rounding is not included; a numerical
      estimate, not an enclosure.
    * comparator and Pauli recipes: `|h N - alpha|` for the extent N of
      `UniformNodeRecipe.alpha`, the effect of rounding `h N` on the exact
      block `j / N` of the ideal circuit, computed exactly and rounded up.

    Gate synthesis and the rounding of gate parameters are not included.
    The metadata records the recipe kind, the mapping, the physical radius,
    the query census of `build_uniform_node_query`, what the error bound
    covers (`"error_bound_evidence"`) and the three structural facts, all
    True.

    Args:
        recipe (UniformNodeRecipe): A recipe from `uniform_node_recipes`.
        max_bytes (int | None): Default `None`, the physical memory.
            As for `build_uniform_node_query` without controls.
        max_work (int | None): Default `None`, no limit. As for
            `build_uniform_node_query`.

    Returns:
        encoding (BlockEncoding): The circuit and its normalization.

    Raises:
        ValueError: If the construction exceeds the limits.
    """
    from qiskit import QuantumCircuit

    gate, residual = _uniform_node_query(recipe, num_controls=0, ctrl_state=None, adjoint=False,
                                         max_bytes=max_bytes, max_work=max_work)
    if residual is not None:
        error = float(np.nextafter(residual, np.inf))
        evidence = ("binary64 evaluation of max |v - alpha cos(theta/2)| over v = fl(h j) and the stored angles, "
                    "rounded up one step; the rounding of h j is not included; a numerical estimate, "
                    "not an enclosure")
    else:
        extent = recipe.half_width if recipe.plan is None else int(recipe.plan.alpha)
        error = _normalization_error(recipe.spacing, extent, recipe.alpha)
        evidence = "|h N - alpha| for the exact block j/N of the ideal circuit, in rational arithmetic, rounded up"
    circuit = QuantumCircuit(gate.num_qubits, name=recipe.kind)
    circuit.append(gate, circuit.qubits)
    return BlockEncoding(circuit=circuit, alpha=recipe.alpha, num_ancillas=recipe.encoding_ancillas,
                         system_qubits=recipe.address_qubits, error_bound=error,
                         implementation="uniform_node_" + recipe.kind,
                         metadata={**dict.fromkeys(STRUCTURE_FACT_KEYS, True), "mapping": recipe.mapping,
                                   "half_width": recipe.half_width, "spacing": recipe.spacing,
                                   "physical_radius": recipe.physical_radius,
                                   "query_census": dict(recipe.query_census), "error_bound_evidence": evidence})


__all__ = [
    "UniformNodeRecipe",
    "angle_table_cx",
    "build_diagonal_table_encoding",
    "build_diagonal_table_query",
    "build_uniform_node_encoding",
    "build_uniform_node_query",
    "uniform_node_labels",
    "uniform_node_query_cx",
    "uniform_node_recipes",
    "uniform_node_values",
]
