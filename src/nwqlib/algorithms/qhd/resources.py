"""Fault-tolerant resource laws of the QHD circuit and its separate error sources.

A Plan with native execution already publishes its CX law, and a binary Plan
also a rotation law that counts rotation gates before angle classification
(``method.QHD._select_binary_native``). This module adds three things that a
user planning for fault-tolerant hardware needs without compiling a
paper-scale circuit:

- the number of arbitrary rotations of the emitted circuit, counted from its
  stored blocks and preparation (``rotation_population``), which planning
  publishes for the one-hot encoding as a ``ResourceLaw`` with metric
  ``arbitrary_rotations``, and the same classification of the computed
  angles of a binary circuit (``_binary_population``);
- for an explicit total synthesis budget, the Clifford replacements that the
  budget pays for, the per-rotation allowance and a leading-order T estimate
  (``synthesis_projection``);
- the error sources of one circuit, each named separately with its own status
  (``circuit_resources``), and the totals of an augmented-Lagrangian run or a
  box refinement over its circuits and shots (``run_resources``).

The laws count a stated decomposition, not a compiler output. Each emitted
gate is expanded by its Qiskit 2.5.2 definition, or by NWQLib's own
phase-diagonal multiplexor, into Clifford gates and single-axis rotations
(``Rz``, ``Ry``, the phase gate ``P(phi) = exp(i phi/2) Rz(phi)``), before any
cross-gate simplification or routing. A single-axis rotation whose binary64
angle equals a stored conventional Clifford angle is not counted, one that
equals the stored ``pi/4`` or ``-pi/4`` is an exact T gate, and every other
nonzero angle is an arbitrary rotation (``resources.records.CLIFFORD_ANGLES``
and ``T_ANGLES``, the classification of the library's resource fold). The
comparison is exact, so it describes the emitted binary64 angles. Nothing is
counted as Clifford because it is near a Clifford angle. That choice is the
budgeted replacement of ``synthesis_projection``, which describes an
approximated construction and says so.

Liu et al., arXiv:2607.16996v1, Sec. VI A, count ``Rz`` gates after Qiskit's
optimization level 0 and replace every rotation within an angle threshold of
0, ``+-pi/2`` or ``pi`` by a Clifford for their encoding comparison. Their
analysis script (``experiments/res_ana/analyze_periodic_qasm_stats.py`` of
the authors' repository) sets that threshold to ``1e-12``. That fixed rule
serves one census. Its error per replacement is about ``5e-13``, but its
total grows with the number of replacements and is charged to no budget.
NWQLib therefore derives its threshold from the user's synthesis budget
instead, so every replacement it makes is paid for in the same error ledger,
and the emitted count uses no tolerance at all. The same authors divide a
total synthesis budget ``1e-4`` evenly over the remaining rotations of each
benchmark circuit (Sec. VI B). The even split is kept as the rule, because
it minimizes the logarithmic synthesis cost for a fixed rotation population
(``synthesis_projection``). Their numerical budget is a benchmark choice, so
NWQLib has no default budget and the user states one.
"""

from collections import Counter
from fractions import Fraction
from math import frexp, inf, isfinite, log2, nextafter, pi
from typing import Literal

import numpy as np
from pydantic import model_validator

from nwqlib.core.records import ContentID, Float64, Nonnegative, Record, Source, Text
from nwqlib.evidence import Evidence
from nwqlib.operators.access import Count
from nwqlib.resources.records import CLIFFORD_ANGLES, T_ANGLES, ResourceLaw
from .binary import _integer_storage
from .circuit_errors import _PI_LOW, _PI_UP
from .evolution_bounds import QHDEvolutionBound, _upward, evolution_bound
from .validation import _normal_range

# Leading coefficient a of the typical Ross-Selinger T count
# 3 log2(1/eps) + O(log log(1/eps)) of one single-qubit Rz approximated to
# operator error eps (Ross and Selinger, arXiv:1403.2975v3). No finite
# intercept is supported by that result or by NWQEC 0.1.2's documentation,
# so the estimate uses the leading term alone (intercept 0). One QHD instance
# compiled by NWQEC 0.1.2 at a total budget of 1e-4 gave 348 T against the
# estimate 392.9 (docs/algorithms/qhd.md, "Fault-tolerant resources", gives
# the instance), a model discrepancy, not an error bar. That comparison ran
# on 2026-09-26 at revision 418f9907ec4d629011486c14eb23cf4fdbdd1d3e with
# Python 3.12.14, Qiskit 2.5.2 and NWQEC 0.1.2 on macOS arm64. Registered in
# docs/ENGINEERING_CONSTANTS.md. Revisit with a qualified synthesis law.
T_PER_PRECISION_BIT = 3

ROTATION_LAW_SOURCE = Source(
    name="qhd.rotation_law",
    version="1",
    domain="one-hot Clifford plus single-axis rotation inventory of the selected construction",
    reference="Qiskit 2.5.2 definitions of XXPlusYYGate, CRYGate, PhaseGate, CPhaseGate and MCPhaseGate "
    "(MCRZ with synth_mcx_n_dirty_i15); nwqlib._multiplexors.append_control_diagonal_phases",
)
T_ESTIMATE_SOURCE = Source(
    name="qhd.t_estimate",
    version="1",
    domain="leading-order T count of the budgeted Clifford+T synthesis of the QHD circuit",
    reference="Ross and Selinger arXiv:1403.2975v3 leading term 3 log2(1/eps), intercept 0; equal split",
)


def _mcx_rotations(controls):
    """Return ``(arbitrary, t)``, the rotations of Qiskit's ``synth_mcx_n_dirty_i15`` on ``controls`` controls.

    One control is a CX. Two controls give the seven-T Toffoli. Three give
    Qiskit's C3X with fifteen phases of ``+-pi/8``, arbitrary rotations in
    this classification. From k >= 4 controls each of two passes holds one
    seven-T Toffoli, one four-T relative-phase Toffoli and ``k - 3``
    action/reset pairs of four T each, ``2 (7 + 4 + 4 (k - 3)) = 8 k - 2`` T
    (qiskit/qiskit 2.5.2, ``crates/synthesis/src/multi_controlled/mcx.rs``).
    """
    if controls == 1:
        return 0, 0
    if controls == 2:
        return 0, 7
    if controls == 3:
        return 15, 0
    return 0, 8 * controls - 2


def _normal_scaling(magnitude, s, name):
    """Raise unless ``magnitude 2**-s`` is finite and at least ``2**-1022``, compared by exponents without scaling.

    With ``magnitude = m 2**e``, ``0.5 <= m < 1`` (``frexp``),
    ``magnitude >= 2**(s - 1022)`` holds exactly when ``e - 1 >= s - 1022``.
    """
    if not (isfinite(magnitude) and frexp(magnitude)[1] - 1 >= s - 1022):
        raise ValueError(f"{name} scaled by 2**-{s} lies outside the normal binary64 range that the rotation "
                         "census and the error ledger assume")


def _projector_slots(block, wraps=None):
    """Return ``(slots, fixed_arbitrary, fixed_t)`` of one emitted number-projector block.

    The block applies ``exp(-i a (prod n - I/2**s))`` on s qubits as the
    phase ``P(-a)`` on the all-ones state (``pauli_evolution.append_number_projector_phase``).
    ``slots`` maps the magnitude of each parameter-dependent rotation angle,
    an exact power-of-two scaling of the stored angle a, to its multiplicity.

    - s = 1: ``P(-a)``, one slot at ``|a|``.
    - s = 2: ``CP(-a)`` lowers to two CX and three phases at ``+-a/2``.
    - s >= 3 with ``MCPhaseGate``: Qiskit emits MCRZ gates with
      ``k = s - 1, ..., 1`` controls and one final phase. With
      ``lambda_k = -a/2**(s - 1 - k)``, an MCRZ with ``k >= 2`` controls has
      four rotations at ``+-lambda_k/4``, magnitude ``|a|/2**(s + 1 - k)``,
      and two MCX gates on each of ``ceil(k/2)`` and ``floor(k/2)`` controls
      (``_mcx_rotations``). The CRZ (k = 1) has two rotations and the final
      phase one, all three at ``|a|/2**(s - 1)``. The parameter slots number
      ``4 s - 5``.
    - 4 <= s <= 7 with the phase-diagonal multiplexor: the phase is first
      wrapped, ``lambda = angle(exp(-i a))``, and the table
      ``(0, ..., 0, lambda)`` has one nonzero pair difference at every
      recursion level. The level with k controls holds the difference
      ``lambda/2**(s - 1 - k)``, since the pair means halve it at each level,
      and its Gray-code Walsh transform spreads it over ``2**k`` angles of
      magnitude ``|lambda|/2**(s - 1)``, so all ``2**s - 1`` rotations have
      that magnitude. The wrap is computed as the multiplexor computes it. A
      zero wrap leaves an exactly zero table and no rotation.

    ``fixed_arbitrary`` and ``fixed_t`` count the fixed rotations of the
    multi-controlled X gates, the ``+-pi/8`` phases and the T gates.

    Range admission (``validation._normal_range``). The nonzero angle a
    needs ``a 2**-s >= 2**-1022``, which also covers the smallest
    parameter-dependent rotation of each lowering, ``a/2**(s - 1)`` from
    s = 3 and a and ``a/2`` below, and the native compensation ``a/2**s``.
    A nonzero wrapped phase lambda of the phase-diagonal provider needs
    ``|lambda| 2**-s >= 2**-1022`` for its repeated means and final child
    phase. Every scaling is then an exact power-of-two scaling of a normal
    number. Otherwise this raises, at planning and when the census is
    formed again.
    """
    s, a = len(block.support), abs(block.angle)
    _normal_scaling(a, s, f"the angle {block.angle!r} of the number projector on {s} qubits")
    if block.provider == "diagonal_synthesis":
        from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import wrapped_projector_phase

        wrapped = wrapped_projector_phase(block.angle, s, wraps)
        if wrapped == 0.0:
            return {}, 0, 0
        _normal_scaling(wrapped, s, f"the wrapped phase {wrapped!r} of the projector diagonal on {s} qubits")
        return {wrapped / 2 ** (s - 1): 2**s - 1}, 0, 0
    if s == 1:
        return {a: 1}, 0, 0
    if s == 2:
        return {a / 2: 3}, 0, 0
    slots = Counter({a / 2 ** (s - 1): 3})
    fixed_arbitrary = fixed_t = 0
    for controls in range(2, s):
        slots[a / 2 ** (s + 1 - controls)] += 4
        for half in ((controls + 1) // 2, controls // 2):
            arbitrary, t = _mcx_rotations(half)
            fixed_arbitrary += 2 * arbitrary
            fixed_t += 2 * t
    return slots, fixed_arbitrary, fixed_t


class RotationPopulation(Record):
    """The arbitrary rotations and exact T gates of one emitted QHD circuit.

    Attributes:
        magnitudes: ``(|angle|, multiplicity)`` of the arbitrary rotations,
            one entry per distinct magnitude, in increasing magnitude. The
            sign does not enter the classification or the Clifford distance.
        exact_t: Exact T and T-inverse occurrences.
    """

    magnitudes: tuple[tuple[Nonnegative, Count], ...]
    exact_t: Count

    @property
    def arbitrary_rotations(self):
        """Number of arbitrary rotation occurrences, the sum of the multiplicities."""
        return sum(count for _, count in self.magnitudes)


def rotation_population(reconstruction, preparation):
    """Classify every single-axis rotation of the emitted one-hot circuit, from its stored blocks and preparation.

    The count reads the emitted construction, so rotations that
    ``rotation_threshold`` removed are absent (their bound is the pruning
    entry of the error ledger, not a proved zero), and exact-zero angles
    emit nothing. Per block kind:

    - Fused hopping ``XXPlusYYGate(theta)`` on one link, with theta twice
      the stored angle ``fl(2 dt c)`` (``pauli_evolution._hopping_parameter``):
      Qiskit's definition holds two ``Ry(-theta/2)`` between Clifford gates
      and two ``Rz(+-beta)`` at ``beta = 0``, and halving theta returns the
      stored angle exactly, so the link has two slots at the stored
      ``|angle|``.
    - Number projector: ``_projector_slots``.
    - Structured one-hot preparation, the amplitude chain
      (``initial_state.append_amplitude_chain``): an X per register and, per
      emitted link, one ``CRY(theta_m)`` whose definition holds
      ``Ry(theta_m/2)`` and ``Ry(-theta_m/2)`` and two CX, followed by a CX.
      The angles come from ``initial_state.chain_angles``. For the uniform
      state the last link has ``theta = 2 atan2(r, r) = pi/2``, so its two
      rotations are exact T gates, and a zero amplitude before a positive
      tail gives ``theta = pi``, two Cliffords. ``preparation`` is
      ``QHD.initial_state_preparation``. ``"none"`` prepares nothing, and the
      caller has no law for ``"qiskit_state_preparation"``.

    Each slot is then classified by its binary64 magnitude against the
    stored conventional angles (module docstring). The work is one pass over
    the stored blocks and the ``d K`` preparation amplitudes.

    Range admission. Planning forms this census for every one-hot Plan with
    a rotation law, and it raises when a scaling leaves the normal binary64
    range: a projector (``_projector_slots``) or a chain parameter
    ``theta/2``, which must be normal (``validation._normal_range``), though
    the Gaussian amplitudes themselves need not be. The selections of
    planning keep both in range (``potential.PotentialCompiler.select_occurrences``,
    ``initial_state.chain_selection``), so the check confirms that the
    census reads the same selection. The hopping angle is
    admitted by its producer (``kinetic.KineticCompiler``). Every slot is
    then an exact power-of-two scaling of a normal stored angle, so the
    count is exact for the stated decomposition.
    """
    from .initial_state import chain_angles

    slots = Counter()
    fixed_arbitrary = fixed_t = 0
    # Wrapped projector phases of this census, by (angle.hex(), s) (wrapped_projector_phase).
    wraps = {}
    for group in reconstruction.steps:
        for block in group:
            if block.kind == "kinetic":
                slots[abs(block.angle)] += 2
                continue
            projector, arbitrary, t = _projector_slots(block, wraps)
            slots.update(projector)
            fixed_arbitrary += arbitrary
            fixed_t += t
    if preparation == "structured":
        for alpha in reconstruction.initial_amplitudes:
            for theta in chain_angles(alpha):
                # Ry(theta/2) and Ry(-theta/2), and halving the doubled atan2 is exact once theta/2 is normal.
                half = _normal_range(theta / 2, "a CRY parameter theta/2 of the structured preparation",
                                     lambda theta=theta: Fraction(theta) / 2)
                slots[abs(half)] += 2
    if fixed_arbitrary:
        slots[pi / 8] += fixed_arbitrary
    magnitudes = Counter()
    exact_t = fixed_t
    for magnitude, count in slots.items():
        if magnitude in CLIFFORD_ANGLES:
            continue
        if magnitude in T_ANGLES:
            exact_t += count
        else:
            magnitudes[magnitude] += count
    return RotationPopulation(magnitudes=tuple(sorted(magnitudes.items())), exact_t=exact_t)


def _dense_diagonal_angles(phases):
    """Return the nonzero ``Rz`` angles that ``append_control_diagonal_phases`` emits for one phase table.

    The helper (``subroutines._multiplexors``) wraps every phase to
    ``(-pi, pi]`` with ``angle(exp(i phase))`` and then, level by level on
    the least-significant remaining qubit, emits one ``Rz`` multiplexor on the
    pairwise differences and keeps the pairwise means, whose multiplexor has
    the Gray-code angles ``gray_code_rotation_angles`` of the differences and
    omits an exactly zero angle. The same NumPy operations on the same values
    reproduce those angles bit for bit, so no circuit is built. An all-zero
    table and a table on no qubit emit no rotation. The work is
    ``O(n 2**n)`` for n qubits.
    """
    from nwqlib.subroutines._multiplexors import gray_code_rotation_angles

    values = np.asarray(phases, dtype=float)
    if values.size < 2 or not np.any(values):
        return []
    values = np.angle(np.exp(1.0j * values))
    angles = []
    while values.size > 1:
        pairs = values.reshape(-1, 2)
        angles += [angle for angle in gray_code_rotation_angles(pairs[:, 1] - pairs[:, 0]).tolist() if angle != 0.0]
        values = pairs[:, 0] / 2 + pairs[:, 1] / 2
    return angles


def merge_angles(slots, angles, occurrences=1):
    """Add the absolute values of one block's emitted nonzero angles, times ``occurrences``, to ``slots``.

    All census angles are finite binary64 numbers. Their absolute values are
    exact. ``np.unique`` groups equal values and returns the exact
    multiplicity of each group. This is precisely the multiset used by
    ``Counter(abs(angle) for angle in angles)``, including merging +0 and -0
    if zeros are present. The producers exclude zero angles. NaNs would make
    Python dictionary behavior and NumPy grouping different, and are outside
    the admitted population. The merge costs Python work proportional to
    distinct magnitudes per block. ``np.unique`` normally sorts, so its work
    is O(N log N) comparisons in the general case, and its arrays are charged
    (``_admit_inspection``). ``_binary_population`` sorts the final
    magnitudes, so insertion order has no effect on its
    ``RotationPopulation``, and the counts and angle magnitudes are unchanged.
    """
    magnitudes, counts = np.unique(np.abs(np.asarray(angles, dtype=np.float64)), return_counts=True)
    for magnitude, count in zip(magnitudes.tolist(), counts.tolist(), strict=True):
        slots[float(magnitude)] += int(count) * int(occurrences)


def _distinct_blocks(steps):
    """Return ``(((block, occurrences), ...), kinetic)``: each distinct stored binary block and its count.

    Blocks are equal when their kind, variables, exponent and synthesis are,
    which fixes the construction within one ``binary.BinaryModel``.
    ``kinetic`` counts the kinetic block occurrences.
    """
    distinct, kinetic = {}, 0
    for group in steps:
        for block in group:
            key = (block.kind, tuple(block.variables), float(block.exponent).hex(), block.synthesis)
            first, count = distinct.get(key, (block, 0))
            distinct[key] = (first, count + 1)
            kinetic += block.kind == "binary_kinetic"
    return tuple(distinct.values()), kinetic


def _unique_work(entries):
    """Return ``W_unique(N) = 2N ceil(log2 max(2, N)) + 8N``, the comparison-visit charge of grouping N angles.

    A conservative comparison-visit charge for a qualified
    introsort/merge-sort owner of ``np.unique``, an O(N log N) accounting
    envelope with an explicit sort premise, not a claim that unique is
    universally linear (``_census_sizes``).
    """
    entries = int(entries)
    return 2 * entries * (max(2, entries) - 1).bit_length() + 8 * entries


def _dense_census_bytes(entries):
    """Return ``D_dense(E) = H0 + 112E + (16 + L(n)) floor(E/2)`` for a dense phase table of ``E = 2**n`` entries.

    A sufficient envelope for the current ``_dense_diagonal_angles``,
    including its returned Python list, excluding the input phase array
    already held by the construction:

    1. ``np.asarray(phases, dtype=float)`` shares the construction's
       admitted float64 array. A dtype conversion or a Python-list input
       would add its conversion buffer.
    2. In ``np.angle(np.exp(1j*values))`` the complex multiplication and
       exponential can overlap at 32E; the exponential and real argument
       array overlap at 24E.
    3. At a recursion level with m real entries, ``q = m/2``. The enclosing
       real array, difference input, copied transform array, two copied
       butterfly halves, arithmetic intermediates, index conversion and
       gathered result fit a 32E numerical envelope; the final gather can
       have ``8m + 40q = 28m`` numerical bytes. The wrapping and recursion
       numerical phases are successive.
    4. The Gray-code helper (``subroutines._multiplexors.gray_code_rotation_angles``)
       builds ``[index ^ (index >> 1) for index in range(q)]``, at most
       ``q <= E/2`` Python integers: 16 bytes per list slot, including
       capacity slack, and ``L(n)`` per integer. Small integers shared by
       CPython are charged as if independent.
    5. At most ``E - 1`` emitted Python floats over all levels: referents at
       most 24E, old and replacement destination slots during growth 32E,
       the ``.tolist()`` and filtered-comprehension slots at most
       ``24q <= 12E``. ``72E`` covers these, rounded up from 68E.
    6. Steps 3 and 5 give 104E; 112E is a conservative whole-level envelope.

    This is a length-E law even when the block reports zero rotations: the
    helper still examines its dense phase array. For ``E < 2`` the helper
    returns immediately and H0 suffices. The array and Python-object
    allowances are qualified for 64-bit CPython 3.12.14, NumPy 2.5.2, SymPy
    1.14.0 and Pydantic 2.13.5; they are not process-RSS bounds and exclude
    allocator fragmentation, module import/code initialization and
    independently owned native-library memory.
    """
    entries = int(entries)
    if entries < 2:
        return 65536
    n = entries.bit_length() - 1
    if entries != 1 << n:
        raise ValueError("a binary phase table must have power-of-two length")
    return 65536 + 112 * entries + (16 + _integer_storage(n)) * (entries // 2)


def _inspection_group_sizes(steps):
    """Return ``(work, bytes)`` of grouping the stored blocks into distinct blocks, before grouping them.

    With B stored block occurrences,
    ``G = H0 + sum_occurrences [512 + 16*len(variables) + L(bit_length(max(1, B)))]``
    covers grouping keys, exponent hex strings, value pairs, counters,
    mapping resize storage, result slots and support-tuple references, and
    overcounts duplicates. The work is ``sum_occurrences [32 + 4*len(variables)]``,
    permitting support-tuple hashing/comparison visits as well as the scalar
    key operations.
    """
    blocks = sum(len(step) for step in steps)
    counter = _integer_storage(max(1, blocks).bit_length())
    work, size = 0, 65536
    for step in steps:
        for block in step:
            support_size = len(block.variables)
            work += 32 + 4 * support_size
            size += 512 + 16 * support_size + counter
    return work, size


def _census_sizes(rows, *, cp_pair, kinetic_occurrences, grouping_bytes, qft_gate_count, grouping_work):
    """Return ``(work, workspace)`` of the binary rotation census from scalar metadata alone.

    ``rows`` has one ``(entries, diagonal_slots, occurrences, dense)`` tuple
    per distinct block u: its full phase-table length ``E_u = 2**n_u``, a
    bound ``N_u`` on its nonzero diagonal rotation slots, its occurrence
    multiplicity ``m_u`` and whether it is a dense diagonal. With
    ``P = 2*qft_controlled_phases(bits, cutoff)`` (``cp_pair``), the CP
    gates of the forward/inverse QFT pair, a kinetic occurrence contributes
    3P single-axis rotation slots, and with ``K_occ`` kinetic occurrences

    ``U_cap = P + sum_u N_u``, ``C_cap = 3P K_occ + sum_u m_u N_u``.

    ``U_cap`` bounds distinct magnitudes, ``C_cap`` a count stored at any
    magnitude, the exact-T total and the sum of all slots. P is
    conservative: many CP angles have the same magnitude, and it covers the
    zero-count keys of the QFT-half loop when there are no kinetic
    occurrences.

    Grouping keeps the ``np.unique`` laws ``B_unique(N) = 64N + H0`` and
    ``W_unique(N) = 2N ceil(log2(max(2, N))) + 8N`` (``_unique_work``). The
    merge holds the ``magnitudes.tolist()`` and ``counts.tolist()``
    arguments simultaneously while ``zip`` is consumed, which fit
    ``B_merge_lists(N) = [56 + L(bit_length(max(1, N)))] N``: 24 + 16 for
    floats and list slots plus 16 for count-list slots, and L for the Python
    counts. The product ``int(count)*occurrences``, old value, new sum and
    scalar conversions need at most a further ``4L(b_C)`` at one merge
    operation, with ``b_C = bit_length(max(1, C_cap))``. A returned dense
    list is input to grouping and stays live there, 40N for its floats and
    slots; Walsh angles are the construction's array and belong to the
    model and cache charge. The block-local envelope is

    ``B_group(N) = H0 + 64N + B_merge_lists(N) + 4L(b_C)``,
    ``B_local,u = max(D_dense(E_u), 40N_u + B_group(N_u))`` (dense),
    ``B_local,u = B_group(N_u)`` (Walsh), with D_dense of
    ``_dense_census_bytes``.

    The whole Python population phase (the ``slots`` and ``magnitudes``
    Counters with resize overlap, float keys, original and classified
    integer counts, sorted pair tuples and list, sort scratch, and
    ``RotationPopulation`` validation and construction) fits

    ``B_population(U, C) = H0 + [1024 + 4L(bit_length(max(1, C)))] U``.

    The loop keeps its previous dense ``angles`` list while evaluating the
    next one, another ``40 N_max``. The QFT-half list and its temporary
    concatenation fit ``128 (G_qft + 1)``, with ``G_qft`` the combined
    forward/inverse gate count. With the grouping pre-bound G
    (``_inspection_group_sizes``),

    ``B_census = G + 128 (G_qft + 1) + B_population(U_cap, C_cap) + 40 N_max + max_u B_local,u``.

    The work is the grouping visits, the dense formation work
    ``W_dense(E) = (8n + 32) E`` of each distinct dense block (butterflies,
    copies, Gray-index formation, gather, filtering and means, plus the
    initial scan and wrap, in the qualified array-operation/visit unit, not
    a CPU-cycle bound), ``sum_u W_unique(N_u)``, at most ``sum_u N_u`` merge
    visits, the QFT-half loop, classification of ``U_cap`` magnitudes and
    ``W_unique(U_cap)`` for the final sort. It supplements the model's own
    construction work.
    """
    # rows is an admitted small metadata sequence, not a sequence of arrays.
    unique_cap = int(cp_pair) + sum(n for _, n, _, _ in rows)
    count_cap = 3 * int(cp_pair) * int(kinetic_occurrences)
    count_cap += sum(n * multiplicity for _, n, multiplicity, _ in rows)
    count_bytes = _integer_storage(max(1, count_cap).bit_length())
    population = 65536 + (1024 + 4 * count_bytes) * unique_cap
    largest = previous_list = dense_work = merge_visits = 0
    grouping_work = int(grouping_work) + int(cp_pair)
    for entries, n, multiplicity, dense in rows:
        if entries < 1 or n < 0 or multiplicity < 1:
            raise ValueError("invalid census metadata")
        lists = (56 + _integer_storage(max(1, n).bit_length())) * n
        group = 65536 + 64 * n + lists + 4 * count_bytes
        local = group
        if dense:
            local = max(_dense_census_bytes(entries), 40 * n + group)
            previous_list = max(previous_list, 40 * n)
            dense_work += (8 * (entries.bit_length() - 1) + 32) * entries
        largest = max(largest, local)
        grouping_work += _unique_work(n)
        merge_visits += n
    workspace = (int(grouping_bytes) + 128 * (int(qft_gate_count) + 1)
                 + population + previous_list + largest)
    work = (grouping_work + dense_work + merge_visits
            + unique_cap + _unique_work(unique_cap))
    return work, workspace


def _admit_inspection(plan, held_bytes=0):
    """Admit the binary rotation census of a Plan (``_admit_census``); None for a one-hot Plan."""
    if plan.method.encoding != "binary":
        return None
    r = plan.reconstruction
    from .method import _binary_source_reservation

    method = plan.method
    specs = tuple((int(t.values.array.size), len(t.support)) for t in r.support_values)
    source, _ = _binary_source_reservation(
        len(plan.problem.variables), method.num_grid_points, specs,
        method.num_steps, method.trotter_order)
    payload = 8 * sum(e for e, _ in specs)
    return _admit_census(method, r.steps, r.support_values,
                         held_bytes=int(held_bytes) + source - payload)


def _admit_census(method, steps, support_values, held_bytes=0):
    """Admit the binary rotation census of ``circuit_resources`` or ``run_resources``; return its grouping and bytes.

    Resource inspection admits distinct-block grouping, the full dense-table
    angle-formation workspace, NumPy unique/grouping arrays, Python
    magnitude/count objects, final sorting and RotationPopulation validation
    against the Plan's QHD limits. Dense formation is charged by the
    phase-table length even when no rotation survives. Count integers are
    sized from the total occurrence bound. The model, its admitted
    construction cache and the caller's retained data share the same byte
    limit with the census. The array and Python-object allowances describe
    the qualified runtime's named allocations; module initialization and
    process RSS have separate scope.

    The grouping bytes (``_inspection_group_sizes``) are checked before the
    blocks are grouped, and that one grouping is returned for the census
    (``_binary_population``). The census work of ``_census_sizes`` is the
    named inspection ``QHD.max_work`` charge, checked before any dense
    formation or sort. The returned bytes are ``held_bytes`` (the caller's
    retained data), the Plan's support tables, 8 bytes per entry, and
    ``B_census``; the caller passes them to ``binary.BinaryModel`` as its
    held bytes, whose baseline admission (``BinaryModel.__init__``,
    documented under ``construction``) adds the model, one current
    construction per table and the largest synthesis or use workspace
    against ``QHD.max_bytes`` before any array is built.
    For each distinct block, ``entries`` is its table length (``2**b`` for a
    kinetic block, ``2**(b |S|)`` for a potential block on support S), its
    diagonal slots are its recorded rotations less the ``3P`` of its QFT
    pair for a kinetic block and its recorded rotations for a potential
    block, and ``dense`` its stored synthesis. Quantum binary planning uses
    this same metadata census admission before returning its Plan. The
    returned held charge includes the caller's other live records, source
    table payloads once and the census workspace. The model baseline is
    admitted against the remainder before model construction.
    ``method``, ``steps`` and ``support_values`` are the binary Plan's QHD
    Method, stored step blocks and support tables (``_admit_inspection``
    passes a Plan's), so planning can admit the same charge before its Plan
    exists. It returns ``(distinct, kinetic, held)``.

    Raises:
        ValueError: The census exceeds the Plan's QHD work or byte limit.
    """
    from .binary import _require_bytes, qft_controlled_phases, register_bits

    held = int(held_bytes) + 8 * sum(int(t.values.array.size) for t in support_values)
    grouping_work, grouping_bytes = _inspection_group_sizes(steps)
    _require_bytes(method.max_bytes, held=held, local=grouping_bytes,
                   stage="resource inspection of the binary block grouping", later=True)
    distinct, kinetic = _distinct_blocks(steps)
    bits = register_bits(method.num_grid_points)
    synthesis = method.binary_synthesis
    phases = qft_controlled_phases(bits, synthesis.aqft_cutoff)
    rows = []
    for block, _occurrences in distinct:
        slots = block.rotations - 6 * phases if block.kind == "binary_kinetic" else block.rotations
        rows.append((1 << (bits * len(block.variables)), slots, _occurrences, block.synthesis == "dense_diagonal"))
    # The QFT pair's gates: H on each qubit, its controlled phases and, with swaps, floor(b/2) swaps, twice.
    gates = 2 * (bits + phases + (bits // 2 if synthesis.qft_bit_reversal == "swap" else 0))
    work, workspace = _census_sizes(rows, cp_pair=2 * phases, kinetic_occurrences=kinetic,
                                    grouping_bytes=grouping_bytes, qft_gate_count=gates, grouping_work=grouping_work)
    if work > method.max_work:
        raise ValueError(
            f"QHD resource inspection of the binary rotation census requires {work} work units, "
            f"with QHD(max_work={method.max_work}). Use QHD(max_work>={work})")
    return distinct, kinetic, held + workspace


def _binary_population(reconstruction, model, distinct, kinetic):
    """Classify every single-axis rotation of the emitted binary circuit from its stored blocks.

    ``model`` is the Plan's ``binary.BinaryModel``, whose ``construction``
    rebuilds each block's diagonal from its stored exponent and synthesis
    exactly as ``native.append_binary_steps`` does. Per block:

    - Kinetic block: the selected QFT and its exact inverse
      (``model.forward``, ``model.inverse``). Qiskit's ``CPhaseGate(lambda)``
      definition holds ``P(lambda/2)``, ``P(-lambda/2)`` and ``P(lambda/2)``
      around two CX, so each controlled phase at ``lambda = pi/2**r`` gives
      three rotations of magnitude ``pi/2**(r+1)``, exact T gates for r = 1.
      H and swaps are Clifford. The cutoff of an approximate QFT has already
      removed the dropped phases from the gate list.
    - Walsh diagonal: one ``Rz(theta_m)`` per kept mask, as a single ``Rz``,
      inside ``RZZGate`` or between parity CX (``apply_pauli_rotation``).
    - Dense diagonal: the nonzero multiplexor angles
      (``_dense_diagonal_angles``). The recorded slot count
      ``2**n - 1`` includes omitted zero angles.

    The structured binary preparation is an H layer and adds no rotation.
    Each magnitude is classified against the stored conventional angles
    (module docstring). Every angle reaches its gate unchanged or, in a
    controlled phase, halved exactly in the normal range. The work is one
    synthesis of every stored block, the
    transforms that planning performed, and for a dense diagonal on n
    qubits the ``O(n 2**n)`` Gray-code angles of each distinct block, which
    planning does not form, with no circuit.

    Each distinct block, keyed by kind, variables, exponent and synthesis,
    is synthesized once and its angles are merged with its occurrence
    multiplicity (``merge_angles``), so equal second-order halves are counted
    twice from one construction. ``_admit_inspection`` admits this census
    before any construction and passes its grouping, ``distinct`` and the
    kinetic occurrence count ``kinetic`` (``_distinct_blocks``).
    """
    slots = Counter()
    halves = [abs(gate[1]) / 2 for gate in model.forward + model.inverse if gate[0] == "cp"]
    for magnitude in halves:
        slots[magnitude] += 3 * kinetic
    for block, occurrences in distinct:
        built = model.construction(block.kind, block.variables, block.exponent, block.synthesis)
        angles = built.angles if built.phases is None else _dense_diagonal_angles(built.phases)
        merge_angles(slots, angles, occurrences)
    magnitudes, exact_t = Counter(), 0
    for magnitude, count in slots.items():
        if magnitude in CLIFFORD_ANGLES:
            continue
        if magnitude in T_ANGLES:
            exact_t += count
        else:
            magnitudes[magnitude] += count
    return RotationPopulation(magnitudes=tuple(sorted(magnitudes.items())), exact_t=exact_t)


def _onehot_wrap_census_bytes(method, reconstruction):
    """Return (source, local) for a one-hot census that can call the wrap.

    Source tables, initial vectors, step/block records and support positions
    follow the existing table and record inventories. The local population
    uses the qualified census rate 1024+4*L(bit_length(C)) per possible
    magnitude, with C bounding total rotation multiplicity. It prices
    Counter keys/counts, sorting and RotationPopulation construction.
    The wrap cache and one miss coexist with these populations.
    Return (0, 0) when no stored block uses the diagonal provider.
    Caller-owned additional live buffers are separate held_bytes.

    U counts possible distinct magnitudes and C total rotation or T
    occurrences, before the census allocates. Each kinetic block adds at most
    one magnitude and two rotations. A diagonal projector on s qubits adds at
    most one magnitude and ``2**s - 1`` rotations. An MCPhase projector adds
    at most ``max(1, s-1)`` parameter-dependent magnitudes, its fixed
    arbitrary rotations share the magnitude pi/8, for which one key is
    reserved globally, and ``16 s**2`` bounds its rotation and T population:
    ``_projector_slots`` has at most ``4s - 5`` parameter rotations at s >= 3,
    and for each ``controls`` value the two copies on each half contribute
    at most ``16 controls`` fixed rotations or T gates, since
    ``_mcx_rotations(h)`` totals at most ``8h``; summing controls from 2 to
    s-1 gives less than ``8 s**2`` with the parameter rotations, and
    ``16 s**2`` also covers s = 1 and 2. Structured preparation adds at most
    ``d(K-1)`` magnitudes and ``2d(K-1)`` rotations; the ``"none"``
    preparation adds none. ``L(b) = 32 + 4 ceil(max(1,b)/30)`` is
    ``binary._integer_storage``. The source inventory charges the table
    payload once, the table metadata ``65536 + 6144T + (24 + L(b_d))M +
    256d``, the stored initial vectors ``8dK + 324d + 64``, ``_STEP_BYTES``
    per schedule row, ``_BLOCK_BYTES`` per block and a support-position rate
    from the register width, because one-hot blocks store qubits. These are
    engineering allowances on CPython 3.12.14 and NumPy 2.5.2 for the named
    populations, not process RSS or arbitrary symbolic or SDK internals.
    """
    from .binary import _integer_storage
    from .method import _BLOCK_BYTES, _STEP_BYTES
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import wrapped_phase_workspace_bytes

    r = reconstruction
    d, k = len(r.initial_amplitudes), method.num_grid_points
    preparation = method.initial_state_preparation == "structured"
    unique = 1 + (d * (k - 1) if preparation else 0)
    count = 2 * d * (k - 1) if preparation else 0
    blocks = positions = wrap_entries = 0
    for group in r.steps:
        for block in group:
            blocks += 1
            s = len(block.support)
            positions += s
            if block.kind == "kinetic":
                unique += 1
                count += 2
            else:
                if block.provider == "diagonal_synthesis":
                    unique += 1
                    entries = 1 << s
                    wrap_entries = max(wrap_entries, entries)
                    count += entries - 1
                else:
                    unique += max(1, s - 1)
                    count += 16 * s * s
    if not wrap_entries:
        return 0, 0
    t = len(r.support_values)
    indices = sum(len(table.support) for table in r.support_values)
    table_metadata = (65536 + 6144 * t
                      + (24 + _integer_storage(max(0, d - 1).bit_length())) * indices
                      + 256 * d)
    width_index = max(0, r.width - 1)
    support_rate = (32 + _integer_storage(width_index.bit_length())
                    + 12 * len(str(width_index)))
    source = (8 * sum(int(table.values.array.size) for table in r.support_values)
              + table_metadata + 8 * d * k + 324 * d + 64
              + _STEP_BYTES * len(r.step_weights) + _BLOCK_BYTES * blocks
              + support_rate * positions)
    population = 65536 + (1024 + 4 * _integer_storage(max(1, count).bit_length())) * unique
    return source, population + wrapped_phase_workspace_bytes(wrap_entries)


def _admitted_onehot_population(method, reconstruction, held_bytes=0):
    """Admit a reached wrap together with the census residents, then count.

    The byte gate ``held_bytes + source + local`` of
    ``_onehot_wrap_census_bytes`` runs against the Method's ``max_bytes``
    before ``rotation_population`` creates its Counters or wrap cache. Its
    zero work argument makes it an additional byte gate, not a claim that
    the census performs zero work.
    """
    source, local = _onehot_wrap_census_bytes(method, reconstruction)
    if local:
        method._admit(0, int(held_bytes) + source + local,
                      stage="one-hot rotation census and wrapped phase")
    return rotation_population(reconstruction, method.initial_state_preparation)


def _plan_population(plan, held_bytes=0):
    """Return the ``RotationPopulation`` of a native QHD Plan's circuit, for either encoding.

    A binary census is admitted first (``_admit_inspection``) with the
    caller's retained data ``held_bytes``, and its model is built under that
    reservation. A one-hot census whose stored blocks use the diagonal
    provider admits its wrapped-phase cache beside its source records and
    the same ``held_bytes`` (``_admitted_onehot_population``).
    """
    from .method import _grid

    r, method = plan.reconstruction, plan.method
    if method.encoding != "binary":
        return _admitted_onehot_population(method, r, held_bytes)
    distinct, kinetic, held = _admit_inspection(plan, held_bytes)
    return _binary_population(r, _inspection_model(_grid(plan), method, r.support_values, held), distinct, kinetic)


def _inspection_model(grid, method, support_values, held_bytes):
    """Return the Plan's ``binary.BinaryModel`` built under the census reservation, or raise a named inspection refusal."""
    from .binary import BinaryModel

    try:
        return BinaryModel(grid, method, support_values, held_bytes=held_bytes)
    except ValueError as error:
        raise ValueError(f"QHD resource inspection of the binary rotation census: {error}. Replan with larger "
                         "QHD limits to inspect this circuit's rotations") from error


def rotation_law(population):
    """Return the planning ``ResourceLaw`` with metric ``arbitrary_rotations`` of one emitted circuit.

    It is exact for the stated decomposition and classification, whose
    scalings the census admits in the normal range (``rotation_population``).
    Its basis is ``selected_logical`` with no precision or synthesis, so the
    default resource fold reports it. The exact T gates are not a T count of
    the circuit, because its arbitrary rotations still need synthesis, so
    they stay in ``QHDCircuitResources.exact_t`` rather than in a law.
    """
    return ResourceLaw(
        metric="arbitrary_rotations",
        basis="selected_logical",
        value=population.arbitrary_rotations,
        interpretation="exact",
        evidence=Evidence(kind="proved_relation", source=ROTATION_LAW_SOURCE),
        assumptions=(
            "emitted blocks and structured preparation expanded by the stated gate definitions, before "
            "cross-gate simplification or routing; an angle equal to a stored conventional Clifford or T "
            "angle is not an arbitrary rotation",
        ),
    )


def _downward(value):
    """Return the greatest binary64 number not above the exact nonnegative rational ``value``."""
    result = float(value)
    if Fraction(result) > value:
        result = nextafter(result, -inf)
    return result


def _distance_bound(angle):
    """Return the exact rational upper bound on ``d_C`` of ``Rz(angle)`` that ``clifford_distance`` describes.

    A simpler valid bound is ``|delta|/2``, because ``|sin t| <= |t|`` gives
    ``2 sin(|delta|/4) <= |delta|/2`` for every delta. It is at most 0.65
    percent above ``d_C`` for ``|delta| <= pi/4``, the largest gap at
    ``|delta| = pi/4``, where the Taylor bound is within ``5e-9`` of ``d_C``.
    The simpler bound took about 5.3 us per distinct angle magnitude against
    12.6 us, for 20,000 magnitudes drawn uniformly below ``1e-6`` and below
    ``1e-2``. On the one-hot Ackley problem of Liu et al.'s Table II
    (arXiv:2607.16996v1) at 64 points per variable, with 100 first-order
    steps to time 10 under ``CubicSchedule(s=1)``, the form of their
    Eq. (92), no rotation threshold and no state preparation, which has
    155,063 distinct magnitudes, ``synthesis_projection`` took 0.85 s
    against 1.83 s, next to 11.9 s of planning, each projection time the
    best of three runs at the budget ``E = 1e-4``. These timings ran on
    2026-09-26 at revision ef056ac9ac9be97ec234e5bcd7e8f5d1b8d68c9f with
    Python 3.12.14 and Qiskit 2.5.2 on an Apple M3 Max. On this instance the
    Taylor bound added about 0.98 s to projection beside 11.9 s of planning,
    and both bounds made zero replacements. The Taylor bound is used for
    its tighter error charge, which can admit more replacements on other
    rotation populations.
    """
    magnitude = abs(angle)
    k = round(magnitude / (pi / 2))
    exact = Fraction(magnitude)
    # |angle - k pi/2| is convex in pi, so its largest value on [pi_low, pi_up] lies at an endpoint.
    delta = max(abs(exact - k * _PI_LOW / 2), abs(exact - k * _PI_UP / 2))
    if delta > 2 * _PI_LOW:
        return Fraction(2)
    # 2 sin(y) <= 2 y (1 - y**2/6 + y**4/120) for y = delta/4 >= 0.
    y2 = (delta / 4) ** 2
    return min(Fraction(2), delta / 2 * (1 - y2 / 6 + y2 * y2 / 120))


def clifford_distance(angle):
    """Return an upper bound on ``d_C = 2 sin(|delta|/4)``, the phase-minimized distance of ``Rz(angle)`` from the nearest Clifford rotation.

    With the nearest ``k pi/2`` and ``delta = angle - k pi/2``,
    ``|delta| <= pi/4``, multiplying by the inverse Clifford leaves
    ``Rz(delta)`` with eigenvalues ``exp(+-i delta/2)``, and the best common
    phase bisects them, so
    ``min_phi ||Rz(angle) - exp(i phi) Rz(k pi/2)|| = |exp(i delta/2) - 1| = 2 sin(|delta|/4)``,
    from ``|exp(i t) - 1| = 2 |sin(t/2)|``. A replacement is admissible for
    a per-rotation allowance eps when ``d_C <= eps``, that is
    ``|delta| <= 4 arcsin(eps/2)``, about ``2 eps`` for small eps, and every
    angle is replaceable once ``eps >= 2 sin(pi/16)``.

    The value is outward and uses no library sine. k is the rounded binary64
    quotient ``|angle|/(pi/2)``. Any integer k names a Clifford rotation, and
    the common phase 0 gives ``2 |sin(delta/4)|``, so a k that misses the
    nearest multiple, as for a huge angle, only loosens the bound. ``|delta|``
    is enclosed from above in exact rational arithmetic with the binary64
    neighbors of pi, where it takes its largest value because it is convex
    in pi. For ``y = |delta|/4`` up to pi/2, where sine increases,
    ``2 sin(y) <= 2 y (1 - y**2/6 + y**4/120)``, the alternating Taylor bound
    of sine for ``y >= 0``, and a larger ``|delta|`` gets the cap 2. The
    result is converted upward. It applies to a standalone circuit, whose
    global phase is free. A controlled reuse would need phase-consistent
    representatives.
    """
    return _upward(_distance_bound(angle))


class QHDSynthesisProjection(Record):
    """Budgeted Clifford replacement and a leading-order T estimate for one QHD circuit.

    The projection describes an approximated construction. The emitted
    circuit is unchanged, and replaced rotations are Cliffords only in the
    construction this record prices. ``synthesis_projection`` states the
    rule.

    Attributes:
        epsilon: E, the total operator-norm allowance of one circuit
            execution for Clifford replacement plus rotation synthesis.
        candidates: N, the arbitrary rotations of the emitted circuit.
        threshold: ``q = E/N``, the initial per-rotation allowance, rounded
            downward, None when N is zero.
        replaced: Candidates whose distance bound (``clifford_distance``)
            is at most q, replaced by Cliffords.
        replacement_error: ``E_C``, the exact sum of their distance bounds,
            rounded upward.
        rotations: ``M = N - replaced``, the projected synthesis population.
        rotation_epsilon: Per-rotation allowance obtained by rounding
            E minus the published replacement_error downward, dividing
            that remaining allowance by M exactly, and rounding downward
            again. None when M is zero. The published replacement charge
            plus M times this allowance is at most E.
        exact_t: T and T-inverse occurrences of the emitted circuit, which
            need no synthesis and spend no budget.
        t: The T estimate as a ``ResourceLaw`` (metric ``t``, basis
            ``clifford_t``, interpretation ``estimate``, precision E), or None.
        t_unavailable: Why there is no estimate, or None.
    """

    epsilon: Nonnegative
    candidates: Count
    threshold: Nonnegative | None
    replaced: Count
    replacement_error: Nonnegative
    rotations: Count
    rotation_epsilon: Nonnegative | None
    exact_t: Count
    t: ResourceLaw | None
    t_unavailable: Text | None

    @model_validator(mode="after")
    def _population(self):
        if self.epsilon <= 0 or self.rotations != self.candidates - self.replaced:
            raise ValueError("a synthesis projection needs a positive budget and M = N - replaced")
        if (self.threshold is None) != (self.candidates == 0) or (
            self.rotation_epsilon is None) != (self.rotations == 0):
            raise ValueError("the threshold needs candidates and the rotation allowance remaining rotations")
        if (self.t is None) == (self.t_unavailable is None):
            raise ValueError("a synthesis projection has a T estimate or the reason it has none")
        return self


def synthesis_projection(population, epsilon):
    """Replace the rotations that the budget pays for, split the rest evenly and estimate the T count.

    Rule, for N arbitrary rotations sharing the total allowance E, with B
    the replacement set and ``M = N - |B|`` remaining rotations:

    1. For N > 0, test each candidate replacement against ``q = E/N`` in
       exact rational arithmetic.
    2. Let C be the exact sum of admitted distance bounds and publish
       ``E_C = up64(C)``. Form ``R = down64(E - E_C)`` from an exact
       subtraction.
    3. For M > 0 remaining rotations, publish ``eps = down64(R/M)`` after
       exact division. If M = 0 no per-rotation tolerance is needed. If N = 0
       the threshold is also None. A positive M with eps = 0 is refused.

    The threshold and selected distance bounds are exact rationals. Their
    sum C is at most |B| E/N. The published replacement_error is C rounded
    upward. Since E is representable, that published charge is at most E.
    The remaining allowance is E minus the published charge, formed
    exactly and rounded downward. Dividing it by M and rounding downward
    gives the recorded rotation_epsilon. Thus the stored charge plus M
    stored per-rotation allowances is at most E. The exact unrounded
    allocation is at least E/N, while the published share may be slightly
    smaller because of rounding. A distance upper bound can leave a
    threshold rotation in the synthesis population, which costs T gates
    without spending uncharged error. Exact T gates spend no budget. Rotations
    that ``rotation_threshold`` pruned left the population, and their error
    stays in the pruning entry, so nothing is charged twice. Dividing the
    original E by M while also charging a nonzero ``E_C`` would overspend.

    Even split. For a fixed population M > 0 and remaining real allowance R
    with 0 < R < M, minimize ``sum_j [a log2(1/eps_j) + beta]`` subject to
    ``0 < eps_j < 1`` and ``sum_j eps_j <= R``, with a > 0. The objective
    decreases with each eps_j, so a real optimum uses the full allowance.
    Strict convexity and Jensen's inequality give the unique real allocation
    ``eps_j = R/M`` and cost at least ``M[a log2(M/R) + beta]``. Published
    downward rounding can leave unused margin and does not assert an optimum
    over binary64 allocations. Discrete replacement choices, integer synthesis costs
    and cross-gate cancellation can change the global optimum, and this rule
    is a sufficient zero-T replacement rule, not that larger optimization.

    Estimate. ``T = T_exact + M a log2(1/eps)`` with ``a = T_PER_PRECISION_BIT``
    and intercept 0, the leading term of the Ross-Selinger count
    (arXiv:1403.2975v3). It treats the circuit's angles as typical
    Ross-Selinger instances. A large rotation count gives no
    law-of-large-numbers justification, because QHD's angles repeat and are
    structured, and the estimate carries no error bar. The model applies to
    allowances below 1. With no rotation left the count is the exact T count
    of the decomposition. The estimate describes synthesis of the stated
    decomposition before final compiler optimization. A compiler's own
    cleanup, grouping and angle transport are not modeled, and the requested
    budget is never an achieved error (docs/algorithms/qhd.md,
    "Fault-tolerant resources").

    Raises:
        ValueError: ``epsilon`` is not a positive finite number, or the
            allowance ``(E - E_C)/M`` of the remaining rotations lies below
            the smallest positive binary64 number, which the record cannot
            state without rounding it up to an allowance above the budget.
    """
    epsilon = float(epsilon)
    if not 0 < epsilon < float("inf"):
        raise ValueError("synthesis_epsilon must be a positive finite total operator-norm budget")
    candidates = population.arbitrary_rotations
    threshold = Fraction(epsilon) / candidates if candidates else None
    replaced, charge = 0, Fraction(0)
    for magnitude, count in population.magnitudes:
        distance = _distance_bound(magnitude)
        if distance <= threshold:
            replaced += count
            charge += count * distance
    rotations = candidates - replaced
    # Publish Cbar first, then allocate down64(E - Cbar) so the stored fields compose within E.
    replacement_error = _upward(charge)
    remaining = _downward(Fraction(epsilon) - Fraction(replacement_error))
    rotation_epsilon = _downward(Fraction(remaining) / rotations) if rotations else None
    if rotation_epsilon == 0:
        raise ValueError(f"synthesis_epsilon={epsilon!r} leaves each of the {rotations} remaining rotations an "
                         "allowance below the smallest positive binary64 number 2**-1074. Give a larger budget")
    assumptions = (
        f"total budget {epsilon!r} for Clifford replacement plus synthesis of one circuit execution, "
        "replacement at the initial per-rotation allowance and an even split of the rest",
        f"leading-order Ross-Selinger count {T_PER_PRECISION_BIT} log2(1/eps) per remaining rotation, "
        "intercept 0, the circuit's angles treated as typical instances",
        "stated decomposition before cross-gate optimization; compiler cleanup, grouping and angle "
        "transport not modeled",
    )
    t, reason = None, None
    if rotations == 0:
        t = ResourceLaw(metric="t", basis="clifford_t", value=population.exact_t, interpretation="exact",
                        evidence=Evidence(kind="proved_relation", source=T_ESTIMATE_SOURCE),
                        precision=Float64(value=epsilon), synthesis=T_ESTIMATE_SOURCE, assumptions=assumptions)
    elif rotation_epsilon < 1:
        # T = T_exact + M a log2(1/eps), the leading-order estimate at the even split. The negated
        # logarithm stays finite where the reciprocal of a subnormal allowance overflows.
        value = population.exact_t + T_PER_PRECISION_BIT * rotations * -log2(rotation_epsilon)
        t = ResourceLaw(metric="t", basis="clifford_t", value=Float64(value=value), interpretation="estimate",
                        evidence=Evidence(kind="numerical_estimate", source=T_ESTIMATE_SOURCE),
                        precision=Float64(value=epsilon), synthesis=T_ESTIMATE_SOURCE, assumptions=assumptions)
    else:
        reason = (f"the per-rotation allowance {rotation_epsilon!r} is not below 1, where the logarithmic "
                  "synthesis model does not apply")
    return QHDSynthesisProjection(
        epsilon=epsilon, candidates=candidates, threshold=None if threshold is None else _downward(threshold),
        replaced=replaced, replacement_error=replacement_error, rotations=rotations, rotation_epsilon=rotation_epsilon,
        exact_t=population.exact_t, t=t, t_unavailable=reason,
    )


ErrorSourceName = Literal[
    "kinetic_model", "time_ordering", "midpoint_quadrature", "coefficient_rounding", "product_formula", "aqft", "angle_formation",
    "rotation_pruning", "gate_parameters", "diagonal_wrap", "identity_phase", "state_preparation",
    "clifford_replacement", "synthesis", "compiler",
]


class QHDErrorSource(Record):
    """One source of error of a QHD circuit against its finite model, with its own status.

    The ``kinetic_model`` source instead compares that finite model with the
    finite-difference one (``circuit_resources``). A ``bound`` is an upper bound on the operator-norm change that the
    source causes, or on the state 2-norm change for the preparation. An
    ``estimate`` is a value without that guarantee, which ``description``
    explains. A ``requested`` value is a budget the user asked for, never an
    achieved error. An ``unavailable`` source has no value and says why, and
    ``not_applicable`` names a source that this construction does not have.
    The sources add by telescoping only when each compares adjacent stages
    of one chain, and the record forms no combined total
    (``circuit_resources``).

    Attributes:
        name: The source.
        status: One of the statuses above.
        value: The value, None exactly for ``unavailable`` and ``not_applicable``.
        description: What the value compares and on what it rests.
    """

    name: ErrorSourceName
    status: Literal["bound", "estimate", "requested", "unavailable", "not_applicable"]
    value: Nonnegative | None = None
    description: Text

    @model_validator(mode="after")
    def _value(self):
        if (self.value is None) != (self.status in ("unavailable", "not_applicable")):
            raise ValueError("an error source has a value exactly when its status is not unavailable or "
                             "not_applicable")
        return self


class QHDCircuitResources(Record):
    """Fault-tolerant resources and error sources of one native QHD circuit.

    All counts are per circuit execution. Shots and outer rounds multiply
    them (``run_resources``).

    Attributes:
        plan_id: Content identity of the Plan.
        arbitrary_rotations: Arbitrary rotations of the emitted circuit,
            classified exactly by angle (``rotation_population`` for one-hot,
            the computed block angles for binary). For one-hot this is the
            value of ``rotation_law``.
        exact_t: Exact T and T-inverse occurrences of the emitted circuit.
        rotation_law: The Plan's ``arbitrary_rotations`` law. The binary
            encoding's law (``method.QHD._select_binary_native``) counts the
            rotation gates before angle classification, including exact T
            gates and the omitted zero angles of dense diagonals, so it is an
            upper bound on ``arbitrary_rotations``.
        synthesis: The budgeted replacement and T estimate, None without a
            synthesis budget.
        evolution: Splitting and schedule bounds (``evolution_bound``).
        error_sources: The replacement of the finite-difference kinetic,
            then every error source of the circuit against the finite model,
            in stage order.
    """

    plan_id: ContentID
    arbitrary_rotations: Count
    exact_t: Count
    rotation_law: ResourceLaw
    synthesis: QHDSynthesisProjection | None
    evolution: QHDEvolutionBound
    error_sources: tuple[QHDErrorSource, ...]


def _native_plan(plan):
    """Return the Plan's rotation law after checking that it selects a native QHD circuit with one."""
    if plan.execution != "quantum":
        raise ValueError("fault-tolerant QHD resources describe a native circuit; plan with execution='quantum'")
    laws = [law for selection in plan.construction.selections for law in selection.resource_laws
            if law.metric == "arbitrary_rotations"]
    if len(laws) != 1:
        raise ValueError("the selected state preparation has no rotation law; initial_state_preparation="
                         "'qiskit_state_preparation' synthesizes an arbitrary amplitude vector per variable")
    return laws[0]


def _one_hot_sources(entry, r, method, grid):
    """Return the one-hot ledger entries from ``angle_formation`` through ``state_preparation`` (``circuit_resources``)."""
    from .circuit_errors import block_errors, phase_allowance, preparation_error

    counts = block_errors(r, method, grid)
    phase = phase_allowance(r, method, grid, counts)
    omitted = r.range_omissions
    return [
        entry("angle_formation", "bound", counts["formation"],
              "stored angles of the kept hopping and projector blocks against their intended exponents"),
        entry("aqft", "not_applicable", None, "the one-hot circuit applies no Fourier transform"),
        entry("rotation_pruning", "bound", counts["omitted"],
              f"blocks omitted by rotation_threshold ({r.dropped_count} nonzero rotations) or below the normal "
              f"binary64 range ({omitted.projectors} projector occurrences), at their exact intended exponents"),
        entry("gate_parameters", "bound", counts["hopping"],
              "fused hopping gate parameter against twice the stored angle, and exact power-of-two scalings of "
              "the projector and chain angles"),
        entry("diagonal_wrap", "unavailable", None,
              f"phase wrap of {counts['dense']} phase-diagonal projector blocks, with no qualified error bound "
              "for NumPy's complex exponential and argument")
        if counts["dense"] else entry("diagonal_wrap", "not_applicable", None,
                                      "no projector uses the phase-diagonal provider"),
        entry("identity_phase", "bound", phase, "native global-phase bookkeeping against the exact identity "
              "phase, the ledger and native assignment allowances plus 21u per phase-diagonal child"
              + (", with the identity phases of contributions omitted below the normal binary64 range "
                 f"({omitted.identity_events} events)" if omitted.identity_events else "")),
        entry("state_preparation", "estimate", preparation_error(r, method, grid),
              "structured chain against the exact product target, first order in the chain's rounding"
              + (", plus the exact charge of chain links omitted below the normal binary64 range "
                 f"({sum(a - b for a, b, _ in omitted.chains)} links)" if any(c for _, _, c in omitted.chains)
                 else ""))
        if method.initial_state_preparation == "structured" else
        entry("state_preparation", "not_applicable", None, "initial_state_preparation='none' prepares no state"),
    ]


def _binary_sources(entry, r, method, grid, model):
    """Return the binary ledger entries from ``angle_formation`` through ``state_preparation`` (``circuit_resources``)."""
    from .binary import register_bits
    from .circuit_errors import binary_angle_formation, binary_phase_allowance

    phase = binary_phase_allowance(r, method, grid)
    formation = binary_angle_formation(r, method, grid, model)
    dense = sum(block.synthesis == "dense_diagonal" and block.rotations > 0 for group in r.steps for block in group)
    omitted = r.range_omissions
    # Omitted nonidentity coefficients enter angle_formation, omitted identity coefficients the Walsh formation.
    walsh = sum(table.walsh_coefficients for table in omitted.tables)
    identities = sum(table.walsh_identity for table in omitted.tables)
    cutoff = method.binary_synthesis.aqft_cutoff
    exact_qft = cutoff is None or cutoff >= register_bits(grid.num_grid_points) - 1
    return [
        entry("angle_formation", "bound", formation,
              "computed Walsh coefficients and angles, dense phases and QFT angles from binary64 pi against the "
              "exact stored-weight split product, before pruning, under the one-ulp premise for math.sin, "
              "math.cos and NumPy's sin"
              + (f", with Walsh coefficients omitted below the normal binary64 range ({walsh})" if walsh else ""))
        if formation is not None else entry("angle_formation", "unavailable", None,
                                            "a table value, kinetic energy or coefficient is not finite"),
        entry("aqft", "not_applicable", None, "every controlled phase of the QFT is kept") if exact_qft else
        entry("aqft", "bound", r.aqft_error_bound,
              f"full stored-angle QFTs against their truncation at cutoff {cutoff}, min(2, 2 N_s d e_F) over "
              "every kinetic conjugation (binary.qft_error_bound), with at most one ulp per library sine"),
        entry("rotation_pruning", "bound", r.pruning_error_bound,
              f"half the exact sum of the |Rz| angles that rotation_threshold removed ({r.dropped_count} "
              f"rotations), plus |x c_m| of each rotation omitted below the normal binary64 range "
              f"({omitted.rotations}) and, per dense block, the largest |x v_z| of its phase entries omitted there "
              f"({omitted.dense_entries} entries), rounded upward"),
        entry("gate_parameters", "bound", 0,
              "the Walsh rotation angles and the controlled phases reach their gates unchanged, a controlled "
              "phase's definition halving its angle exactly"),
        entry("diagonal_wrap", "unavailable", None,
              f"phase wrap, recursive means and Gray-code angles of {dense} dense phase-diagonal blocks, with no "
              "qualified error bound for NumPy's complex exponential and argument")
        if dense else entry("diagonal_wrap", "not_applicable", None, "no block uses the dense phase diagonal"),
        entry("identity_phase", "bound", phase, "native global-phase bookkeeping against the exact constant "
              "phase and the intended Walsh identity phases, the ledger, Walsh formation and native assignment "
              "allowances"
              + (", with constant contributions, Walsh identity phases and Walsh identity coefficients omitted "
                 f"below the normal binary64 range ({omitted.identity_events}, {omitted.identity_phases} and "
                 f"{identities})" if omitted.identity_events or omitted.identity_phases or identities else "")),
        entry("state_preparation", "bound", 0, "the H layer prepares the uniform state exactly, the only state "
              "that the structured binary preparation admits")
        if method.initial_state_preparation == "structured" else
        entry("state_preparation", "not_applicable", None, "initial_state_preparation='none' prepares no state"),
    ]


def circuit_resources(plan, *, synthesis_epsilon=None):
    """Return the rotations, the optional T estimate and the separate error sources of one native QHD circuit.

    ``plan`` is a QHD Plan with ``execution="quantum"`` and the structured or
    no initial-state preparation, in either encoding. ``synthesis_epsilon``
    is the total operator-norm allowance E of one circuit execution for
    Clifford replacement plus rotation synthesis. It has no default because a
    budget is a user's accuracy choice (module docstring), and without it the
    record has no projection or T estimate.

    Error sources, in the order of the stages they compare
    (``evolution_bounds``, ``circuit_errors``). The finite model uses the
    exact schedule functions with the stored tables, constant and binary64
    spacings on the Plan's grid, and the compiled product uses the stored
    step weights.

    - ``kinetic_model``: the replacement of the finite-difference stencil by
      the finite model's kinetic operator. Not applicable to the
      finite-difference model, which every one-hot circuit applies.
      Unavailable for the spectral model of a binary circuit, against which
      every later entry is measured. The two kinetic operators differ by a
      norm that grows as ``1/h**2``, and Duhamel's formula bounds the change
      of the state only through the momentum content of the whole
      trajectory (``split_step.kinetic_eigenvalues``, docs/algorithms/qhd.md,
      "Split-step classical flavor"), which the ledger does not have.
    - ``time_ordering``, ``midpoint_quadrature`` (not applicable under the
      integrated coefficient rule), ``coefficient_rounding`` and
      ``product_formula``: the stages of ``evolution_bound``. The
      coefficient residual is exact except for the quadratic (gamma > 0) and
      cubic kinetic integrals, whose first-order estimate makes it an
      estimate.
    - ``angle_formation``: one-hot, the stored angles of the kept blocks
      against their intended exponents, exact (``circuit_errors.block_errors``).
      Binary, every computed Walsh angle before pruning, the dense phases and
      the QFT angles formed from binary64 pi against the exact stored-weight
      split product (``circuit_errors.binary_angle_formation``), under the
      one-ulp premise for ``math.sin``, ``math.cos`` and NumPy's ``sin``,
      Walsh coefficients omitted below the normal range included.
    - ``aqft``: not applicable to the one-hot circuit, which applies no
      Fourier transform, and to a binary circuit with exact QFTs. With a
      cutoff it is the reconstruction's ``aqft_error_bound``, which bounds
      the full stored-angle QFTs against their truncation.
    - ``rotation_pruning``: one-hot, the blocks the circuit omits, pruned by
      ``rotation_threshold`` or omitted below the normal binary64 range,
      charged at their exact intended exponents. Binary, the
      reconstruction's ``pruning_error_bound``, half the dropped ``Rz``
      angles plus the exact charges of the rotations and dense phase entries
      omitted below the normal range (``records.QHDRangeOmissions``).
    - ``gate_parameters``: one-hot, the fused hopping gate's own parameter
      against twice the stored angle, zero since the builder doubles the
      stored angle's own product, with the projector and chain rotations
      exact power-of-two scalings of their stored angles, which the census
      admits in the normal binary64 range (``rotation_population``). Binary,
      zero, since every angle reaches its gate unchanged or halved exactly.
    - ``diagonal_wrap``: the phase wrap ``angle(exp(-i a))`` of a phase
      diagonal, the one-hot projectors on 4 to 7 qubits or the binary dense
      diagonals, whose lowering then forms recursive means and Gray-code
      angles. NumPy documents no uniform error constant for its complex
      exponential and argument, so it is unavailable when the circuit has
      such a block and not applicable otherwise.
    - ``identity_phase``: the native global-phase bookkeeping against the
      exact identity phase (``circuit_errors.phase_allowance``,
      ``circuit_errors.binary_phase_allowance``), the identity actions
      omitted below the normal range included.
    - ``state_preparation``: one-hot, the prepared state against the exact
      target (``circuit_errors.preparation_error``), a first-order estimate
      whose chain cutoff charge is exact.
      Binary, zero, since the H layer prepares the uniform state exactly. Not
      applicable without preparation.
    - ``clifford_replacement``: ``E_C`` of the synthesis projection, an
      outward bound (``clifford_distance``) describing the approximated
      construction, not the emitted circuit.
    - ``synthesis``: the requested allowance ``E - E_C``, rounded downward, of the remaining
      rotations. NWQEC 0.1.2 applies a fixed ``1e-4`` angle cleanup, groups
      angles to four significant digits and passes them as decimal strings,
      all outside the requested GridSynth tolerance, so an achieved
      synthesis error is unavailable (docs/algorithms/qhd.md,
      "Fault-tolerant resources").
    - ``compiler``: other compiler transformations, unavailable.

    For a normalized input these add by telescoping to a bound on the final
    state's 2-norm error when every entry is a bound on adjacent stages,
    against the finite model without ``kinetic_model`` and against the
    finite-difference model with it. Each entry compares two adjacent whole
    evolutions, so the entries add by the triangle inequality through the
    intermediate evolutions, with no independence assumption. Within an
    entry, for ideal factors U_j and their implementations U~_j, the
    identity
    ``U~_L ... U~_1 - U_L ... U_1 = sum_j U~_L ... U~_(j+1) (U~_j - U_j) U_(j-1) ... U_1``
    bounds the product's error by the sum of the factors' errors, because
    the unitaries around each difference have norm one. The preparation
    error adds once, because later unitaries keep its norm. Any common
    measurement then changes by at most that amount in total variation,
    capped at 1, because measurement cannot increase the trace distance
    ``sqrt(1 - |<psi|psi~>|**2)`` of two pure states, which is at most
    their phase-aligned 2-norm distance. An entry built from an estimate
    makes such a sum conditional on it, and an unavailable entry leaves it
    unavailable, so the record forms no total. Sampling, device noise and
    the optimization gap are outside this ledger. The one-hot work is one pass over the stored
    blocks with a few exact rational operations per block, and the bound's
    pass over tables and steps. The binary work reads the stored Walsh phase
    terms (``QHDReconstruction.walsh_phase``), synthesizes each distinct
    stored block once for its angles, with the table transforms of planning,
    adds the ``O(n 2**n)`` Gray-code angles of each distinct dense-diagonal
    block on n qubits and two upward reductions per table for the angle
    formation, and builds no circuit. The binary rotation census is admitted
    first against the Plan's ``QHD.max_work`` and ``QHD.max_bytes``
    (``_admit_inspection``).

    Returns:
        record (QHDCircuitResources): The rotations, the optional projection,
            the evolution bound and the error sources of the Plan's circuit.

    Raises:
        ValueError: The binary rotation census exceeds the Plan's QHD work
            or byte limit, with the model's baseline (``_admit_inspection``,
            ``binary.BinaryModel``).
    """
    from .method import _grid

    law = _native_plan(plan)
    census = _admit_inspection(plan)
    r, method = plan.reconstruction, plan.method
    grid = _grid(plan)
    binary = method.encoding == "binary"
    model = _inspection_model(grid, method, r.support_values, census[2]) if binary else None
    population = (_binary_population(r, model, *census[:2]) if binary
                  else _admitted_onehot_population(method, r))
    projection = None if synthesis_epsilon is None else synthesis_projection(population, synthesis_epsilon)
    bound = evolution_bound(plan)

    def entry(name, status, value, description):
        # A distance between unitaries is at most 2, so every value but a requested budget is capped there.
        if value is not None:
            value = _upward(value) if status == "requested" else min(2.0, _upward(value))
        return QHDErrorSource(name=name, status=status, value=value, description=description)

    midpoint = method.coefficient_rule == "midpoint"
    factors = "potential and whole-kinetic factors with exact QFTs" if binary else "potential and link factors"
    sources = [
        entry("kinetic_model", "unavailable", None,
              "the spectral kinetic replaces the finite-difference stencil, and the state change it causes depends "
              "on the momentum content of the whole trajectory, since the two operators differ by a norm that "
              "grows as 1/h**2")
        if method.kinetic_model == "spectral" else
        entry("kinetic_model", "not_applicable", None, "the circuit applies the finite-difference stencil"),
        entry("time_ordering", "bound", bound.time_ordering,
              "exact time-ordered step against the exponential of the interval integrals"),
        entry("midpoint_quadrature", "bound", bound.midpoint_quadrature,
              "interval integrals against the midpoint values of the schedule weights")
        if midpoint else entry("midpoint_quadrature", "not_applicable", None,
                               "the integrated coefficient rule uses the interval integrals"),
        entry("coefficient_rounding", bound.coefficient_status, bound.coefficient_residual,
              bound.coefficient_unavailable or "exact products of dt and the stored step weights against the "
              "exact midpoint values or interval integrals, with the first-order integral estimate for the "
              "quadratic and cubic kinetic integrals"),
        entry("product_formula", "bound", bound.splitting,
              f"{bound.formula} splitting of the frozen step into its {factors}"),
    ]
    sources += (_binary_sources(entry, r, method, grid, model) if binary
                else _one_hot_sources(entry, r, method, grid))
    if projection is None:
        sources += [
            entry("clifford_replacement", "not_applicable", None, "no synthesis budget given, so no rotation is "
                  "replaced"),
            entry("synthesis", "unavailable", None, "no synthesis budget given"),
        ]
    else:
        sources += [
            entry("clifford_replacement", "bound", projection.replacement_error,
                  "outward sum of the distances 2 sin(|delta|/4) of the rotations replaced at the budget's "
                  "threshold, describing the approximated construction, not the emitted circuit"),
            entry("synthesis", "requested",
                  _downward(Fraction(projection.epsilon) - Fraction(projection.replacement_error)),
                  "requested allowance of the remaining rotations, split evenly, whose achieved synthesis error "
                  "is unavailable"),
        ]
    sources.append(entry("compiler", "unavailable", None,
                         "compiler cleanup, angle grouping and angle transport, not bounded"))
    return QHDCircuitResources(
        plan_id=plan.content_id, arbitrary_rotations=population.arbitrary_rotations, exact_t=population.exact_t,
        rotation_law=law, synthesis=projection, evolution=bound, error_sources=tuple(sources),
    )


class QHDRunEntry(Record):
    """The circuit of one augmented-Lagrangian round or refinement level and its multiplicities.

    Attributes:
        label: ``"round k"``, ``"level z"`` or, for a round with box
            refinement, ``"round k level z"``.
        plan_id: Content identity of the inner Plan, or None when no Plan was
            selected or the record does not name one.
        circuits: Circuit preparations of its Run, the compiled bodies.
        shots: Raw shots reserved by its attempts of every status, which
            bound the shots it executed from above.
        arbitrary_rotations: Arbitrary rotations of its circuit, the law
            value that its resources recorded, which for the binary encoding
            counts rotation gates before angle classification.
        t_estimate: T estimate of its circuit (``synthesis_projection``).
        unavailable: ``(field, reason)`` for every per-circuit value that is None.
    """

    label: Text
    plan_id: ContentID | None
    circuits: Count | None
    shots: Count | None
    arbitrary_rotations: Count | None
    t_estimate: Nonnegative | None
    unavailable: tuple[tuple[Text, Text], ...] = ()


class QHDRunResources(Record):
    """Rotation and T totals of an augmented-Lagrangian run or a box refinement over its circuits and shots.

    Two populations are kept apart. The body totals count every compiled
    circuit once (``circuits`` of each entry), the static inventory of what a
    compiler synthesizes. The shot totals weight each circuit by the raw
    shots reserved by its attempts of every status, ``sum_e shots_e R_e``,
    because each executed shot runs the whole circuit, state preparation
    included, and reserved shots bound the executed ones from above. Exact
    readout acquires no shots, while hardware would need some, so its shot
    totals are None and ``unavailable`` says to multiply each entry's
    per-circuit values by the intended shots. Per circuit the
    counts are those of ``circuit_resources``. Each round and level uses its
    own tables, spacing and exponents, so its own circuit is counted rather
    than the first one multiplied. A total is None when an entry with a
    positive or unknown multiplicity has no per-circuit value, and
    ``unavailable`` names it. An entry with zero multiplicity contributes
    zero. A sum of T estimates is an estimate.

    The rotation totals add the recorded laws. For the binary encoding the
    law (``method.QHD._select_binary_native``) counts the rotation gates
    before angle classification, exact T gates and the omitted zero angles
    of dense diagonals included, so those totals are upper bounds, while the
    T estimates use the exact angle classification of ``circuit_resources``.

    Attributes:
        synthesis_epsilon: The per-circuit budget E of the T estimates, or None.
        rotation_interpretation: ``"upper_bound"`` when the run uses the
            binary encoding, a live Plan's law is an upper bound, or a round
            or level without a live result adds a positive recorded count
            with a positive or unknown multiplicity, because its record does
            not keep the interpretation of that law. ``"exact"`` otherwise.
        entries: One entry per started round, or per started level of a
            refinement, in order.
        arbitrary_rotations: ``sum_e circuits_e R_e``.
        shot_arbitrary_rotations: ``sum_e shots_e R_e``, None under exact readout.
        t_estimate: ``sum_e circuits_e T_e``.
        shot_t_estimate: ``sum_e shots_e T_e``, None under exact readout.
        unavailable: ``(total, reason)`` for every total that is None.
    """

    synthesis_epsilon: Nonnegative | None
    rotation_interpretation: Literal["exact", "upper_bound"]
    entries: tuple[QHDRunEntry, ...]
    arbitrary_rotations: Count | None
    shot_arbitrary_rotations: Count | None
    t_estimate: Nonnegative | None
    shot_t_estimate: Nonnegative | None
    unavailable: tuple[tuple[Text, Text], ...] = ()


def _run_entry(label, plan_id, resources, result, epsilon, held_bytes=0):
    """Return the ``QHDRunEntry`` of one round or level from its record, its resources and its inner result, if any.

    ``held_bytes`` prices the retained inner Results that stay live while
    this entry's Plan is inspected (``_plan_population``).

    The rotation count is the per-circuit law value that the round's or
    level's resources already hold (``_outer.law_count``), with its recorded
    reason when it is unknown. The T estimate needs the Plan's stored blocks,
    so it comes from the live inner result and is unknown without one. The
    recorded resources keep only the rotation count, which fixes neither the
    Clifford replacements, which depend on each angle's distance from a
    Clifford angle (``synthesis_projection``), nor the exact T gates, so no
    estimate is formed from them, and the entry states that reason.
    """
    counts = dict(resources.unavailable)
    reasons = [("arbitrary_rotations", counts["arbitrary_rotations"])] if resources.arbitrary_rotations is None else []
    t = None
    if result is None:
        reasons.append(("t_estimate", "its QHD result is not available, and the T estimate needs the rotation angles "
                                      "of its Plan, since the recorded rotation count fixes neither the Clifford "
                                      "replacements nor the exact T gates"))
    elif epsilon is None:
        reasons.append(("t_estimate", "no synthesis budget given"))
    elif resources.arbitrary_rotations is None:
        reasons.append(("t_estimate", counts["arbitrary_rotations"]))
    else:
        plan = result.plan
        projection = synthesis_projection(_plan_population(plan, held_bytes), epsilon)
        if projection.t is None:
            reasons.append(("t_estimate", projection.t_unavailable))
        else:
            t = float(projection.t.value if isinstance(projection.t.value, int) else projection.t.value.value)
    return QHDRunEntry(label=label, plan_id=plan_id, circuits=resources.circuit_preparations, shots=resources.shots,
                       arbitrary_rotations=resources.arbitrary_rotations, t_estimate=t, unavailable=tuple(reasons))


def _total(entries, multiplicity, value):
    """Return ``(sum_e multiplicity_e value_e, None)`` or ``(None, reason)`` (``QHDRunResources`` states the rule)."""
    total, missing = 0, []
    for entry in entries:
        count, per = getattr(entry, multiplicity), getattr(entry, value)
        if count == 0:
            continue
        if count is None:
            missing.append(f"{entry.label}: its {multiplicity} are unknown")
        elif per is None:
            missing.append(f"{entry.label}: {dict(entry.unavailable)[value]}")
        else:
            total += count * per
    return (None, "; ".join(missing)) if missing else (total, None)


def run_resources(result, *, synthesis_epsilon=None):
    """Return the rotation and T totals of a quantum augmented-Lagrangian run or box refinement.

    ``result`` is a ``ConstrainedQHDResult``, with or without box refinement
    in each round, or a ``BoxRefinementResult``, with
    ``execution="quantum"``. Each started round, or each started level of a
    round's refinement, contributes the circuit preparations, reserved shots
    and per-circuit rotation count that its resources recorded
    (``ALResources``, ``RefinementResources``), and the T estimate of its
    live inner result's Plan. A round or level
    whose Run raised, and a refinement level that stopped the run, keep
    their recorded counts but have no inner result, so their T estimate is
    unknown when they prepared a circuit. A Plan with Qiskit's state
    preparation has no rotation law, which its resources record as unknown.
    ``synthesis_epsilon`` is the per-circuit budget of ``circuit_resources``.
    Nothing is planned, compiled or executed. The work is one pass over each
    inner Plan's stored blocks, which for the binary encoding synthesizes
    each distinct block's diagonal once more to read its angles, after the
    census is admitted against that Plan's QHD limits (``_admit_inspection``).

    Raises:
        ValueError: The run used classical execution, which selects no circuit,
            or an inner binary Plan's rotation census exceeds that Plan's QHD
            work or byte limit.
    """
    from .constrained import ConstrainedQHDResult

    def levels(prefix, refinement, results):
        # One row per completed level with its live result, and one for the level that stopped it.
        rows = [(f"{prefix}level {level.level}", level.plan_id, level.resources, inner)
                for level, inner in zip(refinement.levels, results, strict=True)]
        if refinement.stopped_resources is not None:
            rows.append((f"{prefix}level {len(refinement.levels) + 1}", None, refinement.stopped_resources, None))
        return rows

    if isinstance(result, ConstrainedQHDResult):
        execution, rows, encoding = result.record.execution, [], result.record.qhd.encoding
        shots = result.record.shots
        for item, inner in zip(result.record.iterations, result.results, strict=True):
            if item.refinement is None:
                rows.append((f"round {item.iteration}", item.plan_id, item.resources, inner))
            else:
                rows += levels(f"round {item.iteration} ", item.refinement, inner or ())
    else:
        execution, rows, encoding = result.execution, levels("", result, result.results), result.qhd.encoding
        shots = result.shots
    if execution != "quantum":
        raise ValueError("run_resources describes the circuits of a run with execution='quantum'")
    epsilon = None if synthesis_epsilon is None else float(synthesis_epsilon)
    # The retained inner Results stay live while each Plan is inspected; their kept states, 16 bytes per
    # entry, and the tables of the other distinct Plans, 8 bytes per entry, are the caller's retained data
    # of the census admission (_admit_inspection), which reserves the inspected Plan's own tables once.
    states = sum(16 * inner.data.artifact(inner.artifact).array.size
                 for *_, inner in rows if inner is not None and inner.artifact is not None)
    plans = {id(inner.plan): inner.plan for *_, inner in rows if inner is not None}
    table_bytes = {key: sum(8 * t.values.array.size for t in plan.reconstruction.support_values)
                   for key, plan in plans.items()}
    total_tables = sum(table_bytes.values())
    entries = tuple(
        _run_entry(*row, epsilon,
                   states + total_tables - (table_bytes[id(row[-1].plan)] if row[-1] is not None else 0))
        for row in rows)
    totals, reasons = {}, []
    for name, multiplicity, value in (("arbitrary_rotations", "circuits", "arbitrary_rotations"),
                                      ("shot_arbitrary_rotations", "shots", "arbitrary_rotations"),
                                      ("t_estimate", "circuits", "t_estimate"),
                                      ("shot_t_estimate", "shots", "t_estimate")):
        if shots is None and multiplicity == "shots":
            # Zero would read as zero executed work, while hardware needs shots that the run did not choose.
            totals[name], reason = None, ("exact readout acquires no shots. Multiply each entry's per-circuit "
                                          "value by the intended shots")
        else:
            totals[name], reason = _total(entries, multiplicity, value)
        if reason is not None:
            reasons.append((name, reason))
    for name in ("t_estimate", "shot_t_estimate"):
        if totals[name] is not None:
            totals[name] = float(totals[name])
    # A recorded positive count without its Plan may come from an upper-bound law, so it is not proved exact.
    unproved = any(
        inner is None and resources.arbitrary_rotations and any(
            count is None or count > 0 for count in (resources.circuit_preparations, resources.shots))
        for *_, resources, inner in rows)
    upper = encoding == "binary" or unproved or any(
        law.interpretation == "upper_bound" for *_, inner in rows if inner is not None
        for selection in inner.plan.construction.selections for law in selection.resource_laws
        if law.metric == "arbitrary_rotations")
    return QHDRunResources(synthesis_epsilon=epsilon, rotation_interpretation="upper_bound" if upper else "exact",
                           entries=entries, **totals, unavailable=tuple(reasons))
