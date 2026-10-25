"""ADAPT query construction, data collection through the common Run, and the ADAPT iteration.

The iteration is ADAPT-GCiM (Zheng et al. 2024, arXiv:2312.07691v3). Each
query runs once through the Run, which handles preparation, submission and
checkpoints, so resume retrieves the original work instead of repeating it.
See docs/development/execution.md and docs/run_archives.md for that guarantee.
"""

from dataclasses import dataclass, field
import json
from math import fsum, isfinite, pi
import numpy as np
from nwqlib._numerics import normalize_state_vector, stable_vector_norm
from nwqlib._projected_eigensolver import gram_formation_allowance
from nwqlib.blocks.kernels import BoundKernel, KernelOutput
from nwqlib.core.planning import Realization, RuntimeOptions
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.evidence.error_model import exact_readout_sampling
from nwqlib.execution import (
    PauliValue,
    ScalarValue,
    KernelApplication,
    ObservationChunk,
    PendingPreparation,
)
from nwqlib._prepared_execution import prepare_experiment, submit_experiment, restore_prepared
from nwqlib.ir import Binding
from nwqlib.ir.expressions import number
from nwqlib.operators._pauli import apply_terms
from nwqlib.operators.access import _check_bytes, _check_products
from nwqlib.problems.inputs import prepare_qiskit
from .adapt_actions import action_sizes, packed_exponential
from .adapt_records import (
    ADAPTResult,
    AdaptRound,
    BoundedCandidate,
    active_screen_labels,
    chain,
    commutator_rows,
    energy_labels,
    label_cache,
    screen_labels,
)
from . import pencil_workflow
from .fixed_basis import (
    GroupMoment,
    PairPreparation,
    ProjectedPencil,
    ShotAllocation,
    _exact_overlap_allowance,
    grouped_pass_requirements,
    pencil_pairs,
    pooled_statistics,
    processed_pauli_enclosure,
    reduce_group,
    sampled_setting_count,
    sampled_settings,
    solve_pencil,
)
from .pencil import (
    _projected_solve_bytes,
    _projected_solve_work,
    assemble_pencil,
    matrix_element_count,
    pair_count,
    assemble_pair_pencil,
    matrix_elements,
)


@dataclass
class AdaptContext:
    """Live state of one ADAPT iteration.

    A run archive (``adapt_archive.save_context``) saves the state, action
    and gate caches, the shift rules, the compiler plans and the current
    pencil. The observation index is rebuilt from the Run's chunks, and
    attributes noted as excluded are rebuilt on demand.

    Attributes:
        vectors: Classical prefix states keyed by generator chain, read-only once stored.
        h_columns: Classical ``(H - cI)|psi>`` (``_pair_scalars``) for chains
            used as a pair's right side.
        preparations: Composed Qiskit preparations keyed by (chain, controlled, inverse).
        gates: Qiskit generator gates keyed by (pool index, angle, controlled, inverse).
        adjoints: Realized adjoints of uncontrolled forward generator gates,
            ``inverse_realized_gate(forward, native_ucg=False)`` of the
            ``(index, angle, False, False)`` entry of ``gates``, keyed by
            (pool index, angle), for the ``"vacuum_restore"`` construction
            (``_realized_adjoint``). The ``inverse=True`` entries of ``gates``
            instead evaluate the generator again at the negated angle. The
            map is separate so that ``_keep_current`` keeps it only for the
            current basis and query chains, not for ``pending_items``. It
            is never saved, and a resumed Run builds each adjoint again from
            its restored forward gate.
        reference_gates: Qiskit reference preparation variants.
        observations: The observation chunk kept for each resolved query, keyed by its content hash.
        contribution_order: Content hashes of the chunks in the order first collected.
        contribution_seen: Set view of ``contribution_order`` for membership tests.
        accepted: Current selected chain, whose ``basis_chains`` the caches
            must keep under the default growth.
        basis: The current explicit chains under ``BoundedRitzGrowth``,
            which the caches keep instead, or None under the default growth.
        pending_items: While a bounded round obtains the rows of its
            admitted extensions, the ``(pool index, theta)`` last item of
            each extension still to be evaluated when the rows phase starts
            or resumes, not refreshed as each row completes, and empty at the
            start of every other phase. An exit from rows leaves it set. ``_keep_current`` keeps the gates of these items, so a
            gate that depth screening built is reused by the queries of the
            same uninterrupted rows phase. The set adds at most
            ``candidates_per_round`` items, and so at most four times as many
            gate variants, because ``controlled`` and ``inverse`` are part of
            the gate key. That counts cache entries, not compiled memory.
            It does not make a generator's gate built once per run: a gate
            pruned in an earlier round may be built again later, while its
            CX count stays in ``depth_weights``. Gates of depth-rejected
            extensions leave at the next prune, a returned Run can still
            hold the context, and a resumed rows phase may build a gate once
            more after the prune that precedes the loop. It is never saved.
        pencil: Latest solved projected pencil.
        pencil_parameters: Chain that produced ``pencil``.
        pencil_attempt: Analysis attempt of the iteration that produced ``pencil``.
        matrix_data: Current ``(H0, S, sampled, allowance)`` workspace between
            data collection and projection (``_matrix_from_observations``, or
            ``_grouped_matrices`` in ``analyze_adapt``). Result copies and saved archives exclude it and rebuild it from
            observations.
        pair_values: Values ``_read_values`` returned for each read pencil
            query, with the preparation record of its chunk, keyed by the
            canonical chain pair, term and quadrature (``_matrix_entries``).
            A grouped setting keeps its canonical ``GroupMoment``, the
            preparation record and the stored bin count of its chunk, keyed
            by the canonical chain pair, quadrature and group.
            Later matrix stages reuse them instead of reading the same
            observations again. Result copies and saved archives exclude
            them, and a restored or reanalyzed context rebuilds them from
            observations.
        rules: Resolved shift rule per selected pool index.
        depth_weights: Under ``BoundedRitzGrowth(basis_depth_limit=...)``,
            ``(w, reason)`` of ``bounded_growth.generator_cx_weight`` per
            examined ``(pool index, theta)``, read once from that generator's
            uncontrolled forward gate and kept for the whole run, because
            the gate cache keeps only the generators of current chains.
        compiler_plans: The quantum Plan's ``adapt.CompilerPlans`` map once
            ``drive_adapt`` has taken it over, so checkpoints save the compiler
            plans built after the Plan archive was written. Before that, a
            reopened Run holds the saved ``(pool index, kind, active modes,
            occupation blocks)`` entries here. Result copies store the
            built entries (``ADAPT.snapshot_result_context``), which
            circuit-level verification uses.

    The Qiskit caches (``gates``, ``adjoints``, ``preparations``,
    ``reference_gates``) hold mutable Qiskit objects and stay private to the
    live Run. Result copies share only the immutable scientific entries.
    """

    vectors: dict = field(default_factory=dict)
    h_columns: dict = field(default_factory=dict)
    preparations: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    adjoints: dict = field(default_factory=dict)
    reference_gates: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    contribution_order: list = field(default_factory=list)
    contribution_seen: set = field(default_factory=set)
    accepted: tuple = ()
    basis: tuple | None = None
    pending_items: frozenset = frozenset()
    pencil: ProjectedPencil | None = None
    pencil_parameters: tuple | None = None
    pencil_attempt: int | None = None
    matrix_data: tuple | None = None
    # The Run's shifted sparse Hamiltonian, keyed by operator identity, exact
    # shift, representation and dtype. It is rebuilt on demand and never saved.
    shifted: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    pair_values: dict = field(default_factory=dict)  # Excluded from snapshots/archives.
    compiler_plans: object = None
    depth_weights: dict = field(default_factory=dict)


def basis_chains(selected):
    """Return the ADAPT-GCiM working basis for the selected chain, in fixed order.

    A chain ``((i_1, t_1), ..., (i_k, t_k))`` denotes the state
    ``exp(t_k A_{i_k}) ... exp(t_1 A_{i_1})|ref>``, applying its first entry
    first. The order is the reference, each single-generator state
    ``G_j|ref>``, the full product, then the products of the first
    ``2..k-1`` generators. The one-generator product equals ``G_1|ref>`` and
    appears once. This gives 2k states after k >= 1 selections. Zheng et al.
    (2024), arXiv:2312.07691v3, METHODS, p. 9, add ``G_k|ref>`` and
    ``G_k ... G_1|ref>`` at iteration k, which yields the same set of states.
    Pencil rows, saved coefficients and residual checks all use this order.
    """
    count = len(selected)
    chains = [()] + [(item,) for item in selected]
    if count >= 2:
        chains += [selected, *(selected[:i] for i in range(2, count))]
    return tuple(chains)


def current_basis(context):
    """Return the chains of the current basis: ``context.basis`` under bounded growth, else ``basis_chains(context.accepted)``."""
    return basis_chains(context.accepted) if context.basis is None else context.basis


def _value_chains(state, bounded):
    """Return the chains of the basis behind the saved value: ``value_basis`` under bounded growth, else ``basis_chains`` of ``value_selected`` and ``value_theta``."""
    if bounded:
        return saved_chains(state.get("value_basis", ()))
    return basis_chains(tuple(zip(state.get("value_selected", ()), state.get("value_theta", ()), strict=True)))


def saved_chains(value):
    """Return chains saved as JSON lists, ``[[[index, theta], ...], ...]``, as tuples of ``(int, float)`` items."""
    return tuple(tuple((int(index), float(theta)) for index, theta in chain) for chain in value)


def saved_cursor(value):
    """Return a bounded frontier cursor saved as JSON lists, ``[[parent position, next pool index], ...]``, as integer pairs."""
    return tuple((int(parent), int(index)) for parent, index in value)


def common_prefix(left, right):
    """Return the number of equal leading items of two generator chains.

    Items ``(pool index, theta)`` match only when the index and the binary64
    angle are equal, so both chains apply the same generator gates to the
    same reference in the same order. Two chains with the same generator at
    different angles share no item. Every chain starts from the Plan's one
    reference preparation, which is therefore always common. No algebraic
    equivalence is searched.
    """
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def pair_route(left, right, rule):
    """Return the construction route of the off-diagonal pair of chains ``left`` and ``right`` under the Plan's ``rule``.

    ``"vacuum_reference"`` uses its reference-row construction only for a
    pair whose left chain is the reference ``()`` and the ``"prefix"``
    construction for every other pair. Every other rule is its own route,
    so ``"vacuum_restore"`` covers the reference row too, where the
    restoration with an empty left chain is the reference-row circuit. The
    reference sorts first among chains, so it is the left chain of every
    canonical and every basis-order pair it belongs to.
    """
    if rule == "vacuum_reference" and left:
        return "prefix"
    return rule


def chain_pair_preparation(left, right, rule, pair):
    """Return the ``PairPreparation`` of the chains ``left``, ``right`` at basis positions ``pair``.

    Under the ``"prefix"`` route the reference and the common leading items
    run once without control, ``1 + common_prefix(left, right)`` blocks.
    Under ``"controlled"`` both complete preparations are controlled, and
    the vacuum routes control only the reference (``pair_route``,
    ``_append_vacuum_pair``).
    """
    route = pair_route(left, right, rule)
    if route == "prefix":
        return PairPreparation(pair=pair, route="prefix", shared_blocks=1 + common_prefix(left, right))
    return PairPreparation(pair=pair, route=route)


def _keep_current(context, basis, *queries):
    """Drop cached states, actions and gates that no current basis or query can reuse.

    ``basis`` is the tuple of chains of the current basis, and ``queries``
    are the chains of the active queries. Only prefixes of these chains are
    kept. The gate cache also keeps the items of
    ``context.pending_items``, while the state, column, pair-value and
    preparation caches keep only these prefixes, so the bounded
    ``2(B + 2L)`` count of the classical caches is unchanged. The realized
    adjoints keep only the items of these chains, so the pending items add
    at most their four gate variants each and no adjoint. Cache memory
    therefore stays proportional to the current basis instead of growing
    with the whole trajectory. A dropped entry is recomputed if a later
    query needs it again.
    """
    keep = {()}
    for items in (*basis, *queries):
        keep.update(items[:i] for i in range(len(items) + 1))
    for cache in (context.vectors, context.h_columns):
        for key in tuple(cache):
            if key not in keep:
                del cache[key]
    for key in tuple(context.pair_values):
        if key[0] not in keep or key[1] not in keep:
            del context.pair_values[key]
    for key in tuple(context.preparations):
        if key[0] not in keep:
            del context.preparations[key]
    items = {item for prefix in keep for item in prefix}
    pairs = items | context.pending_items
    for key in tuple(context.gates):
        if key[:2] not in pairs:
            del context.gates[key]
    for key in tuple(context.adjoints):
        if key not in items:
            del context.adjoints[key]


def _native_preparation(
    items, *, controlled, inverse, inputs, reference, compiler_plans, policy, context,
    max_direct_amplitudes
):
    """Return the cached gate ``U = exp(t_k A_k) ... exp(t_1 A_1) U_ref`` of chain ``items``.

    ``U_ref`` prepares the reference from ``|0>``. With ``controlled`` the gate
    acts on an extra control qubit 0, and with ``inverse`` it is
    ``U^dagger = U_ref^dagger exp(-t_1 A_1) ... exp(-t_k A_k)``. Each factor
    comes from ``build_generator_circuit`` with its compiler plan from the Plan's map, so
    every query of one Plan uses the same synthesis. The physical global
    phase of the reference is kept, because under control it becomes a
    relative phase. ``max_direct_amplitudes`` is the circuit-construction
    limit that the reference preparation was checked against.
    """
    from qiskit import QuantumCircuit

    key = (items, controlled, inverse)
    if key in context.preparations:
        return context.preparations[key]
    q = reference.manifest.basis.dimension.bit_length() - 1
    off = int(controlled)
    reference_gate = _reference_gate(controlled, inverse, reference=reference, policy=policy, context=context,
                                     max_direct_amplitudes=max_direct_amplitudes)
    circuit = QuantumCircuit(q + off)
    if not inverse:
        circuit.append(reference_gate, circuit.qubits, copy=False)
    # An adjoint reverses the generator order as well as each gate action.
    # The reference inverse must then appear after the reversed product.
    for index, theta in reversed(items) if inverse else items:
        circuit.append(
            _native_generator(
                index,
                theta,
                controlled=controlled,
                inverse=inverse,
                inputs=inputs,
                compiler_plans=compiler_plans,
                context=context,
            ),
            circuit.qubits,
            copy=False,
        )
    if inverse:
        circuit.append(reference_gate, circuit.qubits, copy=False)
    gate = circuit.to_gate()
    context.preparations[key] = gate
    return gate


def _reference_gate(controlled, inverse, *, reference, policy, context, max_direct_amplitudes):
    """Return the cached reference preparation gate, controlled on a qubit 0 before the system or inverted as requested.

    The uncontrolled gate comes from ``prepare_qiskit`` with the reference's
    physical global phase, its controlled form from ``qiskit_compat.controlled``
    and an inverse from ``inverse_realized_gate`` of the controlled or
    uncontrolled form. Each variant is built once per Run context.
    """
    from nwqlib.subroutines.qiskit_compat import (
        controlled as controlled_gate,
        inverse_realized_gate,
    )

    rkey = (controlled, inverse)
    if rkey not in context.reference_gates:
        if (False, False) not in context.reference_gates:
            circuit = prepare_qiskit(
                reference, max_bytes=policy.max_bytes, max_direct_amplitudes=max_direct_amplitudes
            ).circuit
            context.reference_gates[(False, False)] = circuit.to_gate()
        gate = context.reference_gates[(False, False)]
        if controlled:
            if (True, False) not in context.reference_gates:
                context.reference_gates[(True, False)] = controlled_gate(gate, 1)
            gate = context.reference_gates[(True, False)]
        if inverse:
            gate = inverse_realized_gate(gate)
        context.reference_gates[rkey] = gate
    return context.reference_gates[rkey]


def _native_generator(index, theta, *, controlled, inverse, inputs, compiler_plans, context):
    """The one generator-gate factory used by queries and explicit verification."""
    from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

    key = (index, theta, controlled, inverse)
    if key not in context.gates:
        context.gates[key] = build_generator_circuit(
            inputs.pool[index],
            theta,
            controlled=controlled,
            inverse=inverse,
            _plan=compiler_plans[index],
        ).to_gate()
    return context.gates[key]


# Rotation instructions of apply_pauli_rotation: one per nonzero nonidentity
# Pauli row of a commuting generator, in row order.
_PAULI_ROTATIONS = frozenset(("rx", "ry", "rz", "rxx", "ryy", "rzz"))
# The largest definition of a multi-controlled RY that _realized_adjoint
# copies: Qiskit 2.6.0 builds 5, 13, 64 and 96 instructions for 2, 3, 4 and 5
# controls (docs/ENGINEERING_CONSTANTS.md).
_CONTROLLED_RY_COPY_LIMIT = 96


def _finite_parameters(operation):
    """Return whether every parameter of ``operation`` is a finite number."""
    try:
        return all(isfinite(float(value)) for value in operation.params)
    except TypeError:
        return False


def _vacuum_phase(index, theta, *, inputs, compiler_plans, context):
    """Return the vacuum phase components of one occurrence of a generator in a vacuum pair construction.

    The generator's uncontrolled forward gate ``U`` (``_native_generator``)
    passed the structural vacuum rule of ``adapt._admit_generator_routes``
    at planning, which gives ``U|vac> = exp(i gamma)|vac>``. ``gamma`` is
    kept as a tuple of components, whose ancilla phase gates multiply to
    ``exp(i gamma)`` exactly, without a rounded binary64 sum or a reduction
    modulo 2 pi:

    - the actual stored global phase g of ``U``'s circuit, which holds the
      identity rows (``build_generator_circuit``), so ``theta * c_I`` is
      never added a second time;
    - for a commuting generator, ``-phi_P / 2`` for each nonidentity word P
      of only I and Z letters, with ``phi_P`` the emitted angle of P's
      rotation ``exp(-i phi_P P / 2)``, because ``P|vac> = |vac>``. The
      rotations are read from the circuit, one per nonzero nonidentity row
      in row order, and never computed again from theta and the
      coefficient: for ``theta = 2**1023`` and ``c = 2**-1023`` the
      constructor's ``-2.0 * theta`` overflows, so the emitted angle is
      nonfinite although the ideal product is -2.

    A noncommuting default double has no rotation component, since its
    complete circuit maps the vacuum to itself with phase 0, and exact
    zeros are left out. The instance qualifies only when g and the
    parameters of every instruction of ``U``'s circuit, the emitted angles
    among them, are finite and, for each component, ``h = RN(phi_P / 2)`` is finite
    with ``2 h == phi_P``, which fails only where the half of ``phi_P`` is
    not a binary64 number, an odd multiple of the smallest subnormal.
    Otherwise ``ApplicabilityError`` refuses the query that needs this
    instance before its acquisition, and the pair is never prepared by
    another construction. The rule is sufficient, not necessary. The
    components describe the represented gates in exact arithmetic and say
    nothing about synthesis or device error.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError

    gate = _native_generator(index, theta, controlled=False, inverse=False, inputs=inputs,
                             compiler_plans=compiler_plans, context=context)
    definition = gate.definition
    g = float(definition.global_phase)
    if not isfinite(g) or not all(_finite_parameters(item.operation) for item in definition.data):
        raise ApplicabilityError(f"pool generator {index} at angle {theta!r} has a nonfinite emitted angle or "
                                 "global phase, so its vacuum phase is unknown")
    components = [g] if g else []
    if compiler_plans[index].kind != "commuting":
        return components
    rows = [label for label, value in inputs.pool[index].pauli_terms
            if complex(value).imag != 0.0 and set(label) != {"I"}]
    rotations = [item.operation for item in definition.data if item.operation.name in _PAULI_ROTATIONS]
    if len(rotations) != len(rows):
        raise ApplicabilityError(f"pool generator {index} at angle {theta!r} has {len(rotations)} Pauli rotations "
                                 f"for {len(rows)} nonzero terms, so its vacuum phase is unknown")
    for label, rotation in zip(rows, rotations):
        if set(label) <= {"I", "Z"}:
            phi = float(rotation.params[0])
            half = phi / 2
            if rotation.name not in ("rz", "rzz") or not isfinite(half) or 2 * half != phi:
                raise ApplicabilityError(f"pool generator {index} at angle {theta!r} has a diagonal rotation by "
                                         f"{phi!r} whose half angle is not a binary64 number, so its vacuum phase "
                                         "is unknown")
            if half:
                components.append(-half)
    return components


def _realized_adjoint(index, theta, *, inputs, compiler_plans, context):
    """Return the cached adjoint of the realized uncontrolled forward gate of one generator, for ``"vacuum_restore"``.

    The adjoint is ``inverse_realized_gate(forward, native_ucg=False)`` of
    the cached gate ``forward = _native_generator(index, theta,
    controlled=False, inverse=False)``. It reverses that realized definition
    and inverts each instruction, so it is the adjoint of the same rounded
    gates. ``build_generator_circuit(inverse=True)``, the ``inverse=True``
    entry of the gate cache, instead evaluates the generator again at
    ``-theta``, which is only the same ideal operator. The adjoint is built
    once per ``(index, theta)`` in ``context.adjoints``.

    Before the helper copies anything, the forward gate must have the
    structure for which it copies only existing definitions and inverts
    only standard gates: a plain ``Gate`` with a definition, whose finite
    instructions are standard gates on the helper's standard-inverse
    branch, a plain ``ucry`` multiplexor whose existing definition holds
    only RY and CX, or a multi-controlled RY with at least two closed
    controls and an existing definition of at most 96 standard-gate
    instructions. That bound is the Qiskit 2.6.0 count for five controls,
    the largest an uncontrolled generator uses, and it limits the copy work
    that the ``"vacuum_restore"`` slot law (``adapt._construction``)
    declares, so a dependency upgrade that enlarges the definition refuses
    the instance rather than exceeding the law. Any other structure, a
    definition still to be built, an extra nesting level or a nonfinite
    parameter makes the instance unavailable: ``ApplicabilityError``
    refuses the query that needs it before its acquisition, and the pair
    is never prepared by another construction. A forward gate restored
    from a Run archive passes the same checks.
    """
    from qiskit.circuit import ControlledGate, Gate
    from qiskit.circuit.library import RYGate
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate

    key = (index, theta)
    if key in context.adjoints:
        return context.adjoints[key]
    forward = _native_generator(index, theta, controlled=False, inverse=False, inputs=inputs,
                                compiler_plans=compiler_plans, context=context)

    def standard(operation):
        # The branch of inverse_realized_gate that returns operation.inverse() without a definition.
        return (operation.base_class.inverse.__module__.startswith("qiskit.circuit.library.standard_gates.")
                and _finite_parameters(operation))

    def supported(operation):
        if standard(operation):
            return True
        # A plain Gate and a generic ControlledGate build no definition when
        # it is read, so a missing one stays None here.
        if type(operation) is Gate and operation.name == "ucry":
            definition = operation.definition
            return definition is not None and all(item.operation.name in ("ry", "cx") and standard(item.operation)
                                                  for item in definition.data)
        if (type(operation) is ControlledGate and type(operation.base_gate) is RYGate
                and operation.num_ctrl_qubits >= 2
                and operation.ctrl_state == (1 << operation.num_ctrl_qubits) - 1):
            definition = operation.definition
            return (definition is not None and len(definition.data) <= _CONTROLLED_RY_COPY_LIMIT
                    and _finite_parameters(operation) and all(standard(item.operation) for item in definition.data))
        return False

    definition = forward.definition if type(forward) is Gate else None
    if definition is None or not all(supported(item.operation) for item in definition.data):
        raise ApplicabilityError(f"pool generator {index} at angle {theta!r} has a realized circuit whose adjoint "
                                 "is not supported by the vacuum_restore construction")
    context.adjoints[key] = inverse_realized_gate(forward, native_ucg=False)
    return context.adjoints[key]


def _append_joint_state(circuit, left, right, rule, kwargs):
    """Append the phase-faithful joint state ``(|0>|left> + |1>|right>)/sqrt(2)`` on ancilla qubit 0.

    ``"controlled"``: a Hadamard and an X gate on the ancilla, the complete
    left preparation controlled on it, a second X and the complete right
    preparation controlled on it, so the zero branch holds the left state
    and the one branch the right state. ``"prefix"``: chains
    ``W_l = V_l C`` and ``W_r = V_r C`` with C the reference and the common
    leading generators (``common_prefix``) run C once without control, then
    the Hadamard, X, the controlled left suffix ``V_l``, X and the controlled
    right suffix ``V_r``. Since C acts on both branches it factors out, which
    gives the same joint state with each preparation's phase. An empty left
    suffix needs no X pair, and an empty suffix adds no gate. The
    reference's global phase is common to both branches in either
    construction, and each suffix generator keeps its controlled phase.
    ``rule`` is the Plan's resolved ``pair_preparation``. Under the vacuum
    rules ``pair_route`` selects ``_append_vacuum_pair`` or the prefix
    construction. Any other value raises ``ValueError``.
    """
    route = pair_route(left, right, rule)
    if route in ("vacuum_reference", "vacuum_restore"):
        _append_vacuum_pair(circuit, left, right, kwargs)
        return
    rule = route
    if rule == "controlled":
        circuit.h(0)
        circuit.x(0)
        circuit.append(_native_preparation(left, controlled=True, inverse=False, **kwargs), circuit.qubits,
                       copy=False)
        circuit.x(0)
        circuit.append(_native_preparation(right, controlled=True, inverse=False, **kwargs), circuit.qubits,
                       copy=False)
        return
    if rule != "prefix":
        raise ValueError(f"ADAPT joint state needs a resolved pair_preparation rule, not {rule!r}")
    shared = common_prefix(left, right)
    circuit.append(_native_preparation(left[:shared], controlled=False, inverse=False, **kwargs),
                   circuit.qubits[1:], copy=False)
    circuit.h(0)
    generator = dict(controlled=True, inverse=False, inputs=kwargs["inputs"],
                     compiler_plans=kwargs["compiler_plans"], context=kwargs["context"])
    if left[shared:]:
        circuit.x(0)
        for index, theta in left[shared:]:
            circuit.append(_native_generator(index, theta, **generator), circuit.qubits, copy=False)
        circuit.x(0)
    for index, theta in right[shared:]:
        circuit.append(_native_generator(index, theta, **generator), circuit.qubits, copy=False)


def _append_vacuum_pair(circuit, left, right, kwargs):
    """Append ``(|0>|left> + |1>|right>)/sqrt(2)`` by controlling only the reference, for the two vacuum rules.

    Write ``phi_i = U_i B|vac>`` for the reference preparation B, the
    all-zero state ``|vac>`` and the uncontrolled generator chain ``U_i``,
    with ``U_i|vac> = exp(i gamma_i)|vac>`` (``_vacuum_phase``). The
    ancilla is qubit 0, C1 is control on ancilla value 1 and C0 is X, C1,
    X, B under control is ``context.reference_gates[(True, False)]``, and
    every generator runs without control on the system. After the Hadamard
    gate puts the ancilla in ``|+>``, the sequence ``C1(B)``, ``U_j``,
    ``U_i^dagger``, ``C0(B)``, ``U_i`` takes the two branches through

        (vac, b0) -> (exp(i gamma_j) vac, U_j b0)
                  -> (exp(i (gamma_j - gamma_i)) vac, U_i^dagger U_j b0)
                  -> (exp(i (gamma_j - gamma_i)) b0, U_i^dagger U_j b0)
                  -> (exp(i (gamma_j - gamma_i)) phi_i, phi_j)

    with ``b0 = B|vac>``. ``diag(1, exp(i (gamma_j - gamma_i)))`` on the
    ancilla, a phase gate P(c) for each component c of ``gamma_j`` and
    P(-c) for each component of ``gamma_i``, multiplies both branches by the
    same scalar and leaves the target joint state. An empty left chain is
    the reference row, H, ``C1(B)``, ``U_j``, ``C0(B)`` and P(c) for
    ``gamma_j``, through ``(vac, b0) -> (exp(i gamma_j) vac, phi_j) ->
    (exp(i gamma_j) b0, phi_j)``. The correction precedes every readout
    rotation, so the real and imaginary quadratures and the exact
    reduction read the same state as the other constructions (Yoshioka et
    al., arXiv:2407.14431v3, Fig. 1b and pp. 2-3, for the controlled
    reference).

    ``U_j`` and ``U_i`` are the cached uncontrolled forward gates in item
    order and ``U_i^dagger`` their realized adjoints in reversed item order
    (``_realized_adjoint``), without B. Every component and adjoint is
    qualified before the first gate is appended, so an unavailable
    instance refuses the query before its acquisition. The ideal cost is
    two controlled B, one uncontrolled ``U_j`` and, for a left chain, two
    uncontrolled ``U_i``, one of them as its adjoint. A smaller compiled
    depth or error is not implied.
    """
    generator = dict(inputs=kwargs["inputs"], compiler_plans=kwargs["compiler_plans"], context=kwargs["context"])
    corrections = [c for item in right for c in _vacuum_phase(*item, **generator)]
    corrections += [-c for item in left for c in _vacuum_phase(*item, **generator)]
    adjoints = [_realized_adjoint(*item, **generator) for item in reversed(left)]
    reference = _reference_gate(True, False, reference=kwargs["reference"], policy=kwargs["policy"],
                                context=kwargs["context"], max_direct_amplitudes=kwargs["max_direct_amplitudes"])
    system = circuit.qubits[1:]

    def forward(items):
        for index, theta in items:
            circuit.append(_native_generator(index, theta, controlled=False, inverse=False, **generator), system,
                           copy=False)

    circuit.h(0)
    circuit.append(reference, circuit.qubits, copy=False)
    forward(right)
    for gate in adjoints:
        circuit.append(gate, system, copy=False)
    circuit.x(0)
    circuit.append(reference, circuit.qubits, copy=False)
    circuit.x(0)
    forward(left)
    for component in corrections:
        circuit.p(component, 0)


def _append_group_rotations(circuit, labels, q, offset):
    """Rotate system qubit k (circuit qubit ``k + offset``) so each QWC label reads as a Z parity: H for X, S^dagger then H for Y."""
    for qubit in range(q):
        axis = next((label[q - 1 - qubit] for label in labels if label[q - 1 - qubit] != "I"), "I")
        if axis == "Y":
            circuit.sdg(qubit + offset)
        if axis in "XY":
            circuit.h(qubit + offset)


def construct_adapt(block, arguments, method_context):
    """Registered circuit builder for the finite ADAPT parameter space.

    Its arguments are scalars that were already checked. The block payload
    carries the Plan's resolved ``pair_preparation``, and the block choice
    names the resolved measurement: exact, grouped or termwise shots. A
    ``pair`` query of any other choice, such as a Plan that resolved no
    measurement, raises ``ValueError``. An exact ``pair``
    prepares the phase-faithful joint state of its two chains
    (``_append_joint_state``), so the zero branch holds the left state and
    the one branch the right state. A diagonal exact pair prepares its system
    state alone. A grouped sampled ``pair`` prepares the same joint state
    and rotates the system for its Hamiltonian group and the ancilla for its
    quadrature, H for the real part and S^dagger then H for the imaginary
    part, which gives ``<X_a P> = Re z`` and ``<Y_a P> = Im z`` for
    ``z = <left|P|right>``. With equal chains it prepares the system state
    of the ``diagonal`` setting and rotates only the system. A termwise
    sampled ``pair`` is a Hadamard test on ancilla qubit 0. It applies a
    Hadamard gate, the controlled right preparation, the controlled Pauli
    term (for a Hamiltonian element), the controlled inverse left
    preparation, an optional S-dagger and a second Hadamard gate. The
    ancilla Z expectation is then ``Re<psi_L|P|psi_R>``, or the imaginary
    part with S-dagger. ``screen`` and ``energy`` prepare the right chain on
    the system register. A sampled query appends the basis rotations of its
    QWC group so every label in the group is measured as a Z-parity.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.hamiltonian_evolution import apply_pauli_rotation, make_pauli_label

    name, inputs, reference, compiler_plans, policy, pair_preparation = block._payload
    # drive_adapt saves this cache once per projected analysis and on return.
    context = method_context(AdaptContext, marks_changes=True)
    if type(context) is not AdaptContext:
        raise ValueError("native ADAPT query requires its actual method context")
    values = dict(arguments)
    left, right = chain(values, "l"), chain(values, "r")
    _keep_current(context, current_basis(context), left, right)
    q = reference.manifest.basis.dimension.bit_length() - 1
    pair = name == "pair"
    circuit = QuantumCircuit(q + int(pair))
    kwargs = dict(
        inputs=inputs,
        reference=reference,
        compiler_plans=compiler_plans,
        policy=policy,
        context=context,
        max_direct_amplitudes=method_context.max_direct_amplitudes,
    )
    choice = block.record.choice
    if pair and choice.startswith(("pair:exact:", "pair:shots:grouped:")):
        # Exact pair query or grouped pencil setting: the phase-faithful joint
        # state (|0>|left> + |1>|right>)/sqrt(2) with the ancilla at native
        # bit 0, or the plain system state of a diagonal pair. An exact query
        # is reduced at acquisition by the registered weighted-pencil
        # reduction, and a grouped setting adds its readout rotations.
        if left == right:
            circuit.append(
                _native_preparation(left, controlled=False, inverse=False, **kwargs),
                circuit.qubits[1:],
                copy=False,
            )
        else:
            _append_joint_state(circuit, left, right, pair_preparation, kwargs)
        if choice.startswith("pair:shots:grouped:"):
            if inputs.energy_groups:
                ordered = energy_labels((label for label, _ in inputs.nonidentity), inputs.cache)
                _append_group_rotations(
                    circuit, tuple(ordered[i] for i in inputs.energy_groups[int(values["group"])]), q, 1)
            if left != right:
                if values["quadrature"]:
                    circuit.sdg(0)
                circuit.h(0)
    elif pair and choice.startswith("pair:shots:termwise:"):
        circuit.h(0)
        circuit.append(
            _native_preparation(right, controlled=True, inverse=False, **kwargs),
            circuit.qubits,
            copy=False,
        )
        term = int(values["term"])
        if term >= 0:
            for qubit, axis in enumerate(reversed(inputs.nonidentity[term][0]), 1):
                if axis == "X":
                    circuit.cx(0, qubit)
                elif axis == "Y":
                    circuit.cy(0, qubit)
                elif axis == "Z":
                    circuit.cz(0, qubit)
        circuit.append(
            _native_preparation(left, controlled=True, inverse=True, **kwargs),
            circuit.qubits,
            copy=False,
        )
        if values["quadrature"]:
            circuit.sdg(0)
        circuit.h(0)
    elif pair:
        raise ValueError("ADAPT pair query needs an exact, grouped or termwise matrix measurement, "
                         f"not the resolved {':'.join(choice.split(':')[1:3])!r}")
    else:
        position = int(values["insert_position"])
        if position == -1:
            circuit.append(
                _native_preparation(right, controlled=False, inverse=False, **kwargs),
                circuit.qubits,
                copy=False,
            )
        else:
            # Insertion follows the complete selected exponential, not one
            # internal factor. Remaining generators preserve the derivative law.
            # The rotation angle -sign*pi/2 gives exp(i*sign*pi/4*P), the same
            # factor (I + i*sign*P)/sqrt(2) as the classical _classical_query_state.
            from nwqlib.subroutines.fermionic_circuits import build_generator_circuit

            circuit.append(
                _native_preparation(
                    right[: position + 1], controlled=False, inverse=False, **kwargs
                ),
                circuit.qubits,
                copy=False,
            )
            label = inputs.pool[right[position][0]].pauli_terms[int(values["insert_term"])][0]
            compact = make_pauli_label({i: a for i, a in enumerate(reversed(label)) if a != "I"})
            apply_pauli_rotation(circuit, compact, -int(values["insert_sign"]) * pi / 2)
            for index, theta in right[position + 1 :]:
                circuit.compose(
                    build_generator_circuit(inputs.pool[index], theta, _plan=compiler_plans[index]),
                    inplace=True,
                )
        # The Program owns measurements. Only SHOTS requests these QWC rotations.
        if block.record.choice.startswith(name + ":shots:"):
            if name == "screen":
                ordered, groups = screen_labels(inputs.members, inputs.cache), inputs.groups
                active = active_screen_labels(
                    inputs.members, inputs.cache, {i for i, _ in right}, len(inputs.pool)
                )[1]
            else:
                ordered = energy_labels((label for label, _ in inputs.nonidentity), inputs.cache)
                groups, active = inputs.energy_groups, frozenset(ordered)
            labels = tuple(ordered[i] for i in groups[int(values["group"])] if ordered[i] in active)
            _append_group_rotations(circuit, labels, q, 0)
    return circuit


def _reference_vector(plan, context, counts):
    """Return the cached normalized reference state, computing it once per Run."""
    if () not in context.vectors:
        from nwqlib.algorithms._eigen_inputs import state_direction

        context.vectors[()] = state_direction(
            plan._native["reference"], max_bytes=plan.method.max_bytes
        )
        context.vectors[()].flags.writeable = False
        counts["reference_materializations"] += 1
    return context.vectors[()]


def _vector(items, plan, context, counts):
    """Return the cached normalized product state of chain ``items``.

    The state of ``items`` is built from the cached state of its prefix
    ``items[:-1]`` by one generator exponential, so each chain in the cache
    costs one packed generator exponential. The exact unitary preserves
    the norm, so each result is renormalized to remove the Taylor and
    rounding drift, and stored read-only. The exponential acts with the
    Plan's packed generator table (``adapt_actions.packed_exponential``).
    """
    if items in context.vectors:
        return context.vectors[items]
    if not items:
        return _reference_vector(plan, context, counts)
    previous = _vector(items[:-1], plan, context, counts)
    index, theta = items[-1]
    tables = plan._native["inputs"].cache["action_tables"]
    vector, _ = normalize_state_vector(packed_exponential(tables["generators"][index], previous, theta))
    vector.flags.writeable = False
    context.vectors[items] = vector
    counts["generator_exponentials"] += 1
    return vector


def stream_ritz(basis, coefficients, reference_at, evolve, normalize, counts=None):
    """Accumulate the Ritz state in ``basis_chains`` order and free a basis or prefix array once it has contributed and its successor has been evolved.

    Product prefixes needed after the full-product contribution remain
    available. Circuit-level verification evolves the same independently built
    circuits, and the final state is normalized by the ``normalize`` callback.

    The nodes are the distinct prefixes of the basis chains. A node has one
    use for each child that must be evolved from it and one for each
    occurrence in the ordered basis. The parent's count drops after a child
    is constructed and a node's own count after its coefficient is added,
    and the array is released at count zero. For ``l`` selections the order
    is the reference, singles ``1..l``, the full product and the products
    ``1..2, ..., 1..(l-1)``, so the first prefix, which already contributed
    as single 1, is released after product ``1..2`` is built, the reference
    after the last single, and every other single right after its
    contribution. The addition order and the final normalization are those
    of the plain sum over ``basis`` in order, so the result is bitwise the
    same when the callbacks return the same arrays. A zero coefficient is
    still added. The node-set iteration order changes only the integer use
    counts, never the evolution or summation order.

    ``reference_at()`` returns the reference array and
    ``evolve(previous, chain)`` the array of ``chain`` from the array of
    ``chain[:-1]``. It receives the whole chain so a classical caller can
    return an array it already holds. ``normalize`` returns
    ``(state, norm)``. With ``counts``, each reuse of a held node adds one to
    ``counts["reuses"]``.
    """
    nodes = {items[:j] for items in basis for j in range(len(items) + 1)}
    uses = dict.fromkeys(nodes, 0)
    for items in nodes:
        if items:
            uses[items[:-1]] += 1
    for items in basis:
        uses[items] += 1
    cache = {}

    def release(items):
        uses[items] -= 1
        if uses[items] == 0:
            del cache[items]

    def get(items):
        if items in cache:
            if counts is not None:
                counts["reuses"] = counts.get("reuses", 0) + 1
        else:
            if not items:
                cache[items] = reference_at()
            else:
                previous = get(items[:-1])
                cache[items] = evolve(previous, items)
                del previous
                release(items[:-1])
        return cache[items]

    total = None
    for items, coefficient in zip(basis, coefficients, strict=True):
        vector = get(items)
        if total is None:
            total = np.zeros_like(vector)
        total += coefficient * vector
        del vector
        release(items)
    return normalize(total)[0]


def _ritz_state(basis, coefficients, reference_at, evolve, counts=None):
    """Return the normalized Ritz state ``sum_i c_i |b_i>`` of the basis chains ``basis``.

    ``|b_i>`` runs over the chains of ``basis`` and ``c_i`` over
    ``coefficients`` in the same order. The sum is streamed by
    ``stream_ritz`` from the callbacks, which supply the classical or the
    circuit-built states, and normalized by ``normalize_state_vector``. No
    Hamiltonian action is applied.
    """
    if len(coefficients) != len(basis):
        raise ValueError("Ritz coordinates differ from the exact basis order")
    return stream_ritz(basis, coefficients, reference_at, evolve, normalize_state_vector, counts)


def _classical_query_state(plan, context, counts, right, values):
    """Return the state vector of a classical query's right chain.

    Without an insertion this is the cached normalized product state of
    ``right``. With ``insert_position = j``, the Pauli factor
    ``exp(i s pi/4 P) = (I + i s P)/sqrt(2)`` is applied after the exponential
    of ``right[j]`` (zero-based), where ``P`` is Pauli term ``insert_term`` of
    that generator and ``s = insert_sign``, and the remaining generators follow.
    ``optimization._generator_shift_rule`` derives why the energy difference
    of the two signs is that term's derivative contribution. The circuit
    query in ``construct_adapt`` applies the same factor.
    """
    insertion = int(number(values["insert_position"]))
    if insertion < 0:
        return _vector(right, plan, context, counts)
    state = _vector(right[: insertion + 1], plan, context, counts)
    tables = plan._native["inputs"].cache["action_tables"]
    term = tables["insertions"][right[insertion][0]][int(number(values["insert_term"]))]
    # exp(sign*i*pi*P/4)=(I+sign*iP)/sqrt(2), an exact Pauli identity. P is
    # the packed unit-coefficient row of the generator's term.
    action = apply_terms(term, state, num_qubits=plan.reconstruction.num_qubits)
    state = (state + 1j * int(number(values["insert_sign"])) * action) / np.sqrt(2)
    counts["generator_actions"] += 1
    for index, theta in right[insertion + 1 :]:
        state = packed_exponential(tables["generators"][index], state, theta)
        counts["generator_exponentials"] += 1
    return state


def _offset_free_column(plan, context, counts, right, state):
    """Return ``(H - cI)|right>``, cached in ``context.h_columns``, computing it once on a miss.

    ``state`` is the normalized product state of chain ``right``. ``c`` is
    ``plan.reconstruction.identity_shift``. The nonidentity Pauli terms act
    directly through the Plan's packed H0 table, for zero and nonzero c
    alike, and a dense or sparse matrix has c removed from its stored
    diagonal first (``fixed_basis._offset_free_actions``), so the column never
    carries the rounding of ``c|right>``. A miss adds one Hamiltonian action
    to ``counts``. A hit adds none. Pair queries (``_pair_scalars``) and
    screening (``_screen_gradients``) share this column.
    """
    if right not in context.h_columns:
        method, rec = plan.method, plan.reconstruction
        inputs = plan._native["inputs"]
        if inputs.hamiltonian.reference.representation == "pauli":
            action = apply_terms(inputs.cache["action_tables"]["h0"], state, num_qubits=rec.num_qubits)
        else:
            # A matrix Hamiltonian acts through the one-column offset-free route.
            from .fixed_basis import _offset_free_actions

            column = np.asarray(state, dtype=np.complex128).reshape(-1, 1)
            hamiltonian = inputs.hamiltonian
            shifted = None
            if rec.identity_shift and hamiltonian.reference.representation in ("csr", "csc"):
                from .fixed_basis import _shifted_sparse

                key = (hamiltonian.reference.identity, rec.identity_shift.hex(),
                       hamiltonian.reference.representation, str(hamiltonian._data.dtype))
                if key not in context.shifted:
                    context.shifted[key] = _shifted_sparse(hamiltonian, rec.identity_shift,
                                                           max_bytes=method.max_bytes)
                shifted = context.shifted[key]
            action = _offset_free_actions(hamiltonian, column, rec.identity_shift,
                                          np.empty_like(column), max_bytes=method.max_bytes,
                                          max_products=method.max_products, shifted=shifted)[:, 0]
        context.h_columns[right] = action
        context.h_columns[right].flags.writeable = False
        counts["hamiltonian_actions"] += 1
    return context.h_columns[right]


def gradients_from_column(state, h0_state, generators, apply_generator):
    """Return ``2 Re<(H - cI)psi|A psi>`` for each generator, in order (see ``_screen_gradients``)."""
    return np.asarray([
        2.0 * np.vdot(h0_state, apply_generator(generator, state)).real
        for generator in generators
    ])


def _screen_gradients(plan, context, counts, right, state):
    """Screen the normalized product state using its cached (H-cI) action.

    For anti-Hermitian A, Re<psi|A|psi> is zero, so 2 Re<(H-cI)psi|A psi>
    equals <psi|[H,A]|psi>. The cache is keyed by the chain and the
    processed Hamiltonian. A missing column is computed once with the same
    offset-free action used for the pair matrix elements.

    For real theta, Hermitian H and anti-Hermitian A,
    ``d/dtheta <exp(theta A)psi|H exp(theta A)psi>`` at 0 is
    ``<psi|[H,A]|psi> = 2 Re<H psi|A psi>`` (Grimsley et al.,
    arXiv:1812.11173v2, Sec. II.B, step 5, and Anastasiou et al.,
    arXiv:2306.03227v3, Eq. (2)). Since ``<psi|A|psi>`` is purely imaginary,
    subtracting a real identity shift changes no derivative:
    ``2 Re<(H-cI)psi|A psi> = 2 Re<H psi|A psi>``. This holds even for an
    unnormalized psi. A Rayleigh-quotient derivative divides it by
    ``psi^dagger psi``, because an exact anti-Hermitian exponential preserves
    that norm, and ``_vector`` normalizes after every generator exponential,
    so screening uses the normalized-state convention. The column
    ``context.h_columns[right]`` holds ``(H-cI)psi_right`` when a pair query
    computed it. It can be absent for a new chain, the initial screen, a
    restored context or after ``_keep_current`` evicts states, and is then
    computed once (``_offset_free_column``). Changing from ``H psi`` to
    ``(H-cI) psi`` changes rounding and can change selection among nearly
    tied gradients. The anti-Hermitian assumption holds exactly for the
    checked generator coefficients and is not a tolerance that permits a
    Hermitian component. Units are the Hamiltonian's energy unit per radian.

    Selected generators are not screened again, following QuGCM
    ``adapt_gcim.py`` (c5efcb0), lines 303-398. With a fixed angle, reusing
    a generator repeats its single-generator basis state, so reuse would
    define another trial space. Each screened generator adds one generator
    action and one scalar product to ``counts``. A cache hit adds no
    Hamiltonian action.
    """
    h0_state = _offset_free_column(plan, context, counts, right, state)
    removed = {i for i, _ in right}
    indices = tuple(index for index in range(len(plan._native["inputs"].pool)) if index not in removed)
    tables = plan._native["inputs"].cache["action_tables"]["generators"]
    values = gradients_from_column(
        state, h0_state, (tables[i] for i in indices),
        lambda table, vector: apply_terms(table, vector, num_qubits=table.num_qubits),
    )
    counts["generator_actions"] += len(indices)
    counts["scalar_products"] += len(indices)
    return {f"g_{index}": float(value) for index, value in zip(indices, values, strict=True)}


def _pair_scalars(plan, context, counts, left, right, state):
    """Return ``S = <left|right>`` and ``H0 = <left|H - cI|right>`` as real and imaginary parts.

    ``c`` is the identity coefficient of a Pauli Hamiltonian, or the mean
    diagonal ``trace(H)/d`` of a dense or sparse matrix
    (``plan.reconstruction.identity_shift``), which the solve adds back to
    the Ritz values (``_pencil_from_observations``). ``(H - cI)|right>``
    comes from ``_offset_free_column``, cached in ``context.h_columns``
    because every left chain of one basis and the screen of that chain reuse
    it. A diagonal pair reports zero imaginary parts, because ``S_ii`` and
    ``H0_ii`` are real for a normalized state and a Hermitian ``H``.
    """
    left_state = _vector(left, plan, context, counts)
    column = _offset_free_column(plan, context, counts, right, state)
    s, h = np.vdot(left_state, state), np.vdot(left_state, column)
    if left == right:
        s, h = complex(s.real), complex(h.real)
    counts["scalar_products"] += 2
    return dict(
        s_real=float(s.real),
        s_imag=float(s.imag),
        h_real=float(h.real),
        h_imag=float(h.imag),
    )


def _residual_norm(plan, context, counts, right, values):
    """Return ``||(H - E) psi||`` for the normalized Ritz state of the basis of ``right``.

    Compute the residual norm by applying the processed Hamiltonian directly
    to the normalized Ritz combination in ``basis_chains`` order. Cached
    offset-free basis actions give an algebraically equal residual, but their
    accumulation has a different rounding error that can be amplified by
    cancellation in the Ritz coordinates.

    ``psi`` combines the ``basis_chains(right)`` states with the query's
    ``coefficient_*`` parameters, and ``E`` is its ``ritz_energy``
    parameter. The norm is in the Hamiltonian's energy unit.
    """
    method = plan.method
    basis = basis_chains(right)
    coefficients = [
        complex(
            number(values[f"coefficient_{i}_real"]),
            number(values[f"coefficient_{i}_imag"]),
        )
        for i in range(len(basis))
    ]
    ritz = _ritz_state(
        basis,
        coefficients,
        lambda: _vector((), plan, context, counts),
        lambda _, items: _vector(items, plan, context, counts),
    )
    residual = (
        plan._native["inputs"].hamiltonian.matvec(
            ritz, max_bytes=method.max_bytes, max_products=method.max_products
        )
        - float(number(values["ritz_energy"])) * ritz
    )
    counts["hamiltonian_actions"] += 1
    return {"residual_norm": stable_vector_norm(residual)}


def _classical_energy_gradient(plan, context, counts, right):
    """Return the energy ``<psi|H|psi>`` of chain ``right`` and its gradient in the chain's angles.

    Classical product optimization evaluates energy and gradient together
    with a forward and backward sweep, sharing one Hamiltonian action per
    parameter point (``optimization.normalized_taylor_energy_gradient``). The
    derivative follows the selected generator action and normalization: the
    forward factors are the scaled Taylor actions of
    ``adapt_actions.packed_exponential`` with the step counts of ``_taylor_steps``
    and the degree ``GENERATOR_TAYLOR_DEGREE``, each normalized by
    ``normalize_state_vector`` where ``_vector`` normalizes. The Hamiltonian
    acts as stored with no identity shift, so the energy is the same
    ``<psi|H|psi>`` that ``_classical_energy`` returns, and the gradient is
    that objective's. Returns ``energy`` and ``gradient_j`` for each angle.
    """
    from nwqlib.subroutines.fermionic_pool import GENERATOR_TAYLOR_DEGREE, _taylor_steps
    from .optimization import normalized_taylor_energy_gradient

    method = plan.method
    inputs = plan._native["inputs"]
    # Each generator acts with its packed table, selected by pool index.
    packed = {
        id(inputs.pool[index]): inputs.cache["action_tables"]["generators"][index] for index, _ in right
    }

    def action(generator, vector):
        counts["generator_actions"] += 1
        table = packed[id(generator)]
        return apply_terms(table, vector, num_qubits=table.num_qubits)

    def h_action(vector):
        counts["hamiltonian_actions"] += 1
        return inputs.hamiltonian.matvec(
            vector, max_bytes=method.max_bytes, max_products=method.max_products
        )

    energy, gradient = normalized_taylor_energy_gradient(
        _vector((), plan, context, counts),
        tuple(inputs.pool[index] for index, _ in right),
        tuple(theta for _, theta in right),
        action=action,
        step_count=lambda generator, theta: _taylor_steps(
            (value for _, value in generator.pauli_terms), theta
        ),
        normalize=normalize_state_vector,
        h0_action=h_action,
        degree=GENERATOR_TAYLOR_DEGREE,
    )
    counts["taylor_derivative_factors"] = len(right)
    counts["scalar_products"] += 1 + 2 * len(right)
    return {"energy": energy, **{f"gradient_{j}": float(value) for j, value in enumerate(gradient)}}


def _classical_energy(plan, state, counts):
    """Return ``{"energy": <psi|H|psi>}`` for the normalized query state."""
    method = plan.method
    h_state = plan._native["inputs"].hamiltonian.matvec(
        state, max_bytes=method.max_bytes, max_products=method.max_products
    )
    counts["hamiltonian_actions"] += 1
    counts["scalar_products"] += 1
    return {"energy": float(np.vdot(state, h_state).real)}


def bind_classical(plan, inputs, reference):
    """Bind the classical ADAPT routine to the Plan without computing anything.

    Each classical query becomes one call of the routine. ``binder``
    attaches a parameter point to the Run's state and action caches, and
    ``invoke`` evaluates it. ``screen`` returns the gradients
    ``<[H, A_i]>``, ``pair`` one ``(S_ij, H_ij)`` pencil entry, ``energy``
    one product-state energy and ``residual`` one Ritz residual norm. The
    returned application record counts the work performed, which is the
    reference constructions and generator exponentials on cache misses, the
    Hamiltonian and generator actions, and the scalar products.
    """
    from .adapt import HOST, cached_state_vectors

    template = plan.construction.kernels[0]

    def binder(realization, declaration, run):
        """Bind one parameter point to run-local state/action caches without executing it."""
        if run.plan is not plan:
            raise ValueError("ADAPT host cache belongs to another live Run")
        # drive_adapt saves this cache once per projected analysis and on return.
        context = run._method_context(AdaptContext, marks_changes=True)
        values = {b.parameter: b.value for b in realization.bindings}
        left, right = chain(values, "l"), chain(values, "r")
        name = realization.experiment

        def invoke():
            """Evaluate one query's scalars and record the work performed."""
            _keep_current(context, current_basis(context), left, right)
            counts = dict(
                reference_materializations=0,
                generator_exponentials=0,
                hamiltonian_actions=0,
                generator_actions=0,
                scalar_products=0,
            )
            # Peak live arrays: C_cache + 12 complex128 vectors of 2**q
            # amplitudes, 16 bytes each, with C_cache the vectors that the
            # state cache and the H column cache can hold after _keep_current
            # (adapt.cached_state_vectors): 8L for the default growth, with
            # L = max_selections, and 2(B + 2L) for a bounded basis of at
            # most B chains. The remaining 12 vectors are an allowance for
            # working arrays such as the query state, H|state>, a generator
            # action, the two Taylor buffers, the Ritz state and its residual,
            # and for a pair query on a dense matrix the state copy and one
            # row of A that removing the mean diagonal holds. The reference
            # preparation's bytes and the sparse blocks of that removal are
            # added.
            #
            # An energy-and-gradient evaluation of a chain of k generators
            # stores k + 1 forward states and k raw tangents, (2k + 1) 16d
            # bytes, and three scalar arrays of norms, step counts and the
            # gradient, 24k bytes. Inside taylor_action_derivative the old
            # base and tangent base, the two current Horner vectors, two
            # action outputs and up to three expression temporaries coexist,
            # at most 2k + 8 complex vectors at the last forward factor. The
            # twelve-vector term leaves room for normalizer scans and callback
            # buffers. With the C_cache <= 8L vectors the Run's caches can
            # still hold, it reserves 16 (C_cache + 2k + 12) d + 24k +
            # B_tables + B_H + max_A(B_A - 32d) + B_reference/setup. The input
            # and output of one Pauli action already occupy two of the vector
            # slots, so each action adds its shared allowance B_A less 32d:
            # the finite mask, action metadata and the largest
            # preprocessing/retry scratch (adapt_actions.action_sizes, over
            # all admitted tables). B_tables is the Plan's packed tables, B_H
            # the full Hamiltonian payload, and B_reference/setup the
            # reference preparation and offset-free setup bytes. A dense or
            # sparse H enters the maximum with its ordinary action envelope.
            rec = plan.reconstruction
            d = 1 << rec.num_qubits
            gradient = name == "energy" and int(number(values["insert_position"])) < 0
            k = len(right)
            cached = cached_state_vectors(plan.method, rec.max_selections)
            outer = 16 * (cached + 2 * k + 12) * d + 24 * k if gradient else 16 * (cached + 12) * d
            tables = inputs.cache["action_tables"]
            need = (outer + tables["resident_bytes"]
                    + inputs.hamiltonian.manifest.payload_bytes
                    + max((size - 32 * d for size in action_sizes(inputs, d)), default=0)
                    + rec.reference_action_bytes + rec.offset_free_action_bytes)
            _check_bytes(need, plan.method.max_bytes, "ADAPT classical frontier")
            if gradient:
                answer = _classical_energy_gradient(plan, context, counts, right)
                return output(answer, counts)
            state = _classical_query_state(plan, context, counts, right, values)
            if name == "screen":
                answer = _screen_gradients(plan, context, counts, right, state)
            elif name == "pair":
                answer = _pair_scalars(plan, context, counts, left, right, state)
            elif name == "residual":
                answer = _residual_norm(plan, context, counts, right, values)
            else:
                answer = _classical_energy(plan, state, counts)
            return output(answer, counts)

        def output(answer, counts):
            """Return the query's requested scalars and its work record."""
            scalars = tuple(
                ScalarValue(label=label, value=answer[label]) for label in declaration.scalars
            )
            return KernelOutput(
                plan_id=plan.content_id,
                selected_kernel_id=declaration.content_id,
                scalars=scalars,
                physical_scale=None,
                physical_scale_unavailable="no recovery applies: the scalars are already in the Problem's "
                "energy unit",
                applications=(
                    KernelApplication(
                        name=name,
                        implementation=HOST,
                        arguments=tuple(Binding(parameter=k, value=v) for k, v in counts.items()),
                    ),
                ),
            )

        return invoke

    return BoundKernel._bind_pointwise(plan, template, binder)


class _PartialData(Exception):
    """Data that a decision needs are missing or unusable. ``pending`` names them.

    Examples are an empty set of counts, a missing label or matrix
    entry, and a gradient norm that overflows binary64. The ADAPT iteration
    stops with ``partial_observation``, or ``uncertain_native_intent`` when
    the message names a submission whose state is uncertain, and analyzes the data it has
    instead of guessing the missing values.
    """

    def __init__(self, pending):
        self.pending = pending


class _AcquisitionPending(Exception):
    """External work for a query is not complete. ``status`` is its provider or preparation state.

    The ADAPT iteration saves a checkpoint and returns, and a later resume
    retrieves the same attempt.
    """

    def __init__(self, status):
        self.status = status


class _PreparedOnly(Exception):
    """Raised after the first new query is prepared when only preparation was requested."""


def _point(
    plan,
    experiment,
    left=(),
    right=(),
    *,
    term=-1,
    quadrature=0,
    group=0,
    insertion=None,
    coefficients=(),
    energy=0.0,
    stage=-1,
    shots=None,
):
    """Resolve one ADAPT query as a canonical point of the Plan's parameter space.

    ``left`` and ``right`` are chains ``((pool_index, theta), ...)``. ``term``
    and ``quadrature`` select a pair query's Pauli term (-1 for the overlap)
    and real (0) or imaginary (1) part, ``group`` a sampled readout group,
    and ``insertion`` an optional ``(position, term, sign)`` Pauli insertion.
    ``coefficients`` and ``energy`` carry a residual query's Ritz vector and
    value. Every parameter not supplied takes its canonical inactive value:
    -1 for pool, term and insertion indices, the experiment's register
    widths for ``control_width`` and ``readout_width``, and 0 otherwise. Equal
    queries therefore resolve to the same parameter point with one content
    hash and reuse its observation (``adapt._parameters``). A
    ``PencilSampling`` Plan also binds ``stage`` (-1 for selection, 0 pilot,
    1 main) and ``shots``, the Plan's shots unless given, so a pilot or main
    point never shares the identity of a selection query.
    """
    from nwqlib.core.records import Float64

    bindings = []
    for side, items in (("l", left), ("r", right)):
        bindings.append(Binding(parameter=f"{side}_count", value=len(items)))
        for i, (index, theta) in enumerate(items):
            bindings.extend(
                (
                    Binding(parameter=f"{side}_pool_{i}", value=index),
                    Binding(parameter=f"{side}_theta_{i}", value=Float64(value=theta)),
                )
            )
    bindings.extend(
        (
            Binding(parameter="term", value=term),
            Binding(parameter="quadrature", value=quadrature),
            Binding(parameter="group", value=group),
        )
    )
    if insertion is not None:
        bindings.extend(
            Binding(parameter=name, value=value)
            for name, value in zip(
                ("insert_position", "insert_term", "insert_sign"), insertion, strict=True
            )
        )
    if experiment == "residual":
        bindings.append(Binding(parameter="ritz_energy", value=Float64(value=energy)))
        for i, value in enumerate(coefficients):
            bindings.extend(
                (
                    Binding(
                        parameter=f"coefficient_{i}_real", value=Float64(value=float(value.real))
                    ),
                    Binding(
                        parameter=f"coefficient_{i}_imag", value=Float64(value=float(value.imag))
                    ),
                )
            )
    supplied = {item.parameter: item for item in bindings}
    defaults = {p.name: 0 for p in plan.construction.program.parameters}
    termwise_pair = experiment == "pair" and plan.reconstruction.acquisition.matrix_measurement != "grouped"
    defaults.update(
        control_width=int(experiment in ("pair", "diagonal")),
        readout_width=1 if termwise_pair else plan.reconstruction.num_qubits,
    )
    defaults.update(
        {
            p.name: -1
            for p in plan.construction.program.parameters
            if "_pool_" in p.name or p.name in {"term", "insert_position", "insert_term"}
        }
    )
    defaults.update(stage=stage, shots=plan.shots if shots is None else shots)
    bindings = tuple(
        supplied.get(p.name)
        or Binding(
            parameter=p.name,
            value=Float64(value=float(defaults[p.name]))
            if p.domain == "real"
            else defaults[p.name],
        )
        for p in plan.construction.program.parameters
    )
    return plan.resolve(experiment, bindings=bindings)


def _pair_reduction_values(chunk):
    """Return ``{"h0": H0, "overlap": S}`` from one weighted-pencil reduction chunk."""
    reduced = [v for v in chunk.values if v.kind == "reduced" and v.component == 0]
    if len(chunk.values) != 1 or len(reduced) != 1 or len(reduced[0].real) != 2:
        raise ValueError("exact pair query requires its two reduced complex scalars")
    (value,) = reduced
    return {"h0": complex(value.real[0], value.imaginary[0]),
            "overlap": complex(value.real[1], value.imaginary[1])}


def _read_values(plan, point, chunk):
    """Return ``{label: value}`` for one query from its own observation chunk.

    The chunk must belong to exactly this point and hold its unconditional
    counts. Classical chunks carry classical scalars and exact quantum
    chunks carry Pauli expectations, which are used as returned. For counts,
    the readout circuit has already rotated every label of the query's group
    to a product of ``Z`` on the label's support, so its estimate is the
    sample mean of ``(-1)**(parity of the measured bits on that support)``.
    A pair query reads the ancilla bit alone. The set of labels must match
    ``ADAPT.labels_at`` exactly.

    The counts are read once through ``ObservationChunk.histogram()``. Each
    label's support is a bit mask in the Histogram index layout (bit ``p`` is
    measured qubit ``p``, the rightmost label character, or bit ``p + 1``
    after the unmeasured phase bit of a grouped Plan), and its parity on
    every stored index is one ``np.bitwise_count`` over the index array, or
    over the ``(entries, words)`` packed array for a readout wider than 64
    bits. The estimate is the exact integer ``total - 2 * odd`` divided once
    by ``total``, where ``odd`` is the total count with odd parity. Each count
    is at most ``MAX_COUNT``, but a chunk's total is not, so ``total`` is a
    Python int. When it is at most ``MAX_COUNT``, every ``odd``, a sum of a
    subset of the nonnegative counts, fits in int64. Otherwise ``odd`` is
    summed as Python ints.
    """
    values = {b.parameter: b.value for b in point.bindings}
    for item in _chunks(chunk):
        if item.realization_id != point.content_id or item.bindings != point.bindings:
            raise ValueError("ADAPT observation differs from its exact query")
        if item.population != "unconditional":
            raise ValueError("ADAPT requires the selected unconditional observation population")
    if isinstance(chunk, tuple) or (point.experiment == "screen" and getattr(chunk, "point", None) is not None):
        # The exact shared full-chain query: its diagonal reduction supplies
        # h0 and the raw norm, and its screen point the active labels.
        result = {}
        for item in _chunks(chunk):
            if getattr(item, "point", None) == "diagonal":
                result.update(_pair_reduction_values(item))
            elif getattr(item, "point", None) == "screen":
                if any(not isinstance(v, PauliValue) for v in item.values):
                    raise ValueError("exact query requires native Pauli expectations")
                result.update({v.label: v.value for v in item.values})
            else:
                raise ValueError("ADAPT shared query has an undeclared point")
        labels = plan.method.labels_at(plan, point.experiment, values)
        missing = tuple(label for label in labels if label not in result)
        if missing:
            raise _PartialData(tuple(f"{point.experiment}: missing {label}" for label in missing))
        if any(label not in frozenset(labels) | {"h0", "overlap"} for label in result):
            raise ValueError("query returned an undeclared scalar label")
        return result
    if point.experiment == "pair" and plan.execution == "quantum" and plan.shots is None:
        # An exact pair query returns its weighted-pencil reduction
        # (H0_ij, S_ij) in the canonical chain orientation.
        return _pair_reduction_values(chunk)
    labels = plan.method.labels_at(plan, point.experiment, values)
    if plan.execution == "classical":
        if any(not isinstance(v, ScalarValue) for v in chunk.values):
            raise ValueError("THEORY query requires selected host scalars")
        result = {v.label: v.value for v in chunk.values if v.value is not None}
    elif plan.execution == "quantum" and plan.shots is None:
        if any(not isinstance(v, PauliValue) for v in chunk.values):
            raise ValueError("exact query requires native Pauli expectations")
        result = {v.label: v.value for v in chunk.values}
    else:
        from nwqlib.execution import MAX_COUNT

        # A grouped Plan puts an unmeasured phase bit before the system bits.
        offset = int(plan.reconstruction.acquisition.matrix_measurement == "grouped")
        width = 1 if point.experiment == "pair" else plan.reconstruction.num_qubits + offset
        histogram = chunk.histogram() if chunk.observation.kind == "counts" else None
        if histogram is None or histogram.entries and histogram.width != width:
            raise ValueError("counts must match the actual measured register")
        counts = histogram.weights
        total = sum(counts.tolist())
        if total != chunk.returned_shots:
            raise ValueError("count population differs from the observed returned shots")
        if not total:
            raise _PartialData((f"{point.experiment}: empty returned counts",))
        packed = histogram.packed_indices()
        if offset and np.any(packed[:, 0] & np.uint64(1)):
            raise ValueError("ADAPT query has a nonzero unmeasured phase bit")
        words = packed.shape[1]
        result = {}
        for label in labels:
            support = (
                1
                if point.experiment == "pair"
                else sum(1 << (i + offset) for i, axis in enumerate(reversed(label)) if axis != "I")
            )
            parity = np.zeros(len(counts), dtype=np.uint8)
            for word in range(words):
                mask = np.uint64((support >> (64 * word)) & 0xFFFFFFFFFFFFFFFF)
                if mask:
                    parity ^= np.bitwise_count(packed[:, word] & mask).astype(np.uint8)
            odd = counts[(parity & 1).astype(bool)]
            odd = int(odd.sum()) if total <= MAX_COUNT else sum(odd.tolist())
            result[label] = (total - 2 * odd) / total
    missing = tuple(label for label in labels if label not in result)
    if missing:
        raise _PartialData(tuple(f"{point.experiment}: missing {label}" for label in missing))
    declared = frozenset(labels)
    if any(label not in declared for label in result):
        raise ValueError("query returned an undeclared scalar label")
    return result


def _matrix_entries(plan, basis, available=(), full=None, *, stage=-1, counts=None):
    """Yield ``((i, j), element, key, point)`` for every upper-triangle pencil entry of the chains ``basis``.

    ``basis`` is the tuple of basis chains in pencil order. Each pair query
    is bound with its two chains in canonical (sorted) order,
    so the same unordered pair of states reuses one query and its observation
    across iterations even when their basis positions change. When the canonical
    order reverses (i, j), the matrix assembly conjugates the amplitude by flipping
    the sign of its imaginary part. Classical plans read all four H/S scalars
    from one query, exact quantum plans read ``(H0_ij, S_ij)`` from one
    weighted-pencil reduction per canonical chain pair (a diagonal pair only
    when the Hamiltonian has a nonidentity term), and sampled quantum plans
    have one query per ``MatrixElement``. In an exact quantum plan the
    diagonal of the chain ``full``, the full product of the selected
    generators, and of a chain that was the full chain in an earlier round
    comes from that chain's shared query on the full product state, which
    also serves its screening (``available`` holds the content hashes of the
    queries already observed). ``full`` matters only before that query is
    observed, so a pass over complete observations passes None.
    ``key`` is ``(canonical left, canonical right, term, quadrature)``, which
    determines the query within one Plan, and ``point()`` resolves it.

    A grouped sampled Plan instead yields ``((i, j), setting, key, point)``
    for each ``fixed_basis.sampled_settings`` entry
    ``(name, i, j, quadrature, group)`` of the current basis, with ``key``
    ``(canonical left, canonical right, quadrature index, group, stage)``.
    An off-diagonal setting is a ``pair`` query of the canonical chains, and
    a diagonal setting a ``diagonal`` query of one chain on both sides.
    Real-only quadratures omit every imaginary query. ``stage`` is -1 for
    the selection queries and, under ``PencilSampling``, 0 or 1 for the pilot
    or main stage of the final pencil, whose k-th setting binds
    ``counts[k]`` shots. ``point(shots)`` binds other shots. The stage in the
    key and in the point keeps the moments and observations of one stage from
    serving another.
    """
    rec = plan.reconstruction
    if rec.acquisition.matrix_measurement == "grouped":
        for k, (name, i, j, quad, g) in enumerate(sampled_settings(
                pencil_pairs(len(basis)), len(rec.energy_groups), rec.acquisition.quadratures)):
            left, right = basis[i], basis[j]
            canonical = (right, left) if right < left else (left, right)
            quadrature = int(quad == "imag")
            yield (i, j), (name, i, j, quad, g), (*canonical, quadrature, g, stage), (
                lambda shots=None if counts is None else counts[k], canonical=canonical, quadrature=quadrature,
                g=g, diagonal=i == j: _point(plan, "diagonal" if diagonal else "pair", *canonical,
                                             quadrature=quadrature, group=g, stage=stage, shots=shots))
        return
    if plan.execution == "classical" or plan.shots is None:
        # Classical host queries and exact pair reductions each return every
        # entry of their canonical chain pair at once.
        for i, left in enumerate(basis):
            for j in range(i if plan.execution == "classical" or plan.reconstruction.nonidentity_terms else i + 1,
                           len(basis)):
                canonical = (basis[j], left) if basis[j] < left else (left, basis[j])
                if plan.execution == "quantum" and i == j:
                    # A chain's diagonal comes from its shared full-chain
                    # query when that is the current full chain, whose screen
                    # point later screening reuses, or when an earlier round
                    # acquired it as its full chain.
                    shared = _point(plan, "screen", right=left)
                    if left == full or shared.content_id in available:
                        yield (i, j), None, (left, left, -2, 0), (lambda shared=shared: shared)
                        continue
                yield (i, j), None, (*canonical, -1, 0), (
                    lambda canonical=canonical: _point(plan, "pair", *canonical))
    else:
        terms = plan._native["inputs"].nonidentity if plan._native else tuple(
            (t.label, t.coefficient) for t in plan.reconstruction.nonidentity_terms)
        indices = {p: i for i, (p, _) in enumerate(terms)}
        for element in matrix_elements(len(basis), terms, plan.reconstruction.num_qubits,
                                       rec.acquisition.quadratures):
            i, j = element.basis_pair
            left, right = basis[i], basis[j]
            canonical = (right, left) if right < left else (left, right)
            term = -1 if element.matrix == "overlap" else indices[element.pauli_label]
            quadrature = int(element.quadrature == "imag")
            yield (i, j), element, (*canonical, term, quadrature), (
                lambda canonical=canonical, term=term, quadrature=quadrature: _point(
                    plan, "pair", *canonical, term=term, quadrature=quadrature))


def _new_pencil_queries(basis, earlier, reconstruction):
    """Return a lower bound on the quantum pencil queries of the chains ``basis`` that no earlier stage obtained.

    A pair query is bound to its two basis chains (``_matrix_entries``).
    ``earlier`` is the set of chains of the completed earlier bases, every
    one of which was analyzed, so all of their pair queries were obtained.
    The bound needs a set of current chains that contains both
    chains of every query the stage can reuse. When a matrix stage is first
    entered, before any of its own queries ran (``drive_adapt`` calls this
    only then), every reusable query belongs to a completed earlier basis,
    so the chains of the earlier bases form such a set. A resumed, partly
    acquired stage also reuses its own queries, whose chains can be new, so
    this shortcut does not apply to it, nor to a pass over a candidate basis
    outside the selected chain. With ``s`` such chains among the ``b``
    current ones, at least ``c(b) - c(s)`` queries are new, where ``c(b)`` is the
    complete b-state population of the Plan's resolved acquisition with L
    nonidentity Pauli terms and G Hamiltonian groups: ``pencil.pair_count``
    for exact pair reductions, ``fixed_basis.sampled_setting_count`` for
    grouped settings, ``b**2 G`` or ``b(b+1)G/2`` real-only, and
    ``pencil.matrix_element_count`` for termwise queries, ``b**2 L + b(b-1)``
    or ``bL + b(b-1)(L+1)/2`` real-only. For the legacy growth from two to
    three selections, b = 6 and s = 4, so ``(36 - 16)G = 20G`` grouped
    complex settings are new and ``(21 - 10)G = 11G`` real-only. The bound
    is exact whenever the earlier chains that the current basis holds all
    belong to the previous basis, whose pairs were all obtained. That holds
    without optimization, where each basis contains the one before it, and
    after an optimization unless an angle returned exactly to its value in
    an older basis, which only makes the bound smaller than the new count.
    It costs O(b) set lookups and forms no query.
    """
    terms = len(reconstruction.nonidentity_terms)
    acquisition = reconstruction.acquisition

    def count(size):
        if acquisition.matrix_measurement is None:
            return pair_count(size, terms)
        if acquisition.matrix_measurement == "grouped":
            return sampled_setting_count(size, len(reconstruction.energy_groups), acquisition.quadratures)
        return matrix_element_count(size * (size + 1) // 2, size, terms, acquisition.quadratures)

    return count(len(basis)) - count(sum(item in earlier for item in basis))


def _missing_queries(plan, basis, position, observations):
    """Return the number of distinct pencil queries of the chains ``basis`` with an entry in row ``position`` that ``observations`` lacks.

    Bounded growth admits its stages with this exact count of the queries
    still to run, because a bounded candidate or a chain added to the basis
    can reuse queries outside every completed earlier basis, where the
    shortcut of ``_new_pencil_queries`` does not apply. It resolves the
    points of the b entries of that row, and in exact quantum execution
    also the shared full-chain point of each diagonal that
    ``_matrix_entries`` checks.
    """
    return len({point().content_id for (i, j), _, _, point in _matrix_entries(plan, basis)
                if position in (i, j)} - observations.keys())


def _group_rows(plan):
    """Return the nonidentity ``(label, coefficient)`` rows in ``energy_labels`` order, which ``energy_groups`` index."""
    rec = plan.reconstruction
    coefficients = {t.label: t.coefficient for t in rec.nonidentity_terms}
    return tuple((label, coefficients[label])
                 for label in energy_labels((t.label for t in rec.nonidentity_terms), label_cache(plan)))


def _reduce_query(plan, point, chunk, rows):
    """Return the ``GroupMoment`` of one grouped pencil query in its canonical chain order.

    The chunk must belong to exactly this point and hold unconditional
    counts. ``fixed_basis.reduce_group`` reads them in its shared layout,
    the same reduction FixedGCIM uses, with ``rows`` the Plan's group rows
    (``_group_rows``), which a pass builds once. The moment's pair is
    ``(0, 1)`` for an off-diagonal query and ``(0, 0)`` for a diagonal one
    until ``_oriented_moment`` places it in a basis. Empty counts are
    missing data.
    """
    rec = plan.reconstruction
    if chunk.realization_id != point.content_id or chunk.bindings != point.bindings:
        raise ValueError("ADAPT observation differs from its exact query")
    if chunk.population != "unconditional":
        raise ValueError("ADAPT requires the selected unconditional observation population")
    values = {b.parameter: b.value for b in point.bindings}
    group = int(number(values["group"]))
    diagonal = point.experiment == "diagonal"
    setting = (point.experiment, 0, 0 if diagonal else 1, "imag" if number(values["quadrature"]) else "real", group)
    moment = reduce_group(chunk, setting, rows=rows,
                          members=rec.energy_groups[group] if rec.energy_groups else (),
                          num_qubits=rec.num_qubits)
    if moment is None:
        raise _PartialData((f"{point.experiment}: empty returned counts",))
    return moment


def _oriented_moment(moment, pair, conjugate):
    """Place a canonical-order group moment at basis positions ``pair``, conjugated when the basis reverses the chains.

    Reversing a pair conjugates its entry, ``<j|P|i> = conj(<i|P|j>)``, so
    the imaginary setting of the reversed pair has the per-shot values
    ``(-X, -Y)`` of the measured one, ancilla sign and weighted parity
    together. Its first moments (``label_means``, ``mean``,
    ``overlap_mean``) change sign, while ``second``, ``overlap_cross``, the
    variances and the H0-overlap covariance are invariant under
    ``(X, Y) -> (-X, -Y)``. A real setting is unchanged. An unavailable
    field stays None.
    """
    fields = {name: getattr(moment, name) for name in type(moment).model_fields if name != "parent_id"}
    fields["pair"] = pair
    if conjugate:
        fields.update(label_means=tuple(-value for value in moment.label_means), mean=-moment.mean,
                      overlap_mean=None if moment.overlap_mean is None else -moment.overlap_mean)
    return GroupMoment(**fields)


def _grouped_moments(plan, basis, context, acquire=None, *, stage=-1, counts=None):
    """Return the settings of the grouped pencil of the chains ``basis`` and their moments at basis positions.

    Each setting of the current basis (``_matrix_entries`` at ``stage`` with
    ``counts``) is one query of its canonical chains, quadrature, group and
    stage, reused from ``context.pair_values`` when an earlier pass read it.
    The pass orients every current setting for the pooling that follows
    (``_grouped_statistics`` or ``pencil_workflow.pilot_decision``), so the
    whole current population, the chunk of each new query and the chunk
    behind each cached moment, is admitted by
    ``fixed_basis.grouped_pass_requirements`` (bytes against ``max_bytes``,
    work against ``max_products``) with its settings, pairs and oriented
    view, which charge that pooling too. With ``acquire`` the pass is
    admitted before the first missing query runs, each table not yet
    acquired charged its largest possible bin count ``min(N, 2**(n+1))`` for
    the shots N that its point binds, so a refusal leaves no new
    acquisition. Once every missing query has its chunk, read from the
    observations or run by ``acquire``, the pass is admitted again with the
    stored bins of every chunk before any count is decoded or any current
    view is formed. Only the new chunks are decoded, and their canonical
    moments enter ``context.pair_values`` with the stored bin count of their
    chunk. The moments are then placed at their basis positions, conjugated
    where the basis reverses the canonical order. Returns
    ``(settings, moments)`` in setting order.
    """
    rec = plan.reconstruction
    entries = tuple(_matrix_entries(plan, basis, stage=stage, counts=counts))
    groups = rec.energy_groups or ((),)
    pending = {}
    for _, setting, key, point in entries:
        if key not in context.pair_values and key not in pending:
            pending[key] = setting, point()

    # Every current setting enters the pass, cache hits included. The
    # allowance covers this pass only, not other passes or the Run's history
    # (fixed_basis.grouped_pass_requirements).
    def admit(bins):
        """Check the pass of every current setting, with ``bins(key)`` the stored bins of its chunk."""
        size, work = grouped_pass_requirements(
            rec.num_qubits, ((len(groups[key[3]]), bins(key)) for _, _, key, _ in entries),
            len(rec.nonidentity_terms), len(basis), settings=len(entries),
            pairs=len({pair for pair, _, _, _ in entries}), oriented=True)
        _check_bytes(size, plan.method.max_bytes, "ADAPT grouped count arrays and scalar payload")
        _check_products(work, plan.method.max_products)

    if acquire is not None:
        # The law increases with every bin count, so the largest one bounds a
        # table not yet acquired: its point's bound shots or the 2**(n+1)
        # outcomes, never the shots of another stage.
        largest = {key: min(point.resolved_observation(plan)[1].shots, 1 << (rec.num_qubits + 1))
                   for key, (_, point) in pending.items()}
        held = {key: context.observations.get(point.content_id) for key, (_, point) in pending.items()}
        admit(lambda key: context.pair_values[key][2] if key not in pending else
              largest[key] if held[key] is None else len(_single_chunk(held[key]).values))
    new = {}
    for key, (setting, point) in pending.items():
        chunk = context.observations.get(point.content_id) if acquire is None else acquire(point)
        if chunk is None:
            raise _PartialData((f"matrix {setting[0]}",))
        new[key] = point, _single_chunk(chunk)
    admit(lambda key: len(new[key][1].values) if key in new else context.pair_values[key][2])
    rows = _group_rows(plan)
    for key, (point, chunk) in new.items():
        context.pair_values[key] = (_reduce_query(plan, point, chunk, rows), chunk.prepared_id, len(chunk.values))
    moments = tuple(
        _oriented_moment(context.pair_values[key][0], (i, j), setting[3] == "imag" and basis[j] < basis[i])
        for (i, j), setting, key, _ in entries)
    return tuple(e[1] for e in entries), moments


def _grouped_statistics(plan, basis, context, acquire=None, *, stage=-1, counts=None):
    """Return the group moments, pooled pair entries and pair variances of the grouped pencil of the chains ``basis``.

    The oriented moments of ``_grouped_moments`` are pooled by
    ``fixed_basis.pooled_statistics``, the same pair reconstruction as
    FixedGCIM, with real-only imaginary parts as exact zeros. Returns
    ``(moments, pairs, variances)``.
    """
    settings, moments = _grouped_moments(plan, basis, context, acquire, stage=stage, counts=counts)
    _, _, pairs, variances = pooled_statistics(settings, moments, plan.reconstruction.identity_shift)
    return moments, pairs, variances


def _matrix_from_observations(plan, basis, context, *, acquire=None, receipt=None, full=None):
    """Return ``(H0, S, sampled, allowance)`` of the basis chains ``basis`` from their pair queries.

    ``H0`` and ``S`` are complex ``b``-by-``b`` arrays in basis-chain order,
    completed below the diagonal by conjugation. ``H0`` omits the identity
    term ``cI`` of the Hamiltonian, ``plan.reconstruction.identity_shift``.
    ``sampled`` is True for every quantum Plan with shots, including one
    whose scalar Hamiltonian leaves no entry to count. ``allowance``
    bounds the error of a deterministic S that the solve's negative-mode
    test allows. It is the Gram formation error of classical inner products
    of length 2**n, or for exact pair reductions the largest row sum of the
    off-diagonal overlap entry bounds from each pair's preparation record
    (``receipt(prepared_id)``, ``fixed_basis._exact_overlap_allowance``),
    zero when a record or its state error is unavailable. With
    ``acquire`` each missing query is run, otherwise only already
    collected observations are used and a missing one raises
    ``_PartialData``. Each observation is validated by ``_read_values``, or
    for a grouped setting reduced by ``_grouped_statistics``, before it is
    used, and its values are kept in ``context.pair_values``, so a later
    stage reads only its new queries. A query bound in swapped canonical
    order measured ``<j|P|i> = conj(<i|P|j>)``, so it is conjugated
    (``_matrix_entries``), once per assembly. ``full`` is the chain whose
    exact diagonal comes from its shared screening query
    (``_matrix_entries``).
    """
    b = len(basis)
    _check_bytes(_projected_solve_bytes(b), plan.method.max_bytes, "ADAPT projected analysis")
    _check_products(_projected_solve_work(b), plan.method.max_products)
    values, sampled = {}, False

    def read(key, point, missing):
        """Return ``(values, prepared_id)`` of one pencil query, reading its chunk once per context."""
        if key not in context.pair_values:
            point = point()
            chunk = context.observations.get(point.content_id) if acquire is None else acquire(point)
            if chunk is None:
                raise _PartialData((missing,))
            context.pair_values[key] = (_read_values(plan, point, chunk), _chunks(chunk)[0].prepared_id)
        return context.pair_values[key]

    if plan.execution == "classical":
        h, s = np.empty((b, b), dtype=complex), np.empty((b, b), dtype=complex)
        for (i, j), _, key, point in _matrix_entries(plan, basis):
            v = read(key, point, f"matrix pair {i},{j}")[0]
            sign = -1 if basis[j] < basis[i] else 1
            h[i, j], s[i, j] = (
                complex(v["h_real"], sign * v["h_imag"]),
                complex(v["s_real"], sign * v["s_imag"]),
            )
            h[j, i], s[j, i] = h[i, j].conjugate(), s[i, j].conjugate()
        allowance = gram_formation_allowance(
            2**plan.reconstruction.num_qubits, fsum(float(s[i, i].real) for i in range(b))
        )
    elif plan.shots is None:
        # Each exact pair reduction returns (H0, S) in its canonical chain
        # orientation. It is conjugated once when basis[j] < basis[i], applied
        # to both H0 and S. The reduction already returns complex entries, so
        # no per-quadrature sign adjustment applies. Its entry bounds come
        # from its own receipt's state error with the ordered-action envelope.
        from .fixed_basis import _coefficient_mass
        from .pair_reducer import entry_bounds

        rec = plan.reconstruction
        terms = len(rec.nonidentity_terms)
        c1 = _coefficient_mass([t.coefficient for t in rec.nonidentity_terms])
        pairs, bounds = {}, {}
        for (i, j), _, key, point in _matrix_entries(plan, basis, context.observations, full):
            v, prepared_id = read(key, point, f"matrix pair {i},{j}")
            h0, overlap = v["h0"], v["overlap"]
            if basis[j] < basis[i]:
                h0, overlap = h0.conjugate(), overlap.conjugate()
            pairs[(i, j)] = (h0, overlap)
            prepared = None if receipt is None else receipt(prepared_id)
            # The input array can carry a shared host phase correction. The
            # pair entries cancel the common global phase (saved-state budget).
            delta = None if prepared is None else prepared.saved_state_error(("amplitude-derived masses",))[0]
            entry = entry_bounds(delta, c1, 1 << rec.num_qubits, terms,
                                 diagonal=basis[i] == basis[j])
            bounds[(i, j)] = None if entry is None else entry[0]
        h, s = assemble_pair_pencil(b, pairs)
        allowance = _exact_overlap_allowance(b, bounds)
        # An unavailable allowance admits no input error beyond the solver's
        # own roundoff.
        allowance = 0.0 if allowance is None else allowance
    elif plan.reconstruction.acquisition.matrix_measurement == "grouped":
        # The shared grouped reduction, pooling and assembly of FixedGCIM.
        h, s, sampled, allowance = _grouped_matrices(b, _grouped_statistics(plan, basis, context, acquire)[1])
    else:
        # Termwise settings with shots, or a Plan with shots whose scalar
        # Hamiltonian resolved no measurement. Both solve with the sampled
        # policy that ADAPTResult requires of a Plan with shots, also when no
        # entry is read.
        sampled = True
        for (i, j), element, key, point in _matrix_entries(plan, basis):
            read_values = read(key, point, f"matrix {element.name}")[0]
            sign = -1 if element.quadrature == "imag" and basis[j] < basis[i] else 1
            values[element.name] = sign * next(iter(read_values.values()))
        h, s = assemble_pencil(
            b,
            plan._native["inputs"].nonidentity if plan._native else tuple(
                (t.label, t.coefficient) for t in plan.reconstruction.nonidentity_terms),
            plan.reconstruction.num_qubits,
            values,
            plan.reconstruction.acquisition.quadratures,
        )
        # A sampled overlap uses the positive-subspace policy, which admits no
        # deterministic input allowance.
        allowance = 0.0
    return h, s, sampled, allowance


def _grouped_matrices(b, pairs):
    """Return ``(H0, S, sampled, allowance)`` of a b-state basis from its pooled grouped pair entries.

    A sampled overlap uses the positive-subspace policy, which admits no
    deterministic input allowance.
    """
    h, s = assemble_pair_pencil(b, pairs)
    return h, s, True, 0.0


def _overlap_settings(moments):
    """Return ``{(i, j): settings}`` of the overlap of every off-diagonal pair from the oriented moments of ``_grouped_statistics``.

    Group zero supplies the overlap. Each setting is ``(contribution_id,
    shots, overlap_variance)`` of its one moment, because ADAPT measures each
    setting once, the real part first and the imaginary part second when
    the Plan measures it (``bounded_growth.sampled_metric``). Orientation
    leaves the variance unchanged (``_oriented_moment``). The moments are
    read, never changed.
    """
    parts = {(m.pair, m.quadrature): (m.contribution_id, m.shots, m.overlap_variance)
             for m in moments if m.group == 0 and m.pair[0] != m.pair[1]}
    return {pair: (setting, *((parts[pair, "imag"],) if (pair, "imag") in parts else ()))
            for (pair, quadrature), setting in parts.items() if quadrature == "real"}


def _pencil_from_observations(plan, basis, context, *, cutoff=None, receipt=None):
    """Solve the pencil of the basis chains ``basis`` with the shared ``fixed_basis.solve_pencil`` and return the ``ProjectedPencil``.

    The matrices already assembled for ``basis`` (``context.matrix_data``,
    by the live matrix stage or by the grouped pass of ``analyze_adapt``)
    are used once and released. Otherwise, as after an interruption or for
    a new ``cutoff`` of a termwise or exact pencil, they are rebuilt from
    the saved observations, which hold every query of the completed stage,
    with ``receipt`` as in ``_matrix_from_observations``. The solve uses
    ``H0`` with the Gram allowance of ``_matrix_from_observations`` and the
    method's ``symmetrize_matrices`` choice, adds the identity coefficient c
    to the Ritz values, and records ``H = H0 + c S`` and the lowest one or
    two Ritz values before c is added (``offset_free_eigenvalues``). With
    shots the pencil also carries the sampled comparison with the processed
    Pauli enclosure (``sampled_*`` fields), which is absent when an endpoint
    of that enclosure is not finite and for deterministic data. A nonfinite
    enclosure violation is recorded as None.
    """
    cutoff = plan.method.overlap_cutoff if cutoff is None else cutoff
    current = context.matrix_data
    context.matrix_data = None
    h, s, sampled, allowance = (current[1] if current is not None and current[0] == basis
                                else _matrix_from_observations(plan, basis, context,
                                                               receipt=receipt))
    return solve_pencil(h, s, cutoff=cutoff, sampled=sampled,
                        enclosure=processed_pauli_enclosure(plan.reconstruction) if sampled else None,
                        identity_shift=plan.reconstruction.identity_shift,
                        overlap_input_tolerance=allowance,
                        symmetrize_matrices=plan.method.symmetrize_matrices)


# The two values of ``pencil_workflow.new_state`` that the final stages fix
# once, by their names there, and the top-level checkpoint fields that hold
# them apart from the stage cursor ``final``.
_FINAL_FIXED = {"pilot_counts": "pilot_counts", "decision": "shot_allocation"}


def _new_controller_state(controls, *, bounded=False):
    """Return the initial checkpoint state of a new ADAPT iteration.

    The state is plain JSON so a Run checkpoint can store it. Its keys:

    - ``kind``, ``controls``: format tag and the Plan identity and optimizer
      resume policy. Resume requires both to be identical.
    - ``selected``, ``theta``: current generator chain, as pool indices and
      angles in radians. Empty under ``bounded`` growth.
    - ``basis``, ``value_basis``, ``cursor``: only under ``bounded`` growth,
      the chains of the basis, the reference ``[]`` first, the chains of the
      latest projected analysis, and ``[parent position, next pool index]``
      of each parent of the last examination in cyclic position order
      ending with the parent of the last examined extension, the anchor
      after which the next examination starts (``bounded_growth.frontier``).
      The order is part of the state, so nothing may sort it.
    - ``phase``: next scientific action, one of ``matrix``, ``analysis``,
      ``residual``, ``decision``, ``screen``, ``optimize`` or ``complete``,
      under ``bounded`` growth ``select`` (examine the extensions of one
      round) and ``rows`` (obtain and score the admitted ones), and under
      ``PencilSampling`` ``final`` (the pilot and main stages of the value
      basis and the analysis of its main stage) once growth has stopped.
    - ``final``: only from the ``final`` phase on, the stop reason that ended
      growth (``stop_reason``) and, once the stages start, their cursor: the
      fields of ``pencil_workflow.new_state`` other than the two fixed
      values below. Its ``pending``, ``analysis_in_progress`` and
      ``analysis_attempts`` belong to the stages and are separate from the
      top-level fields of the same names. Its ``analysis_attempts`` counts
      the started pilot allocations. Its ``analysis_in_progress`` stays
      False, because the analysis of the main stage uses the top-level
      ``analysis_in_progress``.
    - ``pilot_counts``, ``shot_allocation``: once the stages start, the
      pilot shots of each setting and the serialized ``ShotAllocation``,
      None until the decision. ``new_state`` returns them as
      ``pilot_counts`` and ``decision`` (``_FINAL_FIXED``). Each is a field
      of its own, so a checkpoint of the cursor does not write it again.
    - ``pending``, ``pending_labels``: records of queries whose external work
      is not complete, and the labels reported as ``ADAPTResult.pending``.
    - ``history``: one dictionary per projected analysis, the future
      ``AdaptRound`` fields plus ``new_start``/``new_stop`` indices into the
      contribution order.
    - ``value``, ``value_selected``, ``value_theta``, ``coefficients``: the
      latest projected Ritz value, the chain of its basis and its
      coefficient vector as ``[real, imag]`` pairs. ``project_current`` adds
      ``pencil``, the serialized ``ProjectedPencil``.
    - ``previous_energy``, ``flat_count``: flat-counter state
      (``_flat_counter_update``).
    - ``optimizer``, ``optimizer_attempts``, ``optimizer_rounds``,
      ``optimizer_restarts``, ``restart_pending``, ``optimizer_exhausted``:
      the active BFGS invocation, cumulative counters, restarts after resume
      and whether the evaluation limit ended the last optimization.
    - ``energy_queries``, ``active_energy``: logical energy queries counted
      against ``optimize_max_evaluations`` and the query in progress that
      was already counted.
    - ``analysis_attempts``, ``current_analysis_attempt``,
      ``analysis_in_progress``: projected-analysis bookkeeping for
      ``recover_adapt_analysis``, for the analyses of the growth and the
      analysis of the main stage. ``ADAPTResult.analysis_attempts`` adds the
      pilot allocations that ``final`` counts.
    - ``stop_reason``: outcome of the iteration, ``running`` until a stop.
    - ``stage_start``: first contribution index of the next history record.
    """
    extra = dict(basis=[[]], value_basis=[], cursor=[]) if bounded else {}
    return dict(
        **extra,
        kind="adapt_gcim/3",
        controls=controls,
        selected=[],
        theta=[],
        phase="matrix",
        pending=[],
        pending_labels=[],
        history=[],
        value=None,
        value_selected=[],
        value_theta=[],
        coefficients=[],
        previous_energy=None,
        flat_count=0,
        optimizer=None,
        optimizer_attempts=0,
        optimizer_rounds=0,
        optimizer_restarts=[],
        restart_pending=None,
        energy_queries=0,
        active_energy=None,
        optimizer_exhausted=False,
        analysis_attempts=0,
        current_analysis_attempt=None,
        analysis_in_progress=False,
        stop_reason="running",
        stage_start=0,
    )


def _flat_counter_update(flat_count, previous_energy, value, iteration, *, valid, tolerance):
    """Return ``(flat_count, previous_energy)`` after one projected analysis.

    Zheng et al. (2024), arXiv:2312.07691v3, METHODS, p. 9, stop after
    ``T`` consecutive iterations whose change in the lowest eigenvalue is
    below a tolerance (``_stopping_reason``). The count starts with the
    change from the reference-only basis (iteration 0) to the first
    two-state basis, so iteration 0 only records its energy. A change at or
    above ``tolerance`` resets the count. An unavailable energy (no value or
    a solver rejection) also resets the count and forgets the previous energy,
    because it interrupts the sequence of small changes.
    """
    if not valid:
        return 0, None
    change = None if previous_energy is None or value is None else value - previous_energy
    if iteration:
        flat_count = flat_count + 1 if change is not None and abs(change) < tolerance else 0
    return flat_count, value


def _stopping_reason(state, options, reconstruction, count):
    """Return the stop reason of the ADAPT iteration after an analysis with ``count`` selections, or None.

    The rules are checked in this order: the optimizer's evaluation limit
    was reached, the residual norm met ``residual_norm_tolerance``
    (``residual_norm`` stopping only), every pool member is selected, the
    iteration limit is reached, or the flat count is at least
    ``T = min(t_user, t_auto_fraction * unselected)`` (Zheng et al. (2024),
    arXiv:2312.07691v3, METHODS, p. 9, with ``T_auto`` 20% of the
    unselected operators). ``T`` is recomputed at every analysis from the
    current number of unselected generators, and for the integer count the
    rule is ``flat_count >= ceil(T)``. The reference-only analysis (``count == 0``) never stops on
    the optimizer, residual or flat rules. None means screen the pool for
    the next generator.
    """
    if count and state["optimizer_exhausted"]:
        return "optimization_evaluation_budget_exhausted"
    if (
        count
        and options.stop_criterion == "residual_norm"
        and state["history"][-1]["residual_norm"] <= options.residual_norm_tolerance
    ):
        return "residual_norm_threshold"
    if count == len(reconstruction.pool):
        return "pool_exhausted"
    if count == reconstruction.max_selections:
        return "max_iterations"
    if (
        options.stop_criterion == "flat_counter"
        and count > 0
        and state["flat_count"]
        >= min(
            options.t_user,
            options.t_auto_fraction * (len(reconstruction.pool) - count),
        )
    ):
        return "flat_counter"
    return None


def _bounded_stopping_reason(state, options, reconstruction, size):
    """Return the stop reason of bounded growth after an analysis of a basis of ``size`` chains, or None.

    The rules are checked in this order: the basis holds ``max_basis_size``
    chains, ``max_iterations`` rounds have each added a chain, or the flat
    count is at least ``T = min(t_user, t_auto_fraction * P)``, the rule of
    ``_stopping_reason`` with the pool size P as a fixed scale in place of
    the number of unselected generators. P is not a count of remaining
    extensions. The reference-only analysis never stops on the flat rule. These
    rules limit the work and say nothing about convergence. None means
    examine the extensions of the next round.
    """
    rounds = size - 1
    if size >= options.max_basis_size:
        return "max_basis_size"
    if rounds >= options.max_iterations:
        return "max_iterations"
    if (options.stop_criterion == "flat_counter" and rounds > 0
            and state["flat_count"] >= min(options.t_user, options.t_auto_fraction * len(reconstruction.pool))):
        return "flat_counter"
    return None


def _round_stop_reason(labels, exhausted):
    """Return the stop reason of a bounded round that admitted no extension.

    ``basis_depth_limit`` when every examined extension that does not
    repeat a basis chain was excluded by the depth limit and there was at
    least one, ``parent_window_exhausted`` when no parent had a generator
    left, and ``candidate_scans_exhausted`` when ``max_candidate_scans``
    ended the examination first.
    """
    reasons = {label["reason"] for label in labels} - {"duplicate"}
    if reasons and reasons <= {"depth_unknown", "above_depth_limit"}:
        return "basis_depth_limit"
    return "parent_window_exhausted" if exhausted else "candidate_scans_exhausted"


# Stop reasons for unresolved external work, keyed by the provider or
# preparation status with any ``preparation_`` prefix removed. Any other
# ``preparation_*`` status is ``pending_preparation`` and any other status is
# ``pending_acquisition``.
_PENDING_STOP_REASONS = {
    "intent": "uncertain_native_intent",
    "uncertain": "uncertain_native_intent",
    "upload_uncertain": "uncertain_preparation",
    "compile_uncertain": "uncertain_preparation",
    "failed": "backend_failed",
    "cancelled": "cancelled",
}


def _pending_stop_reason(status):
    """Return the ``stop_reason`` recorded when a query's external work is unresolved."""
    return _PENDING_STOP_REASONS.get(
        status.removeprefix("preparation_"),
        "pending_preparation" if status.startswith("preparation_") else "pending_acquisition",
    )


def _single_chunk(completed):
    """Return a query's observation: its one chunk, or the tuple of point chunks of a two-point query.

    A pair reduction has one point and returns its chunk. The exact shared
    query on the full product state has two points at one boundary, its
    weighted diagonal reduction and its screening labels, and keeps both
    chunks in schedule order. They share one data collection and preparation
    record.
    """
    if isinstance(completed, tuple) and len(completed) == 1:
        (completed,) = completed
    return completed


def _chunks(observation):
    """Return the chunks of one query's observation as a tuple."""
    return observation if isinstance(observation, tuple) else (observation,)


def _observation_index(chunks):
    """Index observations by resolved query, keeping the point chunks of one data collection together.

    The first data collection of a query is kept, as for single
    chunks. A later point chunk of that same collection joins it in
    collection order, which is its schedule order.
    """
    index = {}
    for chunk in chunks:
        held = index.get(chunk.realization_id)
        if held is None:
            index[chunk.realization_id] = chunk
            continue
        held = _chunks(held)
        if chunk.point is not None and chunk.attempt == held[0].attempt and all(
                c.point != chunk.point for c in held):
            index[chunk.realization_id] = (*held, chunk)
    return index


def drive_adapt(plan, *, run, prepare_only=False):
    """Run the ADAPT iteration until it finishes, must wait for external work or, with ``prepare_only``, has prepared its first new query.

    The iteration is a checkpointed state machine. Its phases are
    ``matrix`` (obtain every pencil entry of the current basis),
    ``analysis`` (solve the pencil and record one ``AdaptRound``),
    ``residual`` (classical residual stopping only), ``decision`` (apply the
    stopping rules), ``screen`` (measure ``<[H, A_i]>`` for unselected
    generators and append the largest absolute gradient, with equal binary64
    absolute values going to the lowest pool index) and ``optimize`` (optional BFGS every ``optimize_every_m``
    selections). The science follows ADAPT-GCiM, as described in the
    ``ADAPT`` docstring. Under ``BoundedRitzGrowth`` the phases ``select``
    (examine one round's extensions, ``bounded_growth.frontier``) and
    ``rows`` (obtain each admitted extension's pencil row and its score,
    with shots only when its sampled eligibility passes, then append the
    best extension to the explicit basis) take the
    place of ``screen`` and ``optimize``, and the same ``matrix``,
    ``analysis`` and ``decision`` phases follow. The extension that is
    added finds all its entries already obtained, so its matrix stage runs
    no query. Under ``ADAPT.sampling`` the stop that ends growth leads to the
    ``final`` phase instead of ``complete``: once, on the value basis, the
    pilot and main stages of ``pencil_workflow.advance_stages`` with the
    allocation decided between them by ``pencil_workflow.pilot_decision``,
    and then the analysis of the main-stage measurements alone, after
    which the run ends with the stop reason of its growth. Before the first
    pilot preparation the Run must have room for 2m preparations, 2m
    circuits and the whole ``total_shots`` for the m settings of that basis.

    Invariants that make interruption safe:

    - Each phase checkpoint names the next scientific action. The saved state
      holds the trajectory, the latest pencil and the pending queries, while
      raw observations stay in the Run.
    - A new query records its seed and checkpoint sequence before any external
      work. Resume retrieves that original attempt or preparation and never
      draws a second set of shots for the same point.
    - Unresolved external work stops the iteration with a pending or
      uncertain reason instead of retrying, so no preparation, submission or
      shot is duplicated.
    - A circuit that ``ADAPT.max_native_two_qubit_depth`` refuses before its
      submission (``NativeDepthCapRefusal`` from ``ADAPT.before_submit``)
      for its depth, above the limit or unqualified, ends the iteration with
      ``native_depth_refused``. The saved complete phase returns the Result
      of the last solved pencil on resume, without preparing or submitting
      that query again. A refusal for installed package versions other than
      the recorded ones (``provenance``) is raised with the checkpoint
      unchanged, so the same Run continues where the recorded versions are
      installed.
    - A logical energy query is counted against ``optimize_max_evaluations``
      before its circuits run, and an identical point reuses collected
      values without being counted again.
    - SciPy BFGS has no resumable internal state. After resume it restarts
      from the best completed point and records the restart.
    - An interrupted projected analysis, the analysis of the main stage
      included, or an interrupted allocation is not repeated
      automatically. It needs ``run.resume(reanalyze=True)``
      (``recover_adapt_analysis``).
    - The final stages checkpoint each pending point with its seed before
      its preparation and the allocation before the first main
      preparation, so a resume repeats no selection, pilot or allocation.

    Args:
        plan: The ADAPT Plan of ``run``.
        run: The Run whose run log, RNG and observations this iteration uses.
        prepare_only: Stop after preparing the first new query, without submission.

    Returns:
        The analyzed ``ADAPTResult``, or None while external work is pending or
        when only preparation was requested.
    """
    from nwqlib.backends.inspection import NativeDepthCapRefusal
    from .optimization import _EvaluationBudgetExhausted, optimize_product

    if run.plan is not plan:
        raise ValueError("ADAPT execution requires the same selected Plan and Run")
    options, reconstruction = plan.method, plan.reconstruction
    growth = options.basis_growth
    from .adapt import independence_floor
    from .bounded_growth import (
        candidate_score, current_projection, frontier, generator_cx_weight, reference_cx_bound, sampled_eligibility,
        sampled_metric)

    controls = dict(
        plan_id=plan.content_id, optimizer_resume_policy="restart_if_no_official_resume"
    )
    context = run._method_context(AdaptContext, marks_changes=True)
    from .adapt import CompilerPlans

    compilers = plan._native.get("compiler_plans")
    if isinstance(compilers, CompilerPlans) and context.compiler_plans is not compilers:
        # A reopened Run's saved compiler plans join the reopened Plan's map,
        # so its selected generators are not compiled again, and later
        # checkpoints save every plan this map builds.
        if isinstance(context.compiler_plans, tuple):
            compilers.restore(context.compiler_plans)
        context.compiler_plans = compilers
    state = run.checkpoint_state
    restored = state is not None
    # Checkpoint the scientific phase and query frontier together, so reopening
    # can distinguish unfinished work from a new adaptive decision.
    if state is None:
        if run.trace.events:
            raise ValueError("a fresh ADAPT controller requires a run without unrelated attempts")
        state = _new_controller_state(controls, bounded=growth is not None)
    elif state.get("kind") != "adapt_gcim/3" or state.get("controls") != controls:
        raise ValueError("resume requires the identical ADAPT Plan/runtime/policy")
    indices, angles = state["selected"], state["theta"]
    context.accepted = tuple(zip(indices, angles, strict=True))
    context.basis = None if growth is None else saved_chains(state["basis"])
    active = state["active_energy"]
    _keep_current(
        context, current_basis(context),
        *((tuple(zip(indices, active["theta"], strict=True)),) if active else ())
    )
    # Rebuild only the observation index, preserving original contribution order.
    prior = run.observations
    for key, observation in _observation_index(prior.chunks).items():
        context.observations.setdefault(key, observation)
    for chunk in prior.chunks:
        if chunk.content_id not in context.contribution_seen:
            context.contribution_seen.add(chunk.content_id)
            context.contribution_order.append(chunk.content_id)

    # A checkpoint writes only the fields that changed since the previous one.
    # Fields other than the pencil, the fixed values of the final stages and
    # the decision history stay small and are written each time. Each
    # projected analysis replaces the pencil without editing it, the final
    # stages publish their pilot counts once and their allocation once
    # (_FINAL_FIXED), and the history changes only through project_current's
    # append and update_last, which both raise `revision`, so identity checks
    # and one integer comparison find their changes. A reopened Run records
    # the identities of the values it read, so it writes none of them again
    # until it replaces one. Raw observations belong to the Run, not to this
    # state.
    revision, absent = 0, object()
    replaced = ("pencil", *_FINAL_FIXED.values())

    def identities():
        return dict(revision=revision, **{name: state.get(name, absent) for name in replaced})

    saved = identities() if restored else None

    def save():
        """Checkpoint ``state``. After a known previous checkpoint, name the changed fields."""
        nonlocal saved
        changed = None
        if saved is not None:
            changed = [name for name in state
                       if name != "history" and (name not in replaced or state[name] is not saved[name])]
            if revision != saved["revision"]:
                changed.append("history")
        sequence = run.checkpoint(state, changed=changed)
        saved = identities()
        return sequence

    def accept_chunk(point, chunk, *, fresh=False):
        """Index a query's observation (one chunk or its point chunks) and, when new, append each to the contribution order."""
        context.observations[point.content_id] = chunk
        for item in _chunks(chunk):
            if fresh and item.content_id not in context.contribution_seen:
                context.contribution_seen.add(item.content_id)
                context.contribution_order.append(item.content_id)
        return chunk

    # Resolve pending work through its original attempt and preparation receipt.
    # An uncertain submission is not permission to create another population.
    def pending_event(pending):
        """Return the Run's attempt record for a pending query, or None if none was created.

        The attempt is found by its saved identity or by the checkpoint
        sequence reserved before submission. A found attempt must carry this
        query's exact resolved parameter point, seed and ``PreparedArtifact``, otherwise
        the checkpoint and the Run disagree and ``ValueError`` is raised.
        A pending pilot or main point of ``pencil_workflow`` saves no attempt
        identity and is prepared and submitted under its one ``sequence``.
        """
        event = (
            run.attempt_record(pending["attempt"])
            if pending.get("attempt") is not None
            else run.attempt_for_checkpoint(pending["sequence"])
        )
        if event is not None:
            receipt = run.prepared_artifact(event.prepared_id)
            if (
                receipt is None
                or receipt.realization != Realization.model_validate(pending["point"])
                or receipt.runtime != RuntimeOptions(seed=pending["seed"])
                or pending["prepared_id"] is not None
                and pending["prepared_id"] != receipt.content_id
            ):
                raise ValueError("checkpoint attempt differs from its exact pending query/runtime")
        return event

    def pending_status(pending):
        """Return the current status of a pending query's external work.

        Without an attempt it is the remote preparation's status, prefixed
        ``preparation_``, or the saved preparation status (``unsubmitted``
        by default). With an attempt it is ``completed``, the provider's
        submission status, or ``uncertain`` when no submission is recorded.
        ``_pending_stop_reason`` maps it to a stop reason.
        """
        event = pending_event(pending)
        if event is None:
            identity = pending.get("preparation_id") or run.preparation_for_checkpoint(
                pending.get("preparation_sequence", pending["sequence"])
            )
            preparation = run._state["remote_preparations"].get(identity)
            if preparation is not None:
                return "preparation_" + preparation.status
            return pending.get("preparation_status", "unsubmitted")
        if event.status == "completed":
            return "completed"
        if event.submission is not None:
            submission, _ = run._acquisition(event.attempt)
            return submission.status
        return "uncertain"

    def pending_completed():
        """Collect every pending query whose original attempt has completed and remove it from the pending set."""
        for pending in tuple(state["pending"]):
            event = pending_event(pending)
            if event is None or event.status != "completed":
                continue
            point = Realization.model_validate(pending["point"])
            chunk = _single_chunk(run.completed_observation(event.attempt))
            for item in _chunks(chunk):
                if item.prepared_id != event.prepared_id or item.realization_id != point.content_id:
                    raise ValueError("completed checkpoint differs from its original acquisition")
                run.collect(item)
            accept_chunk(point, chunk, fresh=True)
            state["pending"].remove(pending)
            save()

    # Acquire each exact realization once, or resume its saved preparation/job.
    # Persist its seed before any external work can become visible.
    def acquire(point):
        """Resume or start the data collection for one resolved query and record its result once."""
        if point.content_id in context.observations:
            return context.observations[point.content_id]
        pending = next(
            (item for item in state["pending"] if item["point_id"] == point.content_id), None
        )
        if pending is None:
            pending = dict(
                point=point.model_dump(mode="json", exclude_computed_fields=True),
                point_id=point.content_id,
                seed=run.rng.next_seed(),
                sequence=run.checkpoint_sequence + 1,
                preparation_sequence=run.checkpoint_sequence + 1,
                preparation_id=None,
                prepared_id=None,
                attempt=None,
            )
            state["pending"].append(pending)
            save()
        event = pending_event(pending)
        if event is not None:
            if event.status != "completed":
                raise _AcquisitionPending(pending_status(pending))
            chunk = _single_chunk(run.completed_observation(event.attempt))
        else:
            prepared_id = pending["prepared_id"]
            preparation_id = pending["preparation_id"] or run.preparation_for_checkpoint(
                pending["preparation_sequence"]
            )
            if prepared_id is None and preparation_id is not None:
                prepared_id = run._state["preparation_charges"][preparation_id].prepared_id
            _, observation = point.resolved_observation(plan)
            run.check_capacity(circuits=int(plan.execution == "quantum"), shots=observation.shots)
            if prepared_id is not None:
                prepared = run._state["handles"].get(prepared_id) or restore_prepared(
                    prepared_id, run=run, for_submission=True
                )
            elif preparation_id is not None:
                if preparation_id not in run._state["remote_preparations"]:
                    raise RuntimeError(
                        "original ADAPT preparation was interrupted; it was not repeated"
                    )
                from nwqlib._remote_preparation import refresh_preparation

                prepared = refresh_preparation(preparation_id, run=run)
            else:
                prepared = prepare_experiment(
                    point, run=run, runtime=RuntimeOptions(seed=pending["seed"])
                )
            if isinstance(prepared, PendingPreparation):
                pending["preparation_id"] = prepared.preparation_id
                save()
                raise _AcquisitionPending("preparation_" + prepared.status)
            pending["prepared_id"] = prepared.record.content_id
            save()
            if prepare_only:
                raise _PreparedOnly
            chunk = _single_chunk(submit_experiment(prepared, run=run, checkpoint_sequence=pending["sequence"]))
            if not all(isinstance(item, ObservationChunk) for item in _chunks(chunk)):
                raise _AcquisitionPending("pending_acquisition")
        for item in _chunks(chunk):
            run.collect(item)
        accept_chunk(point, chunk, fresh=True)
        state["pending"].remove(pending)
        save()
        return chunk

    def selected_tuple():
        """Return the current chain as ``((pool_index, theta), ...)``."""
        return tuple(zip(state["selected"], state["theta"], strict=True))

    # One optimizer objective may require several Pauli readout groups.
    # They share a logical energy request but remain separate physical acquisitions.
    def energy_points(theta, insertion):
        """Return the queries of one energy evaluation at angles ``theta``.

        Classical plans need one classical query. Exact quantum plans read every
        nonidentity label in one query, and sampled plans need one query per
        QWC energy group. A quantum plan for a scalar Hamiltonian needs none.
        """
        selected = tuple(zip(state["selected"], theta, strict=True))
        if plan.execution == "classical":
            return (_point(plan, "energy", right=selected, insertion=insertion),)
        if not reconstruction.nonidentity_terms:
            return ()
        groups = range(len(reconstruction.energy_groups)) if plan.shots else (0,)
        return tuple(
            _point(plan, "energy", right=selected, insertion=insertion, group=i) for i in groups
        )

    def energy_value(values):
        """Return ``<psi|H|psi>`` from the labels read from the energy queries.

        A classical query returns it directly. Quantum queries give the Pauli
        expectations, and ``<psi|H|psi> = c + sum_k c_k <P_k>`` with the
        identity coefficient ``c``.
        """
        if plan.execution == "classical":
            return values["energy"]
        return fsum(
            (
                reconstruction.identity_shift,
                *(t.coefficient * values[t.label] for t in reconstruction.nonidentity_terms),
            )
        )

    def finish_energy(request, points, values=None, gradient=False):
        """Return a completed energy and update the optimizer's incumbent for objective queries.

        ``values`` are the labels already read from these queries by
        ``complete_saved_energy``. Without them each collected chunk is read
        here once. With ``gradient`` a classical query also returns its
        ``gradient_j`` labels as an array, from the same evaluation.
        """
        if values is None:
            values = {}
            for point in points:
                values.update(_read_values(plan, point, context.observations[point.content_id]))
        value = energy_value(values)
        optimizer = state["optimizer"]
        if request["role"] == "objective" and optimizer is not None:
            if optimizer["initial_value"] is None:
                optimizer["initial_value"] = value
            if optimizer["best_value"] is None or value < optimizer["best_value"]:
                optimizer.update(
                    best_theta=request["theta"],
                    best_value=value,
                    best_sources=[context.observations[p.content_id].content_id for p in points],
                )
        state["active_energy"] = None
        save()
        if gradient:
            return value, np.array([values[f"gradient_{j}"] for j in range(len(request["theta"]))])
        return value

    # Charge a new logical objective before its constituent acquisitions, while
    # reusing already collected values at an identical parameter point.
    def evaluate_energy(theta, *, insertion=None, role="objective", gradient=False):
        """Return the product-state energy at ``theta``, counting a new point against the evaluation limit.

        This is the ``evaluate`` callback of ``optimize_product``. ``role`` is
        ``objective`` or ``derivative``, and ``insertion`` selects a
        Pauli-insertion derivative query. With ``gradient`` a classical query
        returns ``(energy, gradient)`` of one energy-and-gradient evaluation,
        so ``optimize_max_evaluations`` counts distinct evaluated parameter
        points, and an energy and gradient at the same point are counted once.
        Raises ``_EvaluationBudgetExhausted`` when a new point would exceed
        ``optimize_max_evaluations``.
        """
        points = energy_points(theta, insertion)
        request = dict(
            theta=list(theta), insertion=None if insertion is None else list(insertion), role=role
        )
        if all(point.content_id in context.observations for point in points):
            return finish_energy(request, points, gradient=gradient)
        if state["active_energy"] is not None:
            raise ValueError("optimizer advanced before its original energy query completed")
        if state["energy_queries"] >= options.optimize_max_evaluations:
            raise _EvaluationBudgetExhausted
        state["energy_queries"] += 1
        state["active_energy"] = request
        save()  # Reserve the logical query before any of its native attempts.
        return complete_saved_energy(gradient)

    def complete_saved_energy(gradient=False):
        """Run the energy query that was already counted, if any, and return its value (``finish_energy``)."""
        request = state["active_energy"]
        if request is None:
            return
        points = energy_points(request["theta"], request["insertion"])
        values = {}
        for point in points:
            chunk = acquire(point)
            if chunk is not None:
                values.update(_read_values(plan, point, chunk))
        if state["pending"]:
            raise _AcquisitionPending(pending_status(state["pending"][0]))
        return finish_energy(request, points, values, gradient)

    def result():
        """Analyze the Run's saved trajectory into an ``ADAPTResult``."""
        run.progress("analyze", 0, 1)
        analyzed = analyze_adapt(plan, run.data, settings={})
        run.progress("analyze", 1, 1)
        return analyzed

    def stop(reason):
        """Record the outcome that ends growth and checkpoint it, entering the final stages under ``sampling``."""
        if options.sampling is None:
            state.update(stop_reason=reason, phase="complete", pending_labels=[])
        else:
            state.update(stop_reason="running", phase="final", final=dict(stop_reason=reason), pending_labels=[])
        save()

    def receipt(prepared_id):
        """Return this Run's preparation record of an exact chunk, or None when it has none."""
        try:
            return run.prepared_artifact(prepared_id)
        except ValueError:
            return None

    # Assemble the current H/S pencil from acquired matrix elements and publish
    # one projected-energy decision with the exact basis and observation range.
    def project_current(basis):
        """Solve the obtained projected pencil of the basis chains ``basis`` and checkpoint its basis-specific decision."""
        nonlocal revision
        selected = selected_tuple()
        state["analysis_in_progress"] = True
        state["analysis_attempts"] += 1
        save()
        run.progress("analyze", 0, 1)
        pencil = _pencil_from_observations(plan, basis, context, receipt=receipt)
        run.progress("analyze", 1, 1)
        state.update(
            analysis_in_progress=False,
            value=pencil.eigenvalues[0] if pencil.eigenvalues else None,
            value_selected=[i for i, _ in selected],
            value_theta=[x for _, x in selected],
            coefficients=[[z.real, z.imag] for z in pencil.coefficients],
            current_analysis_attempt=state["analysis_attempts"],
            pencil=pencil.model_dump(mode="json", exclude_computed_fields=True),
        )
        if growth is not None:
            state["value_basis"] = [list(map(list, chain)) for chain in basis]

        def publish_pencil():
            context.pencil, context.pencil_parameters = pencil, selected
            context.pencil_attempt = state["current_analysis_attempt"]
            # A durable Run refreshes its method cache once per analysis.
            run._state["cache_method_dirty"] = True

        iteration = len(selected) if growth is None else len(basis) - 1
        previous = state["history"][-1]["energy"] if state["history"] else None
        state["flat_count"], state["previous_energy"] = _flat_counter_update(
            state["flat_count"],
            state["previous_energy"],
            state["value"],
            iteration,
            valid=state["value"] is not None and pencil.failure_reason is None,
            tolerance=options.energy_change_tolerance,
        )
        # The diagonal query of the full product identifies this round's
        # basis in its history record (AdaptRound.basis_realization), and the
        # completed matrix stage must contain it. A classical host query is the
        # diagonal pair itself, and an exact quantum run the shared full-chain
        # query, whose screening values the later screen reuses. A sampled
        # diagonal pair has no overlap query (S_ii = 1), so its first
        # Pauli-term query, or its first-group grouped setting, is used. A
        # bounded basis has no shared full-chain query, so its last chain's
        # diagonal pair query identifies it.
        exact_quantum = plan.execution == "quantum" and plan.shots is None
        grouped = reconstruction.acquisition.matrix_measurement == "grouped"
        full = selected if growth is None else basis[-1]
        basis_point = (
            None
            if not full or ((exact_quantum or grouped) and not reconstruction.nonidentity_terms)
            else _point(plan, "screen", right=full)
            if exact_quantum and growth is None
            else _point(plan, "pair", full, full)
            if exact_quantum
            else _point(plan, "diagonal", full, full, group=0)
            if grouped
            else _point(
                plan,
                "pair",
                full,
                full,
                term=0
                if plan.shots is not None and reconstruction.nonidentity_terms
                else -1,
            )
        )
        if basis_point is not None and basis_point.content_id not in context.observations:
            raise ValueError("completed matrix stage lacks its full-product source")
        record = dict(
            iteration=iteration,
            selected=[i for i, _ in selected],
            theta=[t for _, t in selected],
            analysis_attempt=state["current_analysis_attempt"],
            basis_realization=None if basis_point is None else basis_point.content_id,
            energy=state["value"],
            energy_decrease=None if previous is None or state["value"] is None else previous - state["value"],
            kept_rank=pencil.kept_rank,
            candidates=[],
            new_start=state["stage_start"],
            new_stop=len(context.contribution_order),
            gradient_norm=None,
            gradients=[],
            winner=None,
            residual_norm=None,
            flat_count=state["flat_count"],
            optimizer_attempts=state["optimizer_attempts"],
            optimizer_rounds=state["optimizer_rounds"],
            energy_queries=state["energy_queries"],
            sampled_failure=pencil.sampled_failure,
            sampled_enclosure_violation=pencil.sampled_enclosure_violation,
        )
        if len(state["history"]) != iteration:
            raise ValueError(
                "controller projected stage differs from its committed decision history"
            )
        state["history"].append(record)
        revision += 1
        state["stage_start"] = len(context.contribution_order)
        if (
            pencil.failure_reason is not None
            and pencil.overlap_filter == "sampled_positive_subspace"
        ):
            stop("invalid_sampled_evidence")
        elif state["value"] is None:
            stop("no_usable_overlap_subspace")
        else:
            state["phase"] = "residual" if options.stop_criterion == "residual_norm" else "decision"
            save()
        publish_pencil()

    def update_last(**fields):
        nonlocal revision
        state["history"][-1].update(fields)
        revision += 1

    limited = growth is not None and growth.basis_depth_limit is not None
    reference_bound = reference_cx_bound(reconstruction.reference, reconstruction.num_qubits)[0] if limited else 0

    def depth_weight(index):
        """Return the cached CX count bound of generator ``index`` at ``theta``, read once from its uncontrolled forward gate."""
        key = (index, options.theta)
        if key not in context.depth_weights:
            gate = _native_generator(index, options.theta, controlled=False, inverse=False,
                                     inputs=plan._native["inputs"], compiler_plans=compilers, context=context)
            context.depth_weights[key] = generator_cx_weight(gate.definition)
        return context.depth_weights[key]

    try:
        pending_completed()
        for pending in state["pending"]:
            status = pending_status(pending)
            if status in {"intent", "uncertain", "failed", "cancelled"}:
                raise _AcquisitionPending(status)
        complete_saved_energy()
        if run.cancel_requested is not None:
            state.update(stop_reason="cancelled", pending_labels=[run.cancel_requested])
            save()
            return None if state["pending"] else result()
        # SciPy BFGS has no saved internal continuation here. Resume pending
        # acquisitions first, then explicitly restart from the completed incumbent.
        if restored and state["phase"] == "optimize" and state["optimizer"] is not None:
            if state["pending"] or state["active_energy"] is not None:
                raise ValueError("BFGS restart requires the original energy query to finish")
            optimizer = state["optimizer"]
            state["theta"] = list(
                optimizer["best_theta"]
                if optimizer["best_value"] is not None
                else optimizer["starting"]
            )
            state["restart_pending"] = dict(
                starting=list(state["theta"]), incumbent_value=optimizer["best_value"]
            )
            state["optimizer"] = None
            state.update(stop_reason="running", pending_labels=[])
            save()
        if restored and state["analysis_in_progress"]:
            raise RuntimeError("ADAPT analysis was interrupted; call run.resume(reanalyze=True)")
        if restored and state["phase"] == "complete":
            return result()
        if not restored:
            save()
        def admit_stage(count):
            """Check ``count`` new pencil queries against the Run's limits, and exact pair reductions against ``max_products``.

            Each new pencil query is one circuit preparation and one
            acquisition, and each exact pair reduction registers
            ``pair_work`` host work.
            """
            if plan.shots is None:
                from .fixed_basis import pair_reduction_point
                from .pair_reducer import pair_work

                point = pair_reduction_point(
                    reconstruction.processed_hamiltonian.identity,
                    [t.coefficient for t in reconstruction.nonidentity_terms],
                    reconstruction.num_qubits, diagonal=False)
                _check_products(count * pair_work(json.loads(point.parameters)), plan.method.max_products)
            run.check_capacity(preparations=count, circuits=count, shots=count * (plan.shots or 0))

        # A matrix stage, or bounded rows stage, entered during this call has
        # acquired nothing yet. The stage that a restored call resumes may be
        # partly acquired, so its queries are admitted one at a time in acquire.
        resumed_matrix = restored and state["phase"] in ("matrix", "rows")
        while state["phase"] != "complete":
            if run.cancel_requested is not None:
                state.update(stop_reason="cancelled", pending_labels=[run.cancel_requested])
                save()
                return result()
            selected = selected_tuple()
            context.accepted = selected
            # The basis of this phase, the legacy basis of the selected chain
            # or the explicit bounded chains, computed once and passed to
            # every pencil function.
            basis = basis_chains(selected) if growth is None else saved_chains(state["basis"])
            context.basis = None if growth is None else basis
            # Before this phase's prune, so that the gates that depth
            # screening built for the extensions still to be evaluated
            # survive every prune of the rows phase.
            context.pending_items = frozenset(
                chain[-1] for chain in saved_chains(
                    label["chain"] for label in state["history"][-1]["candidates"] if label["reason"] == "candidate")
            ) if state["phase"] == "rows" else frozenset()
            _keep_current(context, basis)
            # Advance acquisition, projection, optional residual and stopping in order.
            # Each phase checkpoint records the next scientific action still required.
            if state["phase"] == "matrix":
                # Admit the whole stage before its first query, so a limit that
                # the stage would exceed refuses it before any of its work.
                if plan.execution == "quantum" and not resumed_matrix:
                    if growth is None:
                        # Every pair query of an analyzed earlier basis was obtained.
                        earlier = {
                            item
                            for record in state["history"]
                            for item in basis_chains(tuple(zip(record["selected"], record["theta"], strict=True)))
                        }
                        admit_stage(_new_pencil_queries(basis, earlier, reconstruction))
                    else:
                        # The earlier basis was analyzed and the added chain's row
                        # was obtained when it was scored, so normally nothing is new.
                        admit_stage(_missing_queries(plan, basis, len(basis) - 1, context.observations))
                resumed_matrix = False
                # The selected chain is the full product. Its exact shared query
                # gives its diagonal and the values a later screen phase reuses.
                # A bounded basis has no full product and no shared query.
                context.matrix_data = (basis, _matrix_from_observations(
                    plan, basis, context, acquire=acquire, receipt=receipt,
                    full=selected if growth is None else None))
                state["phase"] = "analysis"
                save()
            elif state["phase"] == "analysis":
                project_current(basis)
            elif state["phase"] == "residual":
                coefficients = tuple(complex(*parts) for parts in state["coefficients"])
                point = _point(
                    plan,
                    "residual",
                    right=selected,
                    coefficients=coefficients,
                    energy=state["value"],
                )
                residual = _read_values(plan, point, acquire(point))["residual_norm"]
                update_last(residual_norm=residual)
                state["phase"] = "decision"
                save()
            elif state["phase"] == "decision":
                reason = (_stopping_reason(state, options, reconstruction, len(selected)) if growth is None
                          else _bounded_stopping_reason(state, options, reconstruction, len(basis)))
                if reason is not None:
                    stop(reason)
                else:
                    state["phase"] = "screen" if growth is None else "select"
                    save()
            elif state["phase"] == "select":
                # Examine this round's extensions. The cursor and the labels
                # are saved before any of their rows is obtained, so a resume
                # repeats no examination.
                labels, cursor, exhausted = frontier(
                    basis, saved_cursor(state["cursor"]), pool_size=len(reconstruction.pool), theta=options.theta,
                    growth=growth, weight=depth_weight, reference_bound=reference_bound)
                state["cursor"] = [list(item) for item in cursor]
                update_last(candidates=[dict(label, chain=[list(item) for item in label["chain"]])
                                        for label in labels])
                if any(label["reason"] == "candidate" for label in labels):
                    state["phase"] = "rows"
                    save()
                else:
                    update_last(new_stop=len(context.contribution_order))
                    state["stage_start"] = len(context.contribution_order)
                    stop(_round_stop_reason(labels, exhausted))
            elif state["phase"] == "rows":
                # Each admitted extension's row and diagonal come from the
                # ordinary pencil path for the current chains plus that one
                # extension, so its matrix stage later finds them cached. Its
                # score is saved before the next extension's queries prune
                # the cache.
                labels = state["history"][-1]["candidates"]
                pending_rows = [label for label in labels if label["reason"] == "candidate"]
                if plan.execution == "quantum" and not resumed_matrix:
                    admit_stage(sum(_missing_queries(plan, (*basis, saved_chains([label["chain"]])[0]), len(basis),
                                                     context.observations) for label in pending_rows))
                resumed_matrix = False
                pencil = ProjectedPencil.model_validate(state["pencil"])
                current = metric = None
                floor = independence_floor(options)
                for label in pending_rows:
                    extended = (*basis, saved_chains([label["chain"]])[0])
                    if plan.shots is None:
                        h, s, _, _ = _matrix_from_observations(plan, extended, context, acquire=acquire,
                                                               receipt=receipt)
                    else:
                        # ADAPT.plan admits bounded growth with shots only for
                        # grouped settings, whose overlap moments decide
                        # whether the extension is scored. Planning already
                        # checked the projected-solve size of the enlarged
                        # basis, at most bounded_basis_size states.
                        moments, pairs, _ = _grouped_statistics(plan, extended, context, acquire)
                        h, s, _, _ = _grouped_matrices(len(extended), pairs)
                        entries = _overlap_settings(moments)
                    if current is None:
                        coordinates = [[complex(z.real, z.imag) for z in row] for row in pencil.coordinate_vectors]
                        current = current_projection(h[:-1, :-1], coordinates, pencil.offset_free_eigenvalues[0])
                        if plan.shots is not None:
                            metric = sampled_metric(current, {pair: entry for pair, entry in entries.items()
                                                              if pair[1] < len(basis)},
                                                    pencil.overlap_eigenvalues[-pencil.kept_rank])
                    if plan.shots is not None:
                        label.update(sampled_eligibility(current, metric, s[:-1, -1],
                                                         [entries[i, len(basis)] for i in range(len(basis))], floor),
                                     overlap_cutoff=pencil.overlap_cutoff, kept_rank=pencil.kept_rank)
                    if label["reason"] in ("candidate", None):
                        label.update(candidate_score(current, h[:-1, -1], s[:-1, -1], h[-1, -1].real, floor))
                    update_last(candidates=labels)
                    save()
                ranked = [k for k, label in enumerate(labels) if label["reason"] == "ranked"]
                update_last(new_stop=len(context.contribution_order))
                state["stage_start"] = len(context.contribution_order)
                if not ranked:
                    # With shots, a round none of whose evaluated extensions
                    # passed the sampled eligibility stops as noise-inconclusive
                    # on the current basis. One that passed it but has no finite
                    # score stops the round as with exact entries. A passed
                    # extension cannot fall below independence_cutoff in
                    # candidate_score, because its shared etahat is at least
                    # its margin (R >= 0), which exceeds the cutoff.
                    eligible = any(label["reason"] == "nonfinite_score" for label in labels)
                    stop("no_admissible_candidate" if plan.shots is None or eligible else "noise_inconclusive")
                    continue
                # Largest raw score first, the earliest examined among equal scores.
                winner = max(ranked, key=lambda k: (labels[k]["score"], -k))
                labels[winner]["reason"] = "accepted"
                update_last(candidates=labels)
                state["basis"].append(labels[winner]["chain"])
                state["phase"] = "matrix"
                save()
            # Evaluate unused generators through <[H,A_i]> and select the largest
            # absolute gradient. Pool order resolves only equal binary64 absolute values.
            # Gradients equal in exact arithmetic, for example by symmetry,
            # usually differ by roundoff, and the summation order then decides
            # (docs/algorithms/gcim.md, "Adaptive generator coordinates").
            elif state["phase"] == "screen":
                remaining = tuple(
                    i for i in range(len(reconstruction.pool)) if i not in state["selected"]
                )
                if plan.execution == "classical":
                    point = _point(plan, "screen", right=selected)
                    measured = _read_values(plan, point, acquire(point))
                    gradients = {i: measured[f"g_{i}"] for i in remaining}
                else:
                    active = active_screen_labels(
                        reconstruction.pool, label_cache(plan), state["selected"],
                        reconstruction.max_selections,
                    )[1]
                    ordered = screen_labels(reconstruction.pool, label_cache(plan))
                    measured = {}
                    groups = (
                        tuple(
                            i for i, g in enumerate(reconstruction.groups)
                            if any(ordered[k] in active for k in g)
                        )
                        if plan.shots
                        else (0,)
                        if active
                        else ()
                    )
                    for group in groups:
                        point = _point(plan, "screen", right=selected, group=group)
                        measured.update(_read_values(plan, point, acquire(point)))
                    # [H, A_i] = sum_t c_t P_t (adapt_inputs), so its expectation
                    # is the coefficient-weighted sum of the measured labels.
                    rows = commutator_rows(reconstruction.pool, label_cache(plan))
                    gradients = {
                        i: fsum(coefficient * measured[label] for label, coefficient in rows[i])
                        for i in remaining
                    }
                norm = stable_vector_norm(np.array(tuple(gradients.values())))
                if not isfinite(norm):
                    raise _PartialData(("gradient norm is not representable in binary64",))
                winner = max(remaining, key=lambda i: (abs(gradients[i]), -i))
                update_last(
                    gradient_norm=norm,
                    gradients=sorted(gradients.items()),
                    winner=winner if norm > options.gradient_norm_floor else None,
                    new_stop=len(context.contribution_order),
                )
                state["stage_start"] = len(context.contribution_order)
                if norm <= options.gradient_norm_floor:
                    stop("gradient_norm_floor")
                    continue
                state["selected"].append(winner)
                state["theta"].append(options.theta)
                count = len(state["selected"])
                state["phase"] = (
                    "optimize"
                    if options.optimize_every_m is not None
                    and count % options.optimize_every_m == 0
                    else "matrix"
                )
                state["optimizer"] = None
                save()
            # Optimize the selected product coordinates using the same budgeted energy
            # queries, then rebuild its projected basis at the best completed parameters.
            elif state["phase"] == "optimize":
                generators = tuple(plan._native["inputs"].pool[i] for i in state["selected"])
                rules = {}
                if plan.execution != "classical":
                    rules = _optimizer_rules(plan, generators, context)
                if state["optimizer"] is None:
                    state["optimizer_attempts"] += 1
                    if state["restart_pending"] is not None:
                        if not state["optimizer_restarts"]:
                            import warnings

                            warnings.warn(
                                "ADAPT resumes saved acquisitions and restarts official SciPy BFGS "
                                "from its completed incumbent; the optimizer search trajectory can change.",
                                UserWarning,
                                stacklevel=3,
                            )
                        state["optimizer_restarts"].append(
                            dict(
                                **state["restart_pending"],
                                attempt=state["optimizer_attempts"],
                                completed_rounds=state["optimizer_rounds"],
                            )
                        )
                        state["restart_pending"] = None
                    state["optimizer"] = dict(
                        starting=list(state["theta"]),
                        best_theta=list(state["theta"]),
                        best_value=None,
                        initial_value=None,
                        best_sources=[],
                        rounds=0,
                    )
                    save()
                optimizer = state["optimizer"]

                def completed_round():
                    optimizer["rounds"] += 1
                    state["optimizer_rounds"] += 1
                    save()

                reason, _ = optimize_product(
                    options=options,
                    execution=plan.execution,
                    selected=generators,
                    starting=state["theta"],
                    evaluate=evaluate_energy,
                    completed_round=completed_round,
                    rules=rules,
                )
                state["theta"] = list(optimizer["best_theta"])
                state["optimizer_exhausted"] = reason == "evaluation_budget_exhausted"
                state["optimizer"] = None
                state["phase"] = "matrix"
                save()
            # Under PencilSampling, measure the value basis afresh once growth
            # has stopped: a uniform pilot, one allocation and a main stage
            # (pencil_workflow.advance_stages). Every stage point binds its
            # stage, so no selection query or moment serves it, and the main
            # pencil is solved from the main-stage chunks alone.
            elif state["phase"] == "final":
                final, chosen = state["final"], _value_chains(state, growth is not None)
                if prepare_only:
                    # nwqlib.prepare drives a fresh Run, which reaches this
                    # phase only when it measures nothing.
                    raise _PreparedOnly
                if "stage" not in final:
                    m = (sampled_setting_count(len(chosen), len(reconstruction.energy_groups),
                                               reconstruction.acquisition.quadratures)
                         if reconstruction.acquisition.matrix_measurement == "grouped" else 0)
                    run.check_capacity(preparations=2 * m, circuits=2 * m,
                                       shots=options.sampling.total_shots if m else 0)
                    fields = pencil_workflow.new_state(options.sampling, m)
                    state.update({field: fields.pop(name) for name, field in _FINAL_FIXED.items()})
                    final.update(fields)
                    save()
                # advance_stages reads and changes the fields of new_state in
                # one dictionary, which exists only for this call. Each field
                # that one of its saves names is copied back to the stage
                # cursor `final` or to its fixed field before the one
                # checkpoint of that save, so a pending point takes the
                # sequence that pencil_workflow reserved for it.
                stages = dict(final, **{name: state[field] for name, field in _FINAL_FIXED.items()})

                def save_stages(*names):
                    for name in names:
                        if name in _FINAL_FIXED:
                            state[_FINAL_FIXED[name]] = stages[name]
                        else:
                            final[name] = stages[name]
                    return save()

                points = {}

                def resolve(stage, index, shots):
                    """Return the point of the index-th setting of the value basis at ``stage`` with ``shots``."""
                    if stage not in points:
                        points[stage] = tuple(point for _, _, _, point in _matrix_entries(plan, chosen, stage=stage))
                    return points[stage][index](shots)

                def decide():
                    """Return the allocation that ``pencil_workflow.pilot_decision`` makes from the pilot moments, read and oriented as for the selection pencils."""
                    settings, moments = _grouped_moments(
                        plan, chosen, AdaptContext(observations=_observation_index(run.observations.chunks)),
                        stage=0, counts=stages["pilot_counts"])
                    return pencil_workflow.pilot_decision(
                        options.sampling, settings, moments, basis_size=len(chosen),
                        identity_shift=reconstruction.identity_shift,
                        enclosure=processed_pauli_enclosure(reconstruction), cutoff=options.overlap_cutoff)

                if not pencil_workflow.advance_stages(run, stages, save=save_stages, resolve=resolve,
                                                      decide=decide):
                    if run.cancel_requested is not None:
                        continue
                    raise _AcquisitionPending(pending_status(stages["pending"]))
                counts = stages["decision"]["main_counts"]
                if counts:
                    state["analysis_in_progress"] = True
                    state["analysis_attempts"] += 1
                    save()
                    run.progress("analyze", 0, 1)
                    main = AdaptContext(observations=_observation_index(run.observations.chunks))
                    main.matrix_data = (chosen, _grouped_matrices(
                        len(chosen), _grouped_statistics(plan, chosen, main, stage=1, counts=counts)[1]))
                    pencil = _pencil_from_observations(plan, chosen, main)
                    run.progress("analyze", 1, 1)
                    state.update(analysis_in_progress=False,
                                 value=pencil.eigenvalues[0] if pencil.eigenvalues else None,
                                 coefficients=[[z.real, z.imag] for z in pencil.coefficients],
                                 current_analysis_attempt=state["analysis_attempts"],
                                 pencil=pencil.model_dump(mode="json", exclude_computed_fields=True))
                    context.pencil, context.pencil_attempt = pencil, state["current_analysis_attempt"]
                # The main pencil and the end of the iteration share one checkpoint.
                state.update(stop_reason=final["stop_reason"], phase="complete", pending_labels=[])
                save()
            else:
                raise ValueError("unknown saved ADAPT phase")
        return result()
    except _PreparedOnly:
        return None
    # Preserve unresolved external work as a frontier, with no automatic retry
    # that could duplicate preparation, submission or shots.
    except _AcquisitionPending as error:
        state.update(stop_reason=_pending_stop_reason(error.status), pending_labels=[error.status])
        save()
        return None
    # A refusal for installed package versions other than the recorded ones
    # (provenance) depends on the environment that reads the Run, not on the
    # query. It reaches the caller with the checkpoint unchanged, as in
    # FixedGCIM, so the same Run continues where the recorded versions are
    # installed. A depth above the limit or an unqualified depth would refuse
    # this query again whenever the Run resumes, so the iteration then ends
    # on its last solved pencil. The complete phase keeps a resumed drive from
    # preparing or submitting the refused query. Removing it from the pending
    # queries lets a resume after a cancellation return the Result, because
    # the cancellation branch returns None while a query is pending. The
    # energy request it belongs to, if any, is cleared, and its preparation
    # record stays in the Run without an attempt.
    except NativeDepthCapRefusal as error:
        if error.provenance is not None:
            raise
        state["pending"] = [item for item in state["pending"] if item["prepared_id"] != error.prepared_id]
        state.update(active_energy=None, stop_reason="native_depth_refused", phase="complete",
                     pending_labels=[str(error)])
        save()
        return result()
    except _PartialData as error:
        state.update(
            stop_reason="partial_observation"
            if "uncertain" not in str(error)
            else "uncertain_native_intent",
            pending_labels=list(error.pending),
        )
        save()
        return result()
    finally:
        # The Run's next journal write keeps the caches this drive produced.
        run._state["cache_method_dirty"] = True


def _optimizer_rules(plan, generators, context):
    """Resolve and cache one derivative rule per selected generator.

    The rule must come from the same checked generator and compiler plan
    that the circuit queries use, otherwise the measured derivative would
    describe another unitary. Rules are cached by pool index in the method
    context and saved with it, so resume reuses them.
    """
    from .optimization import _generator_shift_rule
    from nwqlib.execution import ExecutionMode

    rules = {}
    for generator in generators:
        index = generator.pool_index
        if (
            plan._native["inputs"].pool[index] is not generator
            or plan._native["compiler_plans"][index].generator is not generator
        ):
            raise ValueError("optimizer needs its actual selected generator/compiler")
        if index not in context.rules:
            context.rules[index] = _generator_shift_rule(
                generator,
                execution_mode=ExecutionMode.SHOTS if plan.shots else ExecutionMode.STATEVECTOR,
                compilation_plan=plan._native["compiler_plans"][index],
                max_bytes=plan.method.max_bytes,
                max_products=plan.method.max_products,
            )
        rules[index] = context.rules[index]
    return rules


def prepare_adapt(plan, *, run):
    """Prepare the first new query of the ADAPT iteration from the saved pending queries, without submitting it."""
    if run.plan is not plan:
        raise ValueError("ADAPT preparation requires its original Plan")
    return drive_adapt(plan, run=run, prepare_only=True)


def recover_adapt_analysis(plan, *, run):
    """Reopen an interrupted projected analysis for an explicit retry.

    Recovery is allowed only when the saved state is this Plan's analysis
    phase, nothing is pending and every attempt completed. Each original
    matrix observation is validated before the state is cleared, and the
    retry is recorded in ``analysis_retries``. The next drive then repeats the
    analysis from the same observations. This is the explicit step behind
    ``run.resume(reanalyze=True)`` (docs/run_archives.md).

    In the ``final`` phase of ``ADAPT.sampling`` it clears an interrupted
    analysis of the main stage, or an interrupted allocation, which
    ``pencil_workflow.advance_stages`` marks with
    ``final["allocation_in_progress"]``. Before it clears the marker, it
    requires no pending stage point and a count table of every setting of
    the value basis in the interrupted stage, main or pilot, and checkpoints
    the retry, which names that stage. Nothing is measured or submitted
    again.
    """
    state = run.checkpoint_state
    final = state.get("final") if isinstance(state, dict) else None
    allocation = final is not None and final.get("allocation_in_progress", False)
    if (
        run.plan is not plan
        or not isinstance(state, dict)
        or state.get("kind") != "adapt_gcim/3"
        or state.get("controls", {}).get("plan_id") != plan.content_id
        or not (state.get("analysis_in_progress") or allocation)
        or state.get("phase") != ("analysis" if final is None else "final")
        or state.get("pending")
        or state.get("active_energy") is not None
        or final is not None and final.get("pending") is not None
    ):
        raise ValueError(
            "reanalyze requires this ADAPT controller's interrupted projected analysis"
        )
    trace = run.trace
    if any(event.status != "completed" for event in trace.events):
        raise ValueError("analysis recovery cannot resolve an uncertain acquisition")
    bounded = plan.method.basis_growth is not None
    if final is None:
        basis = (saved_chains(state["basis"]) if bounded
                 else basis_chains(tuple(zip(state["selected"], state["theta"], strict=True))))
        stage, counts = -1, None
    else:
        basis = _value_chains(state, bounded)
        stage, counts = (0, state["pilot_counts"]) if allocation else (1, state["shot_allocation"]["main_counts"])
    observations = _observation_index(run.observations.chunks)
    grouped = plan.reconstruction.acquisition.matrix_measurement == "grouped"
    for _, _, _, point in _matrix_entries(plan, basis, observations, stage=stage, counts=counts):
        point = point()
        chunk = observations.get(point.content_id)
        if chunk is None:
            raise ValueError(
                "interrupted analysis lacks its complete original projected population"
            )
        for item in _chunks(chunk):
            trace.validate_observation(item)
        if not grouped:
            _read_values(plan, point, chunk)
    if grouped:
        # One admitted pass reads every grouped chunk, as the retried analysis does.
        _grouped_statistics(plan, basis, AdaptContext(observations=observations), stage=stage, counts=counts)
    attempts = final["analysis_attempts"] if allocation else state["analysis_attempts"]
    retry = dict(previous_attempts=attempts, sequence=run.checkpoint_sequence + 1,
                 **({} if final is None else dict(stage=stage)))
    state["analysis_retries"] = [*state.get("analysis_retries", ()), retry]
    run.checkpoint(state)
    if final is not None:
        final["allocation_in_progress"] = False
    state.update(analysis_in_progress=False, stop_reason="running", pending_labels=[])
    run.checkpoint(state)


def _matched_query_chunks(plan, data):
    """Yield every chunk of ``data`` after validating its association with its own query and readout.

    This is ``_eigen_support.matched_chunks`` for ADAPT's queries, which also
    include the two-point shared query on the full product state. A point
    chunk of a trajectory with several points carries its point's one-point readout
    (``ObservationSpec.point_observations``), which is compared instead of
    the whole schedule.
    """
    if data.trace.plan_id != plan.content_id:
        raise ValueError("analysis trace belongs to another selected Plan")
    seen = set()
    for chunk in data.observations.chunks:
        if chunk.content_id in seen:
            raise ValueError("an acquisition cannot contribute twice")
        seen.add(chunk.content_id)
        data.trace.validate_observation(chunk)
        if chunk.plan_id != plan.content_id or chunk.run_id != data.trace.run_id:
            raise ValueError("analysis contribution belongs to another Plan/run")
        point = plan.resolve(chunk.experiment, bindings=chunk.bindings)
        label, observation = point.resolved_observation(plan)
        if chunk.point is not None and len(observation.positions) > 1:
            observation = observation.point_observations().get(chunk.point)
        if (
            point.content_id != chunk.realization_id
            or point.bindings != chunk.bindings
            or label != chunk.setting
            or observation != chunk.observation
            or chunk.population != "unconditional"
        ):
            raise ValueError("analysis contribution differs from its actual selected point/readout")
        yield chunk


def analyze_adapt(plan, data, *, settings):
    """Read the saved adaptive trajectory and, for a new cutoff, solve its obtained pencil again.

    The history, selections and stop reason come from the saved iteration
    state as they were decided. With ``settings={"overlap_cutoff": c}`` the
    pencil of the value basis (``value_selected`` and ``value_theta``, or the
    explicit ``value_basis`` chains of bounded growth) is solved
    again from its saved observations at cutoff ``c``. No circuit runs and
    the history is not rewritten. Under ``ADAPT.sampling`` the
    ``ShotAllocation`` is read from the saved stage fields as decided, and
    once the analysis of the main stage has completed, the pencil, the group
    moments, the pair variances and ``contribution_ids`` are those of the
    main-stage chunks alone, also at a new cutoff. Otherwise, as after a
    stop inside a stage, ``contribution_ids`` lists the selection chunks
    (``stage`` -1) that grew the basis and no pilot or main-stage chunk.
    Without sampling it lists every chunk.
    """
    from .adapt import METHOD

    chunks = tuple(_matched_query_chunks(plan, data))
    import json

    state = json.loads(data.controller)["state"] if data.controller is not None else {}
    if state.get("controls", {}).get("plan_id") != plan.content_id:
        raise ValueError("ADAPT analysis requires its original controller trajectory")
    raw = state.get("pencil")
    pencil = None if raw is None else ProjectedPencil.model_validate(raw)
    cutoff = settings.get("overlap_cutoff", plan.method.overlap_cutoff)
    if not isinstance(cutoff, (int, float)) or not 0 < cutoff < float("inf"):
        raise ValueError("overlap_cutoff must be positive and finite")
    # A new overlap cutoff reuses the basis that produced the saved value.
    # Later selected-but-unprojected coordinates, or a bounded chain added
    # after the last analysis, must not replace that basis.
    bounded = plan.method.basis_growth is not None
    basis = _value_chains(state, bounded)
    context = AdaptContext(observations=_observation_index(chunks))
    # The saved pencil is the main stage's once its analysis has ended the
    # run, which only follows the complete main stage.
    final, decision = state.get("final"), state.get("shot_allocation")
    allocation = None if decision is None else ShotAllocation.model_validate(decision)
    main = (decision["main_counts"] or None
            if allocation is not None and final["stage"] == 2 and state.get("phase") == "complete" else None)
    # The value basis's group statistics and pair constructions are read from
    # the same observations and Plan rule, without new acquisition. One
    # grouped pass supplies the moments and the matrices a new cutoff solves.
    # Only exact and grouped queries prepare the joint state that
    # PairPreparation describes. A termwise Hadamard test places each Pauli
    # term between the two controlled preparations and prepares no such
    # state, so it has no record. The resolved choice decides: exact
    # execution resolves a requested termwise measurement to None.
    moments = variances = preparations = ()
    acquisition = plan.reconstruction.acquisition
    if (
        raw is not None
        and acquisition.pair_preparation is not None
        and acquisition.matrix_measurement != "termwise"
    ):
        preparations = tuple(chain_pair_preparation(basis[i], basis[j], acquisition.pair_preparation, (i, j))
                             for i, j in pencil_pairs(len(basis)) if i < j)
        if acquisition.matrix_measurement == "grouped":
            moments, pairs, variances = _grouped_statistics(plan, basis, context, stage=-1 if main is None else 1,
                                                            counts=main)
            if "overlap_cutoff" in settings:
                context.matrix_data = (basis, _grouped_matrices(len(basis), pairs))
    if "overlap_cutoff" in settings and raw is not None:
        receipts = {receipt.content_id: receipt for receipt in data.receipts}
        pencil = _pencil_from_observations(plan, basis, context, cutoff=cutoff, receipt=receipts.get)
    history = []
    for item in state.get("history", ()):
        history.append(
            AdaptRound(
                iteration=item["iteration"],
                selected=tuple(item["selected"]),
                theta=tuple(item["theta"]),
                energy=item["energy"],
                gradient_norm=item["gradient_norm"],
                gradients=tuple(tuple(pair) for pair in item["gradients"]),
                winner=item["winner"],
                residual_norm=item["residual_norm"],
                flat_count=item["flat_count"],
                basis_realization=item["basis_realization"],
                analysis_attempt=item["analysis_attempt"],
                optimizer_attempts=item["optimizer_attempts"],
                optimizer_rounds=item["optimizer_rounds"],
                energy_queries=item["energy_queries"],
                contribution_ids=tuple(
                    chunk.content_id for chunk in chunks[item["new_start"] : item["new_stop"]]
                ),
                sampled_failure=item["sampled_failure"],
                sampled_enclosure_violation=item["sampled_enclosure_violation"],
                energy_decrease=item["energy_decrease"],
                kept_rank=item["kept_rank"],
                candidates=tuple(BoundedCandidate.model_validate(dict(label, chain=saved_chains([label["chain"]])[0]))
                                 for label in item["candidates"]),
            )
        )
    value = (
        (pencil.eigenvalues[0] if pencil.eigenvalues else None)
        if pencil is not None
        else state.get("value")
    )
    # The main pencil uses the main-stage chunk behind each of its moments.
    # Before a completed main analysis, as after a stop inside a stage, the
    # Result reports the last pencil that the growth solved, so it uses the
    # whole selection trajectory: the chunks whose `stage` is -1, or every
    # chunk without PencilSampling, which binds no `stage`. Pilot chunks only
    # choose the main shots, and main-stage chunks enter the Result only
    # through a completed main analysis.
    if main is None:
        used = {chunk.content_id for chunk in chunks
                if {item.parameter: item.value for item in chunk.bindings}.get("stage", -1) == -1}
    else:
        used = {moment.contribution_id for moment in moments}
    return ADAPTResult(
        plan_id=plan.content_id,
        construction_id=plan.construction.content_id,
        observation_id=data.observations.content_id,
        contribution_ids=tuple(chunk.content_id for chunk in chunks if chunk.content_id in used),
        eigenvalue=value,
        selected=tuple(state.get("selected", ())),
        theta=tuple(state.get("theta", ())),
        value_selected=tuple(state.get("value_selected", ())),
        value_theta=tuple(state.get("value_theta", ())),
        history=tuple(history),
        pencil=pencil,
        stop_reason=state.get("stop_reason", "not_started"),
        pending=tuple(state.get("pending_labels", ())),
        optimizer_attempts=state.get("optimizer_attempts", 0),
        optimizer_rounds=state.get("optimizer_rounds", 0),
        optimizer_restarts=tuple(state.get("optimizer_restarts", ())),
        energy_queries=state.get("energy_queries", 0),
        # The selection and main analyses are counted at the top level, the
        # pilot allocations in the stage cursor, which a final state that
        # holds only its stop reason does not have yet.
        analysis_attempts=state.get("analysis_attempts", 0) + (final or {}).get("analysis_attempts", 0),
        analysis_cutoff=cutoff,
        group_moments=moments,
        sampled_pair_variances=variances,
        pair_preparations=preparations,
        basis=saved_chains(state.get("basis", ())),
        value_basis=basis if bounded else (),
        frontier_cursor=saved_cursor(state.get("cursor", ())),
        shot_allocation=allocation,
        origin=capture_analysis_origin(
            analyzer=METHOD, method_id=plan.method.content_id, dependencies=("numpy", "scipy")
        ),
        facts=exact_readout_sampling(plan, data.observations, random_draws=False),
    )._attach(plan, data)
