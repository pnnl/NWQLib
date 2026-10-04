"""ADAPT query construction, acquisition on the common Run, and the adaptive controller.

The controller is ADAPT-GCIM (Zheng et al. 2024, arXiv:2312.07691v3). Each
query is acquired once through the Run that owns preparation, submission and
checkpoints, so resume retrieves the original work instead of repeating it.
See docs/development/execution.md and docs/run_archives.md for that contract.
"""

from dataclasses import dataclass, field
import json
from math import fsum, isfinite, pi
import numpy as np
from nwqlib._numerics import normalize_state_vector, stable_vector_norm
from nwqlib._projected_eigensolver import (
    _sampled_pencil_diagnostics,
    _solve_projected_pencil,
    gram_formation_allowance,
)
from nwqlib.blocks.kernels import BoundKernel, KernelOutput
from nwqlib.core.planning import Realization, RuntimeOptions
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.evidence.error_model import exact_readout_sampling
from nwqlib.core.records import Complex128
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
    active_screen_labels,
    chain,
    commutator_rows,
    energy_labels,
    label_cache,
    screen_labels,
)
from .fixed_basis import (
    ProjectedPencil,
    SAMPLED_PENCIL_ENCLOSURE_RTOL,
    _exact_overlap_allowance,
    processed_pauli_enclosure,
    sampled_pencil_fields,
)
from .pencil import (
    _projected_solve_bytes,
    _projected_solve_work,
    assemble_pencil,
    matrix_element_count,
    pair_count,
    assemble_pair_pencil,
    matrix_elements,
    physical_hamiltonian,
)


@dataclass
class AdaptContext:
    """Method state; an explicit run archive keeps actual numerical/SDK data.

    Attributes:
        vectors: Classical prefix states keyed by generator chain, read-only once stored.
        h_columns: Classical ``(H - cI)|psi>`` (``_pair_scalars``) for chains
            used as a pair's right side.
        preparations: Native composed preparations keyed by (chain, controlled, inverse).
        gates: Native generator gates keyed by (pool index, angle, controlled, inverse).
        reference_gates: Native reference preparation variants.
        observations: Accepted observation chunk for each exact realization identity.
        contribution_order: Chunk identities in first-collection order.
        contribution_seen: Set view of ``contribution_order`` for membership tests.
        accepted: Current selected chain whose basis the caches must keep.
        pencil: Latest published projected pencil.
        pencil_parameters: Chain that produced ``pencil``.
        pencil_attempt: Controller analysis attempt that produced ``pencil``.
        matrix_data: Current ``(H0, S, sampled, allowance)`` workspace between
            acquisition and projection (``_matrix_from_observations``).
            Snapshots and archives exclude it and rebuild it from observations.
        pair_values: Values ``_read_values`` returned for each read pencil
            query, with the preparation receipt of its chunk, keyed by the
            canonical chain pair, term and quadrature (``_matrix_entries``).
            Later matrix stages reuse them instead of reading the same
            observations again. Snapshots and archives exclude them, and a
            restored or reanalyzed context rebuilds them from observations.
        rules: Resolved shift rule per selected pool index.
        compiler_plans: The quantum Plan's ``adapt.CompilerPlans`` map once
            ``drive_adapt`` has adopted it, so checkpoints save the compiler
            plans built after the Plan archive was written. Before that, a
            reopened Run holds the saved ``(pool index, kind, active modes,
            occupation blocks)`` entries here. Result snapshots store the
            built entries (``ADAPT.snapshot_result_context``), which native
            verification adopts.
        result: Result restored with a saved context. The live controller does
            not set it, so it is normally None.

    The native caches (``gates``, ``preparations``, ``reference_gates``) hold
    mutable Qiskit objects and stay private to the live Run. Result snapshots
    share only the immutable scientific entries.
    """

    vectors: dict = field(default_factory=dict)
    h_columns: dict = field(default_factory=dict)
    preparations: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    reference_gates: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    contribution_order: list = field(default_factory=list)
    contribution_seen: set = field(default_factory=set)
    accepted: tuple = ()
    pencil: ProjectedPencil | None = None
    pencil_parameters: tuple | None = None
    pencil_attempt: int | None = None
    matrix_data: tuple | None = None  # Current workspace only; excluded from snapshots/archives.
    # The Run's shifted sparse Hamiltonian, keyed by operator identity, exact
    # shift, representation and dtype; rebuilt on demand, never saved.
    shifted: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    pair_values: dict = field(default_factory=dict)  # Excluded from snapshots/archives.
    compiler_plans: object = None


def basis_chains(selected):
    """Return the ADAPT-GCIM working basis for the selected chain, in fixed order.

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


def _keep_current(context, *queries):
    """Drop cached states, actions and gates that no current basis or query can reuse.

    Only prefixes of the accepted basis chains and of the active queries are
    kept. Cache memory therefore stays proportional to the current basis
    instead of growing with the whole trajectory. A dropped entry is
    recomputed if a later query needs it again.
    """
    keep = {()}
    for items in (*basis_chains(context.accepted), *queries):
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
    pairs = {item for items in keep for item in items}
    for key in tuple(context.gates):
        if key[:2] not in pairs:
            del context.gates[key]


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
    relative phase. ``max_direct_amplitudes`` is the lowering limit that the
    reference preparation was admitted against.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import (
        controlled as controlled_gate,
        inverse_realized_gate,
    )

    key = (items, controlled, inverse)
    if key in context.preparations:
        return context.preparations[key]
    q = reference.manifest.basis.dimension.bit_length() - 1
    off = int(controlled)
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
    circuit = QuantumCircuit(q + off)
    if not inverse:
        circuit.append(context.reference_gates[rkey], circuit.qubits, copy=False)
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
        circuit.append(context.reference_gates[rkey], circuit.qubits, copy=False)
    gate = circuit.to_gate()
    context.preparations[key] = gate
    return gate


def _native_generator(index, theta, *, controlled, inverse, inputs, compiler_plans, context):
    """The one actual generator factory used by queries and explicit verification."""
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


def construct_adapt(block, arguments, method_context):
    """Finite registered native builder; arguments are already admitted scalars.

    An exact ``pair`` prepares the phase-faithful joint state: a Hadamard and an
    X gate on ancilla qubit 0, the left preparation controlled on it, a second
    X and the right preparation controlled on it, so the zero branch holds the
    left state and the one branch the right state. A diagonal exact pair
    prepares its system state alone. A sampled ``pair`` is a Hadamard test on
    ancilla qubit 0. It applies a Hadamard gate,
    the controlled right preparation, the controlled Pauli term (for a
    Hamiltonian element), the controlled inverse left preparation, an optional
    S-dagger and a second Hadamard gate. The
    ancilla Z expectation is then ``Re<psi_L|P|psi_R>``, or the imaginary part
    with S-dagger. ``screen`` and ``energy`` prepare the right chain on the
    system register. A sampled query appends the basis rotations of its QWC
    group so every label in the group is read as a Z-parity.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.hamiltonian_evolution import apply_pauli_rotation, make_pauli_label

    name, inputs, reference, compiler_plans, policy = block._payload
    # drive_adapt saves this cache once per projected analysis and on return.
    context = method_context(AdaptContext, marks_changes=True)
    if type(context) is not AdaptContext:
        raise ValueError("native ADAPT query requires its actual method context")
    values = dict(arguments)
    left, right = chain(values, "l"), chain(values, "r")
    _keep_current(context, left, right)
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
    if pair and block.record.choice.startswith("pair:exact:"):
        # Exact pair query: the phase-faithful joint state
        # (|0>|left> + |1>|right>)/sqrt(2) with the ancilla at native bit 0,
        # or the plain system state of a diagonal pair, reduced at acquisition
        # by the registered weighted-pencil reduction.
        if left == right:
            circuit.append(
                _native_preparation(left, controlled=False, inverse=False, **kwargs),
                circuit.qubits[1:],
                copy=False,
            )
        else:
            circuit.h(0)
            circuit.x(0)
            circuit.append(
                _native_preparation(left, controlled=True, inverse=False, **kwargs),
                circuit.qubits,
                copy=False,
            )
            circuit.x(0)
            circuit.append(
                _native_preparation(right, controlled=True, inverse=False, **kwargs),
                circuit.qubits,
                copy=False,
            )
    elif pair:
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
            for qubit in range(q):
                axis = next(
                    (label[q - 1 - qubit] for label in labels if label[q - 1 - qubit] != "I"), "I"
                )
                if axis == "Y":
                    circuit.sdg(qubit)
                if axis in "XY":
                    circuit.h(qubit)
    return circuit


def _reference_vector(plan, context, counts):
    """Return the cached normalized reference state, materializing it once per Run."""
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
    """Accumulate the Ritz state in ``basis_chains`` order and release a basis or prefix array after its final contribution and successor evolution.

    Product prefixes needed after the full-product contribution remain
    available. Native verification evolves the same independently lowered
    circuits, and the final state uses the existing normalization routine.

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
    same when the callbacks return the same arrays; a zero coefficient is
    still added. The node-set iteration order changes only the integer use
    counts, never the evolution or summation order.

    ``reference_at()`` returns the reference array and
    ``evolve(previous, chain)`` the array of ``chain`` from the array of
    ``chain[:-1]``; it receives the whole chain so a classical caller can
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


def _ritz_state(selected, coefficients, reference_at, evolve, counts=None):
    """Return the normalized Ritz state ``sum_i c_i |b_i>`` of the basis of ``selected``.

    ``|b_i>`` runs over ``basis_chains(selected)`` and ``c_i`` over
    ``coefficients`` in the same order. The sum is streamed by
    ``stream_ritz`` from the callbacks, which supply the classical or the
    native states, and normalized by ``normalize_state_vector``. No
    Hamiltonian action is applied.
    """
    basis = basis_chains(selected)
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
    of the two signs is that term's derivative contribution. The native
    query in ``construct_adapt`` applies the same factor.
    """
    insertion = int(number(values["insert_position"]))
    if insertion < 0:
        return _vector(right, plan, context, counts)
    state = _vector(right[: insertion + 1], plan, context, counts)
    tables = plan._native["inputs"].cache["action_tables"]
    term = tables["insertions"][right[insertion][0]][int(number(values["insert_term"]))]
    # exp(sign*i*pi*P/4)=(I+sign*iP)/sqrt(2), an exact Pauli identity; P is
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
    to ``counts``; a hit adds none. Pair queries (``_pair_scalars``) and
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
    equals <psi|[H,A]|psi>. The cache is keyed by the actual chain and
    processed Hamiltonian. A missing column is computed once with the same
    offset-free action used by pair acquisition.

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
    acquired it; it can be absent for a new chain, the initial screen, a
    restored context or after ``_keep_current`` evicts states, and is then
    computed once (``_offset_free_column``). Changing from ``H psi`` to
    ``(H-cI) psi`` changes rounding and can change selection among nearly
    tied gradients. The anti-Hermitian premise is exact admission of the
    generator coefficients, not a tolerance that permits a Hermitian
    component. Units are the Hamiltonian's energy unit per radian.

    Selected generators are not screened again, following QuGCM
    ``adapt_gcim.py`` (c5efcb0), lines 303-398. With a fixed angle, reusing
    a generator repeats its single-generator basis state, so reuse would
    define another trial space. Each screened generator adds one generator
    action and one scalar product to ``counts``; a cache hit adds no
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
        right,
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
    """Bind the classical ADAPT host kernel to the Plan without computing anything.

    Each classical query becomes one host-kernel invocation. ``binder``
    attaches a parameter point to the Run's state and action caches, and
    ``invoke`` evaluates it. ``screen`` returns the gradients
    ``<[H, A_i]>``, ``pair`` one ``(S_ij, H_ij)`` pencil entry, ``energy``
    one product-state energy and ``residual`` one Ritz residual norm. The
    returned application record counts the work actually performed:
    reference materializations and generator exponentials on cache misses,
    Hamiltonian and generator actions, and scalar products.
    """
    from .adapt import HOST

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
            """Evaluate one query's scalars and record the work actually performed."""
            _keep_current(context, left, right)
            counts = dict(
                reference_materializations=0,
                generator_exponentials=0,
                hamiltonian_actions=0,
                generator_actions=0,
                scalar_products=0,
            )
            # Peak live arrays: 8L + 12 complex128 vectors of 2**q amplitudes,
            # 16 bytes each, with L = max_selections. After _keep_current both
            # caches are keyed by the same kept prefixes: the 2k prefixes of the
            # basis chains of k <= L selected generators, including the
            # reference, and at most 2L more from the two query chains. The
            # state cache and the H column cache therefore hold at most 4L
            # vectors each. The remaining 12 vectors are an allowance for
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
            # at most 2k + 8 complex vectors at the last forward factor; the
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
            outer = (16 * (8 * rec.max_selections + 2 * k + 12) * d + 24 * k
                     if gradient else 16 * (8 * rec.max_selections + 12) * d)
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
            """Return the query's selected scalars and its work record."""
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

    Examples are an empty count population, a missing label or matrix
    entry, and a gradient norm that overflows binary64. The controller stops
    with ``partial_observation``, or ``uncertain_native_intent`` when the
    message names an uncertain acquisition, and analyzes the data it has
    instead of guessing the missing values.
    """

    def __init__(self, pending):
        self.pending = pending


class _AcquisitionPending(Exception):
    """External work for a query is not complete. ``status`` is its provider or preparation state.

    The controller checkpoints and returns, and a later resume retrieves the
    same attempt.
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
    queries therefore resolve to one realization identity and reuse its
    observation (``adapt._parameters``).
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
    defaults.update(
        control_width=int(experiment == "pair"),
        readout_width=1 if experiment == "pair" else plan.reconstruction.num_qubits,
    )
    defaults.update(
        {
            p.name: -1
            for p in plan.construction.program.parameters
            if "_pool_" in p.name or p.name in {"term", "insert_position", "insert_term"}
        }
    )
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

    The chunk must belong to exactly this point and its unconditional
    population. Classical chunks carry host scalars and exact quantum
    chunks carry Pauli expectations, which are used as returned. For counts,
    the readout circuit has already rotated every label of the query's group
    to a product of ``Z`` on the label's support, so its estimate is the
    sample mean of ``(-1)**(parity of the measured bits on that support)``.
    A pair query reads the ancilla bit alone. The set of labels must match
    ``ADAPT.labels_at`` exactly.

    The counts are read once through ``ObservationChunk.histogram()``. Each
    label's support is a bit mask in the Histogram index layout (bit ``p`` is
    measured qubit ``p``, the rightmost label character), and its parity on
    every stored index is one ``np.bitwise_count`` over the index array, or
    over the ``(entries, words)`` packed array for a readout wider than 64
    bits. The estimate is the exact integer ``total - 2 * odd`` divided once
    by ``total``, where ``odd`` is the count mass with odd parity. Each count
    is at most ``MAX_COUNT``, but a chunk's total is not, so ``total`` is a
    Python int. When it is at most ``MAX_COUNT``, every ``odd``, a sum of a
    subset of the nonnegative counts, fits in int64; otherwise ``odd`` is
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

        width = 1 if point.experiment == "pair" else plan.reconstruction.num_qubits
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
        words = packed.shape[1]
        result = {}
        for label in labels:
            support = (
                1
                if point.experiment == "pair"
                else sum(1 << i for i, axis in enumerate(reversed(label)) if axis != "I")
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


def _matrix_entries(plan, selected, available=()):
    """Yield ``((i, j), element, key, point)`` for every upper-triangle pencil entry.

    Each pair query is bound with its two chains in canonical (sorted) order,
    so the same unordered pair of states reuses one realization across
    iterations even when their basis positions change. When the canonical
    order reverses (i, j), the consumer conjugates the amplitude by flipping
    the sign of its imaginary part. Classical plans read all four H/S scalars
    from one query, exact quantum plans read ``(H0_ij, S_ij)`` from one
    weighted-pencil reduction per canonical chain pair (a diagonal pair only
    when the Hamiltonian has a nonidentity term), and sampled quantum plans
    have one query per ``MatrixElement``. In an exact quantum plan the
    diagonal of a chain that is, or was in an earlier round, the full chain
    comes from that chain's shared full-chain query (``available`` holds
    the realization identities already observed).
    ``key`` is ``(canonical left, canonical right, term, quadrature)``, which
    determines the query within one Plan, and ``point()`` resolves it.
    """
    basis = basis_chains(selected)
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
                    if left == tuple(selected) or shared.content_id in available:
                        yield (i, j), None, (left, left, -2, 0), (lambda shared=shared: shared)
                        continue
                yield (i, j), None, (*canonical, -1, 0), (
                    lambda canonical=canonical: _point(plan, "pair", *canonical))
    else:
        terms = plan._native["inputs"].nonidentity if plan._native else tuple(
            (t.label, t.coefficient) for t in plan.reconstruction.nonidentity_terms)
        indices = {p: i for i, (p, _) in enumerate(terms)}
        for element in matrix_elements(len(basis), terms, plan.reconstruction.num_qubits):
            i, j = element.basis_pair
            left, right = basis[i], basis[j]
            canonical = (right, left) if right < left else (left, right)
            term = -1 if element.matrix == "overlap" else indices[element.pauli_label]
            quadrature = int(element.quadrature == "imag")
            yield (i, j), element, (*canonical, term, quadrature), (
                lambda canonical=canonical, term=term, quadrature=quadrature: _point(
                    plan, "pair", *canonical, term=term, quadrature=quadrature))


def _matrix_points(plan, selected, available=()):
    """Yield ``((i, j), element, point)`` for every upper-triangle pencil entry (``_matrix_entries``)."""
    for pair, element, _, point in _matrix_entries(plan, selected, available):
        yield pair, element, point()


def _new_pencil_queries(selected, history, terms, *, exact=False):
    """Return a lower bound on the quantum pencil queries of ``selected`` that no earlier stage acquired.

    A pair query is bound to its two basis chains (``_matrix_points``), and
    every basis in ``history`` was analyzed, so all of its pair queries were
    acquired. Only a pair of chains that both appear among the earlier bases
    can have been acquired. With ``s`` such chains among the ``b`` current
    ones, at least ``c(b) - c(s)`` queries are new, where ``c`` is the count
    law ``pencil.matrix_element_count`` and ``terms`` the number of
    nonidentity Pauli terms. The bound is exact whenever the earlier chains
    that the current basis holds all belong to the previous basis, whose
    pairs were all acquired. That holds without optimization, where each
    basis contains the one before it, and after an optimization unless an
    angle returned exactly to its value in an older basis. It costs O(b)
    per history record and forms no query.
    """
    def count(size):
        if exact:
            return pair_count(size, terms)
        return matrix_element_count(size * (size + 1) // 2, size, terms)

    earlier = {
        item
        for record in history
        for item in basis_chains(tuple(zip(record["selected"], record["theta"], strict=True)))
    }
    current = basis_chains(selected)
    return count(len(current)) - count(sum(item in earlier for item in current))


def _matrix_from_observations(plan, selected, context, *, acquire=None, receipt=None):
    """Return ``(H0, S, sampled, allowance)`` of the basis of ``selected`` from its pair queries.

    ``H0`` and ``S`` are complex ``b``-by-``b`` arrays in basis-chain order,
    completed below the diagonal by conjugation. ``H0`` omits the identity
    term ``cI`` of the Hamiltonian, ``plan.reconstruction.identity_shift``.
    ``sampled`` is True when any entry came from shot counts. ``allowance``
    bounds the error of a deterministic S, which the solve's negative-mode
    test admits: the Gram formation error of classical inner products of
    length 2**n, or for exact pair reductions the largest row sum of the
    off-diagonal overlap entry bounds from each pair's receipt
    (``receipt(prepared_id)``, ``fixed_basis._exact_overlap_allowance``),
    zero when a receipt or its state error is unavailable. With
    ``acquire`` each missing query is acquired, otherwise only already
    collected observations are used and a missing one raises
    ``_PartialData``. Each observation is validated by ``_read_values``
    before it is used, and its values are kept in ``context.pair_values``,
    so a later stage reads only its new queries. A query bound in swapped
    canonical order measured ``<j|P|i> = conj(<i|P|j>)``, so it is
    conjugated (``_matrix_entries``), once per assembly.
    """
    basis = basis_chains(selected)
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
        for (i, j), _, key, point in _matrix_entries(plan, selected):
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
        # to both H0 and S; the reduction already returns complex entries, so
        # no per-quadrature sign adjustment applies. Its entry bounds come
        # from its own receipt's state error with the ordered-action envelope.
        from .fixed_basis import _coefficient_mass
        from .pair_reducer import entry_bounds

        rec = plan.reconstruction
        terms = len(rec.nonidentity_terms)
        c1 = _coefficient_mass([t.coefficient for t in rec.nonidentity_terms])
        pairs, bounds = {}, {}
        for (i, j), _, key, point in _matrix_entries(plan, selected, context.observations):
            v, prepared_id = read(key, point, f"matrix pair {i},{j}")
            h0, overlap = v["h0"], v["overlap"]
            if basis[j] < basis[i]:
                h0, overlap = h0.conjugate(), overlap.conjugate()
            pairs[(i, j)] = (h0, overlap)
            prepared = None if receipt is None else receipt(prepared_id)
            # The input array can carry a shared host phase correction; the
            # pair entries cancel the common global phase (saved-state budget).
            delta = None if prepared is None else prepared.saved_state_error(("amplitude-derived masses",))[0]
            entry = entry_bounds(delta, c1, 1 << rec.num_qubits, terms, 1, terms,
                                 diagonal=basis[i] == basis[j], ordered=True)
            bounds[(i, j)] = None if entry is None else entry[0]
        h, s = assemble_pair_pencil(b, pairs)
        allowance = _exact_overlap_allowance(b, bounds)
        # An unavailable allowance admits no input error beyond the solver's
        # own roundoff.
        allowance = 0.0 if allowance is None else allowance
    else:
        for (i, j), element, key, point in _matrix_entries(plan, selected):
            read_values = read(key, point, f"matrix {element.name}")[0]
            sign = -1 if element.quadrature == "imag" and basis[j] < basis[i] else 1
            values[element.name] = sign * next(iter(read_values.values()))
            # _read_values reads counts exactly when a quantum Plan has shots, and it
            # returned, so the chunk holds at least one count.
            sampled |= plan.execution == "quantum" and plan.shots is not None
        h, s = assemble_pencil(
            b,
            plan._native["inputs"].nonidentity if plan._native else tuple(
                (t.label, t.coefficient) for t in plan.reconstruction.nonidentity_terms),
            plan.reconstruction.num_qubits,
            values,
        )
        # A sampled overlap uses the positive-subspace policy, which admits no
        # deterministic input allowance.
        allowance = 0.0
    return h, s, sampled, allowance


def _pencil_from_observations(plan, selected, context, *, cutoff=None, receipt=None):
    """Solve the pencil of ``selected`` and return ``(ProjectedPencil, diagnostics)``.

    The matrices assembled by the live matrix stage (``context.matrix_data``)
    are used once and released. Otherwise, as after an interruption or for
    a new ``cutoff``, they are rebuilt from the saved observations, with
    ``receipt`` as in ``_matrix_from_observations``. The solve uses ``H0``
    and adds the identity coefficient c to the Ritz values, and the pencil
    records ``H = H0 + c S``. ``diagnostics`` is the sampled enclosure
    comparison (``_sampled_pencil_diagnostics``), or None for deterministic
    data or when an endpoint of the processed Pauli enclosure is not finite.
    Unlike ``fixed_basis.solve_pencil``, the solve honors the method's
    ``symmetrize_matrices`` setting.
    """
    cutoff = plan.method.overlap_cutoff if cutoff is None else cutoff
    current = context.matrix_data
    context.matrix_data = None
    h, s, sampled, allowance = (current[1] if current is not None and current[0] == selected
                                else _matrix_from_observations(plan, selected, context,
                                                               receipt=receipt))
    shift = plan.reconstruction.identity_shift
    result, spectrum, failure, _ = _solve_projected_pencil(
        h,
        s,
        overlap_eigenvalue_cutoff=cutoff,
        symmetrize_matrices=plan.method.symmetrize_matrices,
        overlap_input_tolerance=allowance,
        identity_shift=shift,
        _sampled_overlap=sampled,
    )
    kept = [] if spectrum is None else [float(x) for x in spectrum if x > cutoff]
    condition = max(kept) / min(kept) if kept else None
    if condition is not None and not isfinite(condition):
        condition = None

    def matrix_record(array):
        return tuple(
            tuple(Complex128(real=float(z.real), imag=float(z.imag)) for z in row) for row in array
        )

    diagnostics = None
    enclosure = processed_pauli_enclosure(plan.reconstruction) if sampled else None
    if enclosure is not None:
        diagnostics = _sampled_pencil_diagnostics(
            eigensolver=result,
            solver_failure_reason=failure,
            enclosure=enclosure,
            relative_tolerance=SAMPLED_PENCIL_ENCLOSURE_RTOL,
        )
    pencil = ProjectedPencil(
        hamiltonian=matrix_record(physical_hamiltonian(h, s, shift)),
        overlap=matrix_record(s),
        overlap_eigenvalues=() if spectrum is None else tuple(map(float, spectrum)),
        eigenvalues=() if result is None else tuple(map(float, result.eigenvalues)),
        coordinate_vectors=() if result is None else matrix_record(result.eigenvectors),
        kept_rank=0 if result is None else result.kept_overlap_rank,
        overlap_cutoff=cutoff,
        overlap_filter="sampled_positive_subspace" if sampled else "deterministic_gram",
        overlap_condition_number=condition,
        projected_residual=None if result is None else result.residual_norm,
        projected_backward_error=None
        if result is None
        else result.generalized_eigenpair_backward_error,
        overlap_normalization_error=None if result is None else result.overlap_normalization_error,
        failure_reason=failure,
        **sampled_pencil_fields(diagnostics),
    )
    return pencil, diagnostics


def _new_controller_state(controls):
    """Return the initial checkpoint state of a fresh ADAPT controller.

    The state is plain JSON so a Run checkpoint can store it. Its keys:

    - ``kind``, ``controls``: format tag and the Plan identity and optimizer
      resume policy. Resume requires both to be identical.
    - ``selected``, ``theta``: current generator chain, as pool indices and
      angles in radians.
    - ``phase``: next scientific action, one of ``matrix``, ``analysis``,
      ``residual``, ``decision``, ``screen``, ``optimize`` or ``complete``.
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
    - ``energy_queries``, ``active_energy``: logical energy queries charged
      to ``optimize_max_evaluations`` and the reserved query in progress.
    - ``analysis_attempts``, ``current_analysis_attempt``,
      ``analysis_in_progress``: projected-analysis bookkeeping for
      ``recover_adapt_analysis``.
    - ``stop_reason``: controller outcome, ``running`` until a stop.
    - ``stage_start``: first contribution index of the next history record.
    """
    return dict(
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


def _admit_saved_state(state, reconstruction, options):
    """Check the saved chain and counters before they are used to form states.

    A restored checkpoint must hold a chain of distinct in-pool indices with
    finite angles, no longer than ``max_selections``, and nonnegative
    integer counters within their limits. A state that fails raises
    ``ValueError`` instead of preparing a state or charging a query.
    """
    indices, angles = state["selected"], state["theta"]
    if (
        type(indices) is not list
        or type(angles) is not list
        or len(indices) != len(angles)
        or len(indices) > reconstruction.max_selections
        or len(indices) != len(set(indices))
        or any(type(i) is not int or not 0 <= i < len(reconstruction.pool) for i in indices)
        or any(type(x) not in (int, float) or not isfinite(x) for x in angles)
        or type(state["energy_queries"]) is not int
        or state["energy_queries"] < 0
        or options.optimize_max_evaluations is not None
        and state["energy_queries"] > options.optimize_max_evaluations
        or type(state["analysis_attempts"]) is not int
        or state["analysis_attempts"] < 0
        or state["current_analysis_attempt"] is not None
        and (
            type(state["current_analysis_attempt"]) is not int
            or not 0 < state["current_analysis_attempt"] <= state["analysis_attempts"]
        )
    ):
        raise ValueError("invalid saved ADAPT parameter/query state")


def _flat_counter_update(flat_count, previous_energy, value, iteration, *, valid, tolerance):
    """Return ``(flat_count, previous_energy)`` after one projected analysis.

    Zheng et al. (2024), arXiv:2312.07691v3, METHODS, p. 9, stop after
    ``T`` consecutive iterations whose change in the lowest eigenvalue is
    below a tolerance (``_stopping_reason``). The count starts with the
    change from the reference-only basis (iteration 0) to the first
    two-state basis, so iteration 0 only records its energy. A change at or
    above ``tolerance`` resets the count. An unavailable energy (no value or
    a solver refusal) also resets the count and forgets the previous energy,
    because it interrupts the sequence of small changes.
    """
    if not valid:
        return 0, None
    change = None if previous_energy is None or value is None else value - previous_energy
    if iteration:
        flat_count = flat_count + 1 if change is not None and abs(change) < tolerance else 0
    return flat_count, value


def _stopping_reason(state, options, reconstruction, count):
    """Return the controller's stop reason after an analysis with ``count`` selections, or None.

    The rules are checked in this order: the optimizer's evaluation limit
    was reached, the residual norm met ``residual_norm_tolerance``
    (``residual_norm`` stopping only), every pool member is selected, the
    iteration limit is reached, or the flat counter reached
    ``T = min(t_user, t_auto_fraction * unselected)`` (Zheng et al. (2024),
    arXiv:2312.07691v3, METHODS, p. 9, with ``T_auto`` 20% of the
    unselected operators). The integer counter reaches a fractional ``T`` at
    its ceiling. The reference-only analysis (``count == 0``) never stops on
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
    full-chain query has two points at one boundary, its weighted diagonal
    reduction and its screening labels, and keeps both chunks in schedule
    order; they share one acquisition and receipt.
    """
    if isinstance(completed, tuple) and len(completed) == 1:
        (completed,) = completed
    return completed


def _chunks(observation):
    """Return the chunks of one query's observation as a tuple."""
    return observation if isinstance(observation, tuple) else (observation,)


def _observation_index(chunks):
    """Index observations by realization, keeping the point chunks of one acquisition together.

    The first acquisition of a realization is kept, as for single chunks;
    a later point chunk of that same acquisition joins it in collection
    order, which is its schedule order.
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
    """One method controller over the original run, query budget and journal.

    The controller is a checkpointed state machine. Its phases are
    ``matrix`` (acquire every pencil entry of the current basis),
    ``analysis`` (solve the pencil and record one ``AdaptRound``),
    ``residual`` (classical residual stopping only), ``decision`` (apply the
    stopping rules), ``screen`` (measure ``<[H, A_i]>`` for unselected
    generators and append the largest absolute gradient, with equal binary64
    absolute values going to the lowest pool index) and ``optimize`` (optional BFGS every ``optimize_every_m``
    selections). The science follows ADAPT-GCIM, as described in the
    ``ADAPT`` docstring.

    Invariants that make interruption safe:

    - Each phase checkpoint names the next scientific action. The saved state
      holds the trajectory, the latest pencil and the pending frontier, while
      raw observations stay in the Run.
    - A new query records its seed and checkpoint sequence before any external
      work. Resume retrieves that original attempt or preparation and never
      creates a second population for the same point.
    - Unresolved external work stops the controller with a pending or
      uncertain reason instead of retrying, so no preparation, submission or
      shot is duplicated.
    - A logical energy query is charged to ``optimize_max_evaluations``
      before its acquisitions start, and an identical point reuses collected
      values without a new charge.
    - SciPy BFGS has no resumable internal state. After resume it restarts
      from the best completed point and records the restart.
    - An interrupted projected analysis is not repeated automatically. It
      needs ``run.resume(reanalyze=True)`` (``recover_adapt_analysis``).

    Args:
        plan: The ADAPT Plan that owns ``run``.
        run: The Run whose journal, RNG and observations this controller uses.
        prepare_only: Stop after preparing the first new query, without submission.

    Returns:
        The analyzed ``ADAPTResult``, or None while external work is pending or
        when only preparation was requested.
    """
    from .optimization import _EvaluationBudgetExhausted, optimize_product

    if run.plan is not plan:
        raise ValueError("ADAPT execution requires the same selected Plan and Run")
    options, reconstruction = plan.method, plan.reconstruction
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
        state = _new_controller_state(controls)
    elif state.get("kind") != "adapt_gcim/3" or state.get("controls") != controls:
        raise ValueError("resume requires the identical ADAPT Plan/runtime/policy")
    _admit_saved_state(state, reconstruction, options)
    indices, angles = state["selected"], state["theta"]
    context.accepted = tuple(zip(indices, angles, strict=True))
    active = state["active_energy"]
    _keep_current(
        context, *((tuple(zip(indices, active["theta"], strict=True)),) if active else ())
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
    # Fields other than the pencil and the decision history stay small and are
    # written each time. Each projected analysis replaces the pencil without
    # editing it, and the history changes only through project_current's
    # append and update_last, which both raise `revision`, so an identity
    # check and one integer comparison find their changes. Raw observations
    # belong to the Run, not to this state.
    revision = 0
    saved = None if not restored else dict(pencil=state.get("pencil"), revision=revision)

    def save():
        """Checkpoint ``state``. After a known previous checkpoint, name the changed fields."""
        nonlocal saved
        changed = None
        if saved is not None:
            changed = [name for name in state if name not in {"pencil", "history"}]
            if state.get("pencil") is not saved["pencil"]:
                changed.append("pencil")
            if revision != saved["revision"]:
                changed.append("history")
        sequence = run.checkpoint(state, changed=changed)
        saved = dict(pencil=state.get("pencil"), revision=revision)
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
        query's exact realization, seed and prepared artifact, otherwise the
        checkpoint and the Run disagree and ``ValueError`` is raised.
        """
        event = (
            run.attempt_record(pending["attempt"])
            if pending["attempt"] is not None
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
            identity = pending["preparation_id"] or run.preparation_for_checkpoint(
                pending["preparation_sequence"]
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
        """Collect every pending query whose original attempt has completed and remove it from the frontier."""
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
        """Resume or create the acquisition for one exact realization and publish it once."""
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

        Classical plans need one host query. Exact quantum plans read every
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
        ``complete_saved_energy``; without them each collected chunk is read
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
        """Return the product-state energy at ``theta``, charging a new point to the allowance.

        This is the ``evaluate`` callback of ``optimize_product``. ``role`` is
        ``objective`` or ``derivative``, and ``insertion`` selects a
        Pauli-insertion derivative query. With ``gradient`` a classical query
        returns ``(energy, gradient)`` of one energy-and-gradient evaluation,
        so ``optimize_max_evaluations`` counts distinct acquired parameter
        points and an energy and gradient at the same point share one charge.
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
        """Acquire the reserved energy query, if any, and return its value (``finish_energy``)."""
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
        """Record a final controller outcome and checkpoint it."""
        state.update(stop_reason=reason, phase="complete", pending_labels=[])
        save()

    def receipt(prepared_id):
        """Return this Run's preparation receipt of an exact chunk, or None when it has none."""
        try:
            return run.prepared_artifact(prepared_id)
        except ValueError:
            return None

    # Assemble the current H/S pencil from acquired matrix elements and publish
    # one projected-energy decision with the exact basis and observation range.
    def project_current():
        """Solve the acquired projected pencil and checkpoint its basis-specific decision."""
        nonlocal revision
        selected = selected_tuple()
        state["analysis_in_progress"] = True
        state["analysis_attempts"] += 1
        save()
        run.progress("analyze", 0, 1)
        pencil, diagnostics = _pencil_from_observations(plan, selected, context, receipt=receipt)
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

        def publish_pencil():
            context.pencil, context.pencil_parameters = pencil, selected
            context.pencil_attempt = state["current_analysis_attempt"]
            # A durable Run refreshes its method cache once per analysis.
            run._state["cache_method_dirty"] = True

        iteration = len(selected)
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
        # Pauli-term query is used.
        exact_quantum = plan.execution == "quantum" and plan.shots is None
        basis_point = (
            None
            if not selected or (exact_quantum and not reconstruction.nonidentity_terms)
            else _point(plan, "screen", right=selected)
            if exact_quantum
            else _point(
                plan,
                "pair",
                selected,
                selected,
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
            sampled_failure=None if diagnostics is None else diagnostics["failure_reason"],
            sampled_enclosure_violation=None
            if diagnostics is None
            else diagnostics["enclosure_violation"],
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
        # A matrix stage entered during this call has acquired nothing yet. The
        # stage that a restored call resumes may be partly acquired, so its
        # queries are admitted one at a time in acquire.
        resumed_matrix = restored and state["phase"] == "matrix"
        while state["phase"] != "complete":
            if run.cancel_requested is not None:
                state.update(stop_reason="cancelled", pending_labels=[run.cancel_requested])
                save()
                return result()
            selected = selected_tuple()
            context.accepted = selected
            _keep_current(context)
            # Advance acquisition, projection, optional residual and stopping in order.
            # Each phase checkpoint records the next scientific action still required.
            if state["phase"] == "matrix":
                # Admit the whole stage before its first query, so a limit that
                # the stage would exceed refuses it before any of its work. Each
                # new pencil query is one circuit preparation and one acquisition.
                if plan.execution == "quantum" and not resumed_matrix:
                    count = _new_pencil_queries(
                        selected, state["history"], len(reconstruction.nonidentity_terms),
                        exact=plan.shots is None)
                    if plan.shots is None:
                        # Each exact pair reduction registers pair_work host work;
                        # the whole stage is admitted against max_products here.
                        from .fixed_basis import pair_reduction_point
                        from .pair_reducer import pair_work

                        point = pair_reduction_point(
                            reconstruction.processed_hamiltonian.identity,
                            [t.coefficient for t in reconstruction.nonidentity_terms],
                            reconstruction.num_qubits, diagonal=False)
                        _check_products(count * pair_work(json.loads(point.parameters)), plan.method.max_products)
                    run.check_capacity(preparations=count, circuits=count, shots=count * (plan.shots or 0))
                resumed_matrix = False
                context.matrix_data = (selected, _matrix_from_observations(
                    plan, selected, context, acquire=acquire, receipt=receipt))
                state["phase"] = "analysis"
                save()
            elif state["phase"] == "analysis":
                project_current()
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
                reason = _stopping_reason(state, options, reconstruction, len(selected))
                if reason is not None:
                    stop(reason)
                else:
                    state["phase"] = "screen"
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

    The rule must come from the same admitted generator and compiler plan
    that the native queries execute, otherwise the measured derivative would
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
    """Prepare the controller's first new query through the saved frontier, without submission."""
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
    """
    state = run.checkpoint_state
    if (
        run.plan is not plan
        or not isinstance(state, dict)
        or state.get("kind") != "adapt_gcim/3"
        or state.get("controls", {}).get("plan_id") != plan.content_id
        or not state.get("analysis_in_progress")
        or state.get("phase") != "analysis"
        or state.get("pending")
        or state.get("active_energy") is not None
    ):
        raise ValueError(
            "reanalyze requires this ADAPT controller's interrupted projected analysis"
        )
    trace = run.trace
    if any(event.status != "completed" for event in trace.events):
        raise ValueError("analysis recovery cannot resolve an uncertain acquisition")
    selected = tuple(zip(state["selected"], state["theta"], strict=True))
    observations = _observation_index(run.observations.chunks)
    for _, _, point in _matrix_points(plan, selected, observations):
        chunk = observations.get(point.content_id)
        if chunk is None:
            raise ValueError(
                "interrupted analysis lacks its complete original projected population"
            )
        for item in _chunks(chunk):
            trace.validate_observation(item)
        _read_values(plan, point, chunk)
    retry = dict(previous_attempts=state["analysis_attempts"], sequence=run.checkpoint_sequence + 1)
    state["analysis_retries"] = [*state.get("analysis_retries", ()), retry]
    run.checkpoint(state)
    state.update(analysis_in_progress=False, stop_reason="running", pending_labels=[])
    run.checkpoint(state)


def _matched_query_chunks(plan, data):
    """Yield every chunk of ``data`` after validating its association with its own query and readout.

    This is ``_eigen_support.matched_chunks`` for ADAPT's queries, which also
    include the two-point shared full-chain query: a point chunk of a
    trajectory with several points carries its point's one-point readout
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
    """Read the saved adaptive trajectory, optionally rebuilding only its acquired pencil at a
    new cutoff.

    The history, selections and stop reason come from the saved controller
    state as they were decided. With ``settings={"overlap_cutoff": c}`` the
    pencil of the value basis (``value_selected``, ``value_theta``) is solved
    again from its saved observations at cutoff ``c``. Nothing is acquired
    and the history is not rewritten.
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
    # Later selected-but-unprojected coordinates must not replace that basis.
    if "overlap_cutoff" in settings and raw is not None:
        context = AdaptContext(observations=_observation_index(chunks))
        selected = tuple(zip(state["value_selected"], state["value_theta"], strict=True))
        receipts = {receipt.content_id: receipt for receipt in data.receipts}
        pencil, _ = _pencil_from_observations(plan, selected, context, cutoff=cutoff,
                                              receipt=receipts.get)
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
            )
        )
    value = (
        (pencil.eigenvalues[0] if pencil.eigenvalues else None)
        if pencil is not None
        else state.get("value")
    )
    return ADAPTResult(
        plan_id=plan.content_id,
        construction_id=plan.construction.content_id,
        observation_id=data.observations.content_id,
        contribution_ids=tuple(chunk.content_id for chunk in chunks),
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
        analysis_attempts=state.get("analysis_attempts", 0),
        analysis_cutoff=cutoff,
        origin=capture_analysis_origin(
            analyzer=METHOD, method_id=plan.method.content_id, dependencies=("numpy", "scipy")
        ),
        facts=exact_readout_sampling(plan, data.observations, random_draws=False),
    )._attach(plan, data)
