"""Selected Pauli product-formula actions for finite classical LCHS outputs."""

from math import cos, sin
import numpy as np

from nwqlib.operators._pauli import apply_terms, _pauli_masks, pauli_action_requirements
from nwqlib.operators.access import _check_bytes
from .time_independent_terms import (
    _PAULI_COEFFICIENT_ATOL,
    _trotter_pauli_decomposition, _pauli_decomposition_requirements,
    _distinct_combined_nodes, _node_bound_coefficients, _node_cache_bytes, _node_key, _first_use,
    _coefficient_preparation_work, _selected_node_table, _union_nonidentity_labels, _BorrowedRows,
    _budgeted_trotter_node_record, _lchs_trotter_error_budget_per_node,
    _LCHSTrotterNodeRecord,
)


# Logical payloads and qualified CPython allowances for compact host PF records.


def integer_bytes_from_bits(bits):
    """32 header/sign bytes plus four bytes per 30-bit magnitude limb."""
    return 32 + 4*((max(1, bits)+29)//30)


def integer_bytes(value):
    return integer_bytes_from_bits(abs(int(value)).bit_length())


def host_position_record_reserve(union, *, budgeted, fixed_steps, max_work):
    """Pre-admit one host position before its record or route is constructed.

    8192 covers the fixed-layout wrappers, floats and status strings.
    All integer fields are additional. A budgeted independent selection
    has at most 8500 step-count bits under the finite binary64 coefficient,
    time and remainder premises. Fixed steps use their actual width.
    The candidate reserve also covers a budgeted count refused by the
    method's step cap. Position work cannot exceed max_work when admitted.
    """
    pairs = union*(union-1)//2
    nested = union*(union-1)*(2*union-1)//6
    step_bytes = (integer_bytes_from_bits(8500) if budgeted
                  else integer_bytes(fixed_steps))
    counters = integer_bytes(union)+integer_bytes(pairs)+integer_bytes(nested)
    total = 8192 + step_bytes + counters + integer_bytes(max_work)
    if budgeted:
        total += integer_bytes(2) + step_bytes + counters
    return total


def host_phase_and_accumulation_work(dimension):
    """Charge -elapsed*c_I, exp(1j*phase)*v and one weighted accumulation.

    Negation and the identity-coefficient product cost two scalar visits.
    Forming and exponentiating the imaginary argument cost two more, and
    phase application costs D. Accumulation costs one scalar product,
    D vector products and D additions. Zero phase still takes this path.
    """
    return (dimension + 4) + (2*dimension + 1)


def _action_route(dimension, steps, rotations, *, max_work, max_bytes, retained_bytes=0):
    """Choose the cheaper admitted action route for one node.

    These are matrix/vector operation size proxies, not a wall-time ranking.
    The work excludes shared selection and accumulation, admitted by the caller.

    The vector route applies r*R Pauli rotations to one vector, with
    r = max(1, steps). Each rotation uses one single-label apply_terms action,
    priced by pauli_action_requirements(D, 1, 1, complex_input=True,
    label_width=q), followed by a D-entry scaling and a D-entry scaled
    addition. Its work is r*R*(W_action + 2*D) kernel/pass units. The scalar
    cosine, sine and imaginary-factor preparations are bundled into these
    update kernels. The three elementwise vector operations would instead
    cost 3*D visits under literal scalar counting.
    The dense route builds the D-by-D step matrix with R rotations (R*D**2),
    raises it to the r-th power by binary powering and applies it once (D**2).
    Binary powering needs floor(log2 r) squarings and popcount(r) - 1
    further products, each D**3.

    Returns:
        (route, work, known peak bytes, dense matrix products), with route
        ``vector`` or ``dense_power``. Equal work selects ``vector``.
    """
    vector_work, vector_bytes, _ = _route_requirements(dimension, steps, rotations, "vector")
    dense_work, dense_bytes, products = _route_requirements(dimension, steps, rotations, "dense_power")
    vector_bytes += retained_bytes
    dense_bytes += retained_bytes
    vector = vector_work <= max_work and vector_bytes <= max_bytes
    dense = dense_work <= max_work and dense_bytes <= max_bytes
    if vector and (not dense or vector_work <= dense_work):
        return "vector", vector_work, vector_bytes, 0
    if dense:
        return "dense_power", dense_work, dense_bytes, products
    raise ValueError("selected Pauli action exceeds max_select_work or max_bytes")


def _route_requirements(dimension, steps, rotations, route):
    """Return (work, scratch bytes, dense matrix products) of one saved action route (_action_route's laws)."""
    r = max(1, steps)
    if route == "vector":
        action_bytes, action_work = pauli_action_requirements(
            dimension, 1, 1, complex_input=True,
            label_width=dimension.bit_length()-1,
        )
        # One D-entry scaling and one D-entry scaled addition follow P @ v.
        # Scalar cos/sin preparation is bundled into these kernel/pass units.
        # An envelope of eight complex128 D-vectors for the result, the
        # scratch vector and temporaries, plus the single-label action's
        # bytes under its owner's law (B_action, retry-capable tile included).
        return (
            r*rotations*(action_work+2*dimension),
            128*dimension+action_bytes,
            0,
        )
    products = r.bit_length()-1+r.bit_count()-1
    # An envelope of six complex128 D-by-D arrays for the step matrix, its
    # transformed copy and matrix_power workspace, and eight D-vectors.
    return (rotations*dimension**2+products*dimension**3+dimension**2,
            96*dimension**2+128*dimension, products)


def _rotation_sequence(terms, elapsed, steps, order):
    """The same ordered Lie/Suzuki-2 step as the selected circuit synthesis.

    For terms P1,...,PM, Suzuki applies P1/2,...,P(M-1)/2,PM,
    P(M-1)/2,...,P1/2. Angles here are exponent angles, half Qiskit RZ angles,
    formed with the exact original operand order
    float(real(c))*elapsed/steps, followed by /2 for the order-two half
    occurrences, never real(c)*(elapsed/steps). The sequence is formed for
    the current application and node at application time from the stored
    coefficient rows, and ``terms`` may be a repeatable view of them.
    """
    if not len(terms):
        return ()
    # Same absolute imaginary-residue admission as the SELECT angle tables
    # (_PAULI_COEFFICIENT_ATOL, registered in ENGINEERING_CONSTANTS). Pauli
    # coefficients of the Hermitian L and H are real up to decomposition rounding.
    if any(abs(complex(value).imag)>_PAULI_COEFFICIENT_ATOL for _,value in terms):
        raise ValueError("Hamiltonian coefficients must be real for Pauli rotations")
    step = tuple((label, float(complex(value).real)*elapsed/steps) for label,value in terms)
    if order == 1:
        return step
    halves = tuple((label, angle/2) for label,angle in step[:-1])
    return halves+(step[-1],)+tuple(reversed(halves))


def _rotation_count(kept, order):
    """R_k = p_k at order one and max(0, 2*p_k - 1) at order two."""
    return kept if order == 1 else max(0, 2*kept-1)


def compact_pf_peak_bytes(
    *, held_other, decomposition_peak, decomposition_output,
    alignment, nodes, applications, union_count,
    live_selected, max_rotations, action_scratch,
    other_phase_peak=0,
):
    """B_PF = B_held,other + max{B_dec, B_coeff, B_live,selected + B_angles + max B_action}.

    B_coeff = B_dec,out + B_align + B_nodes + B_apps + 24u + 128: the
    decomposition output, the alignment, the node tables, the application
    records, a current node's mutable coefficients/indices before its
    snapshot (24u) and the streamed upper dropped-mass accumulator (128
    logical bytes, one node at a time). B_angles = 16*max_k R_k is the
    single shared angle buffer. The action maximum covers the routes
    actually selected, excluding held inputs. ``other_phase_peak`` is a
    separately reached phase, here the shared census. Inputs are logical
    populations from the actual representation, not RSS measurements.
    """
    coefficient_phase = (
        decomposition_output + alignment + nodes + applications
        + 24 * union_count + 128
    )
    action_phase = live_selected + 16 * max_rotations + action_scratch
    return held_other + max(
        decomposition_peak, coefficient_phase, action_phase, other_phase_peak
    )


def select_actions(data, *, elapsed, source_nodes, source_weights, has_initial):
    """Select actual per-node steps and route once, before host action.

    Store the selected coefficient table once per k node and the elapsed
    time, step count and route per application. Derive rotations with the
    selected coefficient-times-elapsed-divided-by-steps order. Admission
    follows the maximum of decomposition, node-table construction and
    action phases, while every selected payload still live remains charged.

    The node table (time_independent_terms.LCHSProductFormulaNodes) holds
    one shared union label table, each distinct node's coefficient and
    union-index arrays, its identity coefficient and upper dropped mass.
    Each application (the initial state, then each Duhamel node) keeps, for
    every quadrature position, its step record (fixed, or budgeted per node)
    and the cheaper admitted route; per-application records stay separate
    even when two durations coincide. Every distinct k-node is combined and
    pruned once, and its bound coefficient comes from one admitted shared
    census (_node_bound_coefficients); each application then selects its
    own steps at its elapsed time, pruning bound and remaining allowance.
    Routes are chosen only after every record is admitted, so a later
    record cannot invalidate an earlier dense route. Execution replays these
    choices unchanged and forms each rotation sequence at application time
    (_rotation_sequence).

    Bytes, with u the union size including identity, s the summed L/H
    support, K distinct nodes, p_k their kept counts, d_k their dropped
    counts, A applications and N_grid quadrature positions:
    B_dec,out = 32*d**2 + (q+16)*s, B_align for the aligned union rows,
    B_nodes = 24*sum_k p_k + 128*K, B_apps = 128*A + 128*A*N_grid + B_records,
    B_live,selected = 32*d**2 + q*u + B_nodes + B_apps, and the phase maximum
    of compact_pf_peak_bytes.
    B_records = 8*sum_k d_k + 8*N_grid + A*N_grid*R_pos, where R_pos is
    host_position_record_reserve's fixed metadata allowance plus the typed
    integer fields of the node record, independent step selection and route
    work. Fixed steps use their actual integer width. Budgeted candidates use
    the finite-binary64 8500-bit step-count envelope. The cached rational
    bound coefficients belong to _node_cache_bytes. The census arrays,
    including each node's restricted pair and triple tables, belong to the
    envelope that _lchs_census_choice admits: the shared census_bytes law
    plus 9*max(E, F) bytes for one node's row selection, with E pair rows
    and F reserved triple rows. These are logical payload and qualified
    native object allowances, not a process heap bound. The cached
    combined nodes (_node_cache_bytes) stay live through record selection
    and are held in every phase.

    Work, with d the padded dimension, N = d**2, K distinct nodes and M
    applications: W_coeff = W_dec + W_align + K*(2*m_L + 23*u + 8), the
    decomposition (_pauli_decomposition_requirements), the one-time
    alignment of the actual L/H supports and each distinct node's
    combination, pruning and streamed upper dropped mass
    (_coefficient_preparation_work), then the census p*q + G + K_n*V, the
    scalar selections 16*M*K, M*N + 4*M*K*d + 4*d and each selected action
    route (_action_route). Align the actual L/H support once in lexical
    order, then combine and prune it once per distinct k node. Selection
    charges the alignment and each reached coefficient operation.
    Application count affects step records and actions, not repeated
    coefficient preparation or table storage. With full support, K = 200
    and M = 9, W_coeff alone is 21,547,072, 86,790,720, 349,573,184 and
    1,407,991,872 units at q = 6 to 9, so full-support q = 8 is refused
    under max_select_work=100_000_000 while sparse supports may fit. These
    units are admission proxies, not timings or equal-cost CPU operations.
    """
    method, d = data.method, len(data.initial)
    specifications = ([(0.,1.,False)] if has_initial else [])
    specifications += [(float(t),float(w),True) for t,w in zip(source_nodes,source_weights,strict=True)]
    k_nodes = data.quadrature.k_nodes
    calls = len(specifications)*len(k_nodes)
    keys = tuple(_node_key(k) for k in k_nodes)
    distinct = tuple(dict.fromkeys(keys))
    decomposition_work, decomposition_bytes = _pauli_decomposition_requirements(d)
    # The admitted decomposition first; the alignment and node combinations
    # are admitted from the actual supports once they exist.
    selection_work = decomposition_work
    _check_bytes(decomposition_bytes, method.max_bytes, "LCHS Pauli selection arrays")
    if selection_work > method.max_select_work:
        raise ValueError(
            f"LCHS Pauli decomposition needs {selection_work} units of max_select_work; this "
            f"stage needs LCHS(max_select_work=...) to be at least {selection_work} "
            f"(now {method.max_select_work})")
    decomposition = _trotter_pauli_decomposition(data.quadrature.l_part,data.quadrature.h_part)
    q = decomposition.num_qubits
    m_l, m_h = len(decomposition.l_coefficients), len(decomposition.h_coefficients)
    union = len(set(decomposition.l_coefficients) | set(decomposition.h_coefficients))
    align_work, node_work = _coefficient_preparation_work(q, m_l, m_h, union, len(distinct))
    selection_work += align_work+node_work
    if selection_work > method.max_select_work:
        raise ValueError(
            f"LCHS PF coefficient preparation brings the charged max_select_work to {selection_work} "
            f"units, including the Pauli decomposition; this stage needs LCHS(max_select_work=...) "
            f"to be at least {selection_work} (now {method.max_select_work})")
    cache_bytes = _node_cache_bytes(len(distinct), union, q)
    # Before combination the kept and dropped counts are unknown, so the
    # node-table terms are admitted at their largest values (every union
    # row kept or dropped at every node).
    decomposition_output = 32*d*d+(q+16)*(m_l+m_h)
    # The cached five-field rows and their construction temporaries use at
    # most 256 + 128*u + 64*(m_L+m_H) bytes under the 64-bit CPython container
    # allowance. With q >= 1 and m_L+m_H <= 2*u, 128*(u+1)*(q+1) bounds it.
    # Label strings and coefficient objects are borrowed from the decomposition.
    alignment = 128*(union+1)*(q+1)
    grid = len(k_nodes)
    records_bytes = 8*grid + calls*host_position_record_reserve(
        union,
        budgeted=method.hamiltonian_evolution_backend == "trotter_error_budgeted",
        fixed_steps=method.trotter_steps,
        max_work=method.max_select_work,
    )
    envelope = compact_pf_peak_bytes(held_other=cache_bytes, decomposition_peak=decomposition_bytes,
        decomposition_output=decomposition_output, alignment=alignment,
        nodes=24*union*len(distinct)+128*len(distinct),
        applications=128*len(specifications)+128*calls+records_bytes+8*union*len(distinct),
        union_count=union, live_selected=0, max_rotations=0, action_scratch=0)
    _check_bytes(envelope, method.max_bytes, "LCHS Pauli node tables")
    combined_nodes = _distinct_combined_nodes(decomposition, k_nodes)
    table = _selected_node_table(decomposition, combined_nodes, k_nodes)
    kept = [len(node["union_indices"]) for node in table.nodes]
    nodes_bytes = 24*sum(kept)+128*len(kept)
    applications_bytes = (128*len(specifications)+128*calls+records_bytes
                          +8*sum(len(node["pruned_indices"]) for node in table.nodes))
    coefficient_phase = (decomposition_output+alignment+nodes_bytes+applications_bytes+24*union+128)
    budgeted = method.hamiltonian_evolution_backend == "trotter_error_budgeted"
    fixed_work = len(specifications)*d*d+4*calls*d+4*d
    # Scalar step selection from a cached coefficient, per application/node.
    scalar_work = 16*calls if budgeted else 0
    total = selection_work+scalar_work+fixed_work
    evaluations, ledger = {}, dict(work=0, pair_checks=0, nested_checks=0)
    if budgeted:
        census_keys = tuple(key for key in distinct if combined_nodes[key].traceless_terms
                            and any(elapsed-start for start,_,_ in specifications))
        evaluations, ledger = _node_bound_coefficients(
            _union_nonidentity_labels(decomposition), q,
            {key: combined_nodes[key].traceless_terms for key in census_keys}, census_keys,
            order=method.trotter_order, max_work=method.max_select_work, max_bytes=method.max_bytes,
            census_held=cache_bytes+coefficient_phase, work_used=total)
        total += ledger["work"]
    applications = []
    dense_products = vector_rotations = dense_rotations = 0
    commutators = ledger["pair_checks"]+ledger["nested_checks"]
    used = set()
    for start,weight,source in specifications:
        positions = []
        duration = elapsed-start
        for key in keys:
            combined = combined_nodes[key]
            terms = combined.traceless_terms
            if budgeted and duration != 0:
                record = _budgeted_trotter_node_record(combined, _first_use(evaluations, key, used),
                    final_time=duration, error_budget=_lchs_trotter_error_budget_per_node(method),
                    order=method.trotter_order)
            elif budgeted:
                record = _LCHSTrotterNodeRecord(pf_bound_value=0.0,pruned_l1_mass=combined.pruned_l1_mass,
                    combined_bound_value=0.0,step_count=0,selection=None,bound_value_status="not_applicable")
            else:
                record = _LCHSTrotterNodeRecord(pf_bound_value=None,pruned_l1_mass=combined.pruned_l1_mass,
                    combined_bound_value=None,step_count=method.trotter_steps if terms else 0,selection=None)
            if record.step_count > method.max_trotter_steps:
                raise ValueError("selected product-formula steps exceed max_trotter_steps")
            positions.append(dict(record=record))
        applications.append(dict(start=start,weight=weight,source=source,positions=tuple(positions)))
    selection_work += scalar_work+ledger["work"]
    live_selected = 32*d*d+q*union+nodes_bytes+applications_bytes
    rotations = [_rotation_count(count, method.trotter_order) for count in kept]
    action_scratch = 0
    # Select against the actual whole retained population, not a prefix whose
    # later records could invalidate a previously admitted dense route.
    for application in applications:
        for index, position in zip(table.grid_to_node, application["positions"], strict=True):
            r,R = position["record"].step_count,rotations[index]
            route,work,peak,products = _action_route(d,r,R,
                max_work=method.max_select_work-total,max_bytes=method.max_bytes,
                retained_bytes=cache_bytes+live_selected+16*max(rotations, default=0))
            position.update(route=route,work=work)
            total += work
            action_scratch = max(action_scratch,peak-cache_bytes-live_selected-16*max(rotations, default=0))
            dense_products += products
            vector_rotations += r*R if route=="vector" else 0
            dense_rotations += R if route=="dense_power" else 0
    workspace = compact_pf_peak_bytes(held_other=cache_bytes, decomposition_peak=decomposition_bytes,
        decomposition_output=decomposition_output, alignment=alignment, nodes=nodes_bytes,
        applications=applications_bytes, union_count=union, live_selected=live_selected,
        max_rotations=max(rotations, default=0), action_scratch=action_scratch,
        other_phase_peak=ledger.get("bytes", 0))
    _check_bytes(workspace,method.max_bytes,"LCHS selected action arrays and tables")
    if total > method.max_select_work:
        raise ValueError("selected classical LCHS work exceeds max_select_work")
    counts = dict(dimension=d,operator_applications=len(specifications),evolution_calls=calls,
        pauli_decompositions=2,matrix_power_products=dense_products,vector_rotations=vector_rotations,
        dense_step_rotations=dense_rotations,commutator_checks=commutators,
        combined_nodes=len(distinct),census_structures=ledger.get("structures",0),
        census_contractions=ledger.get("contractions",0),
        selection_work=selection_work,size_units=total)
    return dict(nodes=table, applications=tuple(applications)),counts,workspace


def application_actions(table, application, *, elapsed, order):
    """Yield (position, sequence, phase, steps, route) of one stored application, in position order.

    Each position's rotation sequence is formed from its node's stored
    coefficient rows at this application's elapsed time and saved
    max(1, steps) (_rotation_sequence), once per current application and
    position, and released by the caller after use. The identity phase is
    -elapsed*c_I with the saved generator coefficient c_I.
    """
    duration = elapsed-application["start"]
    for position, (index, stored) in enumerate(zip(table.grid_to_node, application["positions"], strict=True)):
        node = table.nodes[int(index)]
        steps = stored["record"].step_count
        sequence = _rotation_sequence(_BorrowedRows(table.labels, node), duration, max(1, steps), order)
        yield position, sequence, -duration*node["identity_coefficient"], steps, stored["route"]


def apply_node(sequence, phase, steps, route, vector):
    """Apply the node action exp(i*phase)*S**r to vector, with phase = -t*c_I.

    S is one product-formula step, the product of exp(-i*a*P) over the
    (P, a) sequence with the first entry applied first, and r = max(1, steps)
    is the saved step count. Each factor is cos(a)*I - i*sin(a)*P because
    P**2 = I. The dense route builds S by multiplying the current matrix by
    P from the left with the bit masks from _pauli_masks. With Y = i*X*Z,
    P|j> = i**n_Y * (-1)**popcount(j & z) |j ^ x>, where x marks X or Y
    positions, z marks Y or Z positions and n_Y counts Y. The route saved
    at selection is used. Nothing is reselected here.
    """
    d = len(vector)
    steps = max(1,steps)
    if route == "vector":
        result = np.array(vector,dtype=complex)
        scratch = np.empty_like(result)
        for _ in range(steps):
            for label,angle in sequence:
                apply_terms(((label,1.),),result,num_qubits=d.bit_length()-1,out=scratch)
                result *= cos(angle)
                result += (-1j*sin(angle))*scratch
    else:
        step = np.eye(d,dtype=complex)
        indices = np.arange(d,dtype=np.intp)
        for label,angle in sequence:
            x,z,y = _pauli_masks(label)
            signs = 1-2*((np.bitwise_count(indices & z)&1).astype(np.int8))
            transformed = np.empty_like(step)
            transformed[indices ^ x] = ((1j**y)*signs)[:,None]*step
            step = cos(angle)*step-1j*sin(angle)*transformed
        result = np.linalg.matrix_power(step,steps) @ vector
    return np.exp(1j*phase)*result
