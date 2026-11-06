"""The saved data of an LCHS Plan and its exact physical application schedule."""

from nwqlib._limits import exceeds
from dataclasses import dataclass
from math import isfinite

import numpy as np

from nwqlib.blocks.selection import SelectedBlock
from nwqlib.core.records import Float64
from nwqlib.operators.access import _check_bytes
from .references import dense_work
from .solution_error_budget import application_weight, psd_recovery, psd_recovery_exponent, psd_recovery_part


@dataclass(frozen=True)
class GridApplication:
    """One physical application of the saved finite sum.

    Attributes:
        start: Time at which the input enters.
        elapsed: Evolution time T - start.
        weight: Duhamel quadrature weight, or one for the initial state.
        vector: Padded physical input vector.
        steps: Saved product-formula step count per k-node, empty for dense nodes.
        source: Whether the input is the constant source.
        input_norm: Physical norm of the input, or None when not representable.
        slots: SELECT address slots of this application in a quantum layout.
        omitted: Set of k-node indices of this application whose nonzero
            slots planning omitted under ``source_drop_allowance``, so its
            circuit never applies them.
    """

    start: float
    elapsed: float
    weight: float
    vector: object
    steps: tuple[int, ...]
    source: bool = False
    input_norm: float | None = None
    slots: tuple[int, ...] = ()
    omitted: frozenset[int] = frozenset()


@dataclass(frozen=True)
class SelectedGrid:
    """Saved L, H, k-nodes and coefficients with the Result's application schedule.

    Every field is read from the LCHSData that the Plan's own selected block
    or kernel carries (selected_payload), and a classical Result's
    application records are checked against the Plan's schedule
    (host_applications), so references and refinement evaluate the finite
    sum that was executed.
    """

    l_part: object
    h_part: object
    nodes: object
    coefficients: object
    shift: float
    applications: tuple[GridApplication, ...]
    payload: object


def selected_payload(plan):
    """Return the Plan's original numerical payload, matched without hashing or reconstructing it."""
    from .selection import LCHSData
    from .periodic import _PeriodicPayload

    data = plan._native.get("native_data")
    if type(data) not in {LCHSData, _PeriodicPayload}:
        raise ValueError("this LCHS Plan has no saved selected numerical data")
    if plan.execution == "quantum":
        selected = {record.content_id:record for record in plan.construction.selections}
        if type(data) is LCHSData and data.selected_select == "coherent_lcu":
            # The coherent LCU body carries the shared payload itself.
            matches = [block for block in plan.blocks if type(block) is SelectedBlock
                       and block.record.signature.target.name == "lchs.coherent_lcu"]
            if (len(matches) != 1 or selected.get(matches[0].record.content_id) != matches[0].record
                    or matches[0]._payload is not data):
                raise ValueError("selected-grid requires the exact coherent LCU block and shared numerical payload")
        elif type(data) is LCHSData:
            matches = [block for block in plan.blocks if type(block) is SelectedBlock
                       and block.record.signature.target.name == "lchs.select"]
            if (len(matches) != 1 or selected.get(matches[0].record.content_id) != matches[0].record
                    or type(matches[0]._payload) is not tuple or len(matches[0]._payload) != 2
                    or matches[0]._payload[0] is not data or matches[0]._payload[1] != plan.problem.elapsed_time):
                raise ValueError("selected-grid requires the exact SELECT leaf and shared numerical payload")
        else:
            matches = [block for block in plan.blocks if type(block) is SelectedBlock and block._payload is data]
            if (len(matches) != 2 or any(selected.get(block.record.content_id) != block.record
                    or block.record.semantics.input.identity != data.parameter_id for block in matches)):
                raise ValueError("periodic refinement requires its exact selected phase and Strang payload")
    else:
        from nwqlib.blocks.kernels import BoundKernel
        if (type(data) is not LCHSData or len(plan.blocks) != 1 or type(plan.blocks[0]) is not BoundKernel
                or plan.blocks[0].plan_id != plan.content_id or plan.blocks[0].record != plan.construction.kernels[0]):
            raise ValueError("selected-grid requires the original bound classical kernel")
    return data


def _specifications(plan, data):
    """List the physical applications in the order the solve recorded them.

    Each entry is (is_source, start, elapsed, weight, padded input vector,
    input norm). The initial state comes first when it is nonzero, then one
    entry per Duhamel node when the source is nonzero. Only the dense recipe
    has padded input vectors.
    """
    rec = plan.reconstruction
    dense = data.dense_inputs()
    specs = []
    if rec.initial_scale.mantissa != 0:
        specs.append((False, 0., rec.elapsed_time, 1., dense.initial, rec.initial_scale.as_float()))
    if dense.source is not None and rec.source_scale.mantissa != 0:
        specs.extend((True, start, rec.elapsed_time-start, weight, dense.source, rec.source_scale.as_float())
                     for start, weight in zip(rec.source_nodes, rec.source_weights, strict=True))
    return specs


def host_applications(plan, result, data):
    """Check a classical Result's application records against the Plan and return its grid applications.

    Each record must name the Plan's application at its position, with the
    same start and elapsed time, source role, weight and recovery scale, so
    source roles are validated before any reference or component
    propagation. The weight is formed again from the Plan's saved weight
    and input PhysicalScale (``solution_error_budget.application_weight``):
    a record must bind that value when binary64 represents it and must
    omit the weight otherwise. Per-node product-formula step counts come from the Plan's
    classical actions, the choice that execution applied. The records do not
    repeat them.
    """
    rec = plan.reconstruction
    specs = _specifications(plan, data)
    if len(result.applications) != len(specs):
        raise ValueError("selected-grid application count differs from its saved source schedule")
    pf = plan.method.hamiltonian_evolution_backend in {"trotter", "trotter_error_budgeted"}
    actions = data.host_actions["applications"] if pf else None
    applications = []
    for index, (application, spec) in enumerate(zip(result.applications, specs, strict=True)):
        source, start, elapsed, weight, vector, norm = spec
        args = {entry.parameter:entry.value.value if isinstance(entry.value, Float64) else entry.value
                for entry in application.arguments}
        if (application.name != f"application_{index}" or application.implementation != plan.construction.kernels[0].implementation
                or args.get("start_time") != start or args.get("elapsed_time") != elapsed
                or args.get("source_application") != int(source)
                or args.get("weight") != application_weight(weight, rec.source_scale if source else rec.initial_scale)
                or args.get("recovery_scale") != psd_recovery(rec.psd_shift, elapsed)):
            raise ValueError("selected-grid application differs from the recorded physical schedule")
        steps = tuple(position["record"].step_count for position in actions[index]["positions"]) if pf else ()
        applications.append(GridApplication(start, elapsed, weight, vector, steps, source, norm))
    return tuple(applications)


def selected_grid(plan, result, *, checks):
    """Return the saved L, H, nodes and coefficients with the applications and nodes the Plan executes, without spectral or quadrature selection."""
    from .selection import LCHSData
    data = selected_payload(plan)
    if type(data) is not LCHSData or data.dense is None:
        raise ValueError("selected-grid vector references require saved dense LCHS data; "
                         "no periodic or compact densification")
    rec, quad = plan.reconstruction, data.quadrature
    nodes, coefficients = quad.k_nodes, quad.coefficients
    n = len(nodes)
    specs = _specifications(plan, data)
    pf = plan.method.hamiltonian_evolution_backend in {"trotter", "trotter_error_budgeted"}
    # Planning can omit, under source_drop_allowance, an application or some
    # nonzero slots of one whose PREP amplitude underflows. The grid lists
    # only what the circuit applies, so the reference and the refined bounds
    # leave out the omitted part that dropped_bound reports. The schedule
    # bytes are checked from the kept application count before any slot is
    # read, and the node limit from the applied nodes before any schedule is
    # built. Only a layout with an omission reads its slot amplitudes.
    coherent = plan.execution == "quantum" and data.selected_select == "coherent_lcu"
    layout = data.source_layout if plan.execution == "quantum" and not coherent else None
    dropped, omitted_nodes = set(), None
    if coherent:
        if data.qsp_plan.candidate.alternative != "lchs":
            raise ValueError("the selected imaginary-time filter or commuting product evolves exp(-t A) without a "
                             "kernel grid, so a selected-grid reference does not describe its finite sum; compare "
                             "with the 'expm' or 'closed_form' reference instead")
        # The coherent sum applies every kernel node to every kept
        # application (coherent_selection). omitted_applications indexes the
        # applications with a nonzero input, in the order of specs.
        dropped = set(data.qsp_plan.preparation.omitted_applications)
        count = len(specs)-len(dropped)
    else:
        count = len(specs) if layout is None else len(layout.applications)
    _check_bytes(128*count*(n+1), checks.max_bytes, "selected reference schedule")
    evaluations = n*count
    if layout is not None and layout.dropped_bound is not None:
        nonzero = np.asarray(coefficients) != 0
        omitted_nodes = [frozenset(np.flatnonzero((layout.amplitudes[list(slots)] == 0) & nonzero).tolist())
                         for slots in layout.application_slots]
        evaluations -= sum(map(len, omitted_nodes))
    if exceeds(evaluations, checks.max_node_evaluations):
        raise ValueError("selected grid exceeds max_node_evaluations before reference work")
    if coherent:
        applications = [GridApplication(start, elapsed, weight, vector, (), source, norm)
                        for index, (source, start, elapsed, weight, vector, norm) in enumerate(specs)
                        if index not in dropped]
    elif plan.execution == "quantum":
        # The planned source layout maps each kept application, identified by
        # its role and start time, to its SELECT address slots. Without a
        # source the homogeneous application uses the first n slots.
        select = None if not pf else data.select_data.plan
        if layout is None:
            kept = [(specs[0], tuple(range(n)))]
        else:
            planned = {spec[:2]: spec for spec in specs}
            kept = [(planned[item.kind == "source", item.start_time], slots)
                    for item, slots in zip(layout.applications, layout.application_slots, strict=True)]
        applications = []
        for index, ((source, start, elapsed, weight, vector, norm), slots) in enumerate(kept):
            steps = () if select is None else tuple(select.branch_step_counts[slot] for slot in slots)
            omitted = frozenset() if omitted_nodes is None else omitted_nodes[index]
            applications.append(GridApplication(start, elapsed, weight, vector, steps, source, norm, slots, omitted))
    else:
        applications = host_applications(plan, result, data)
    return SelectedGrid(data.dense.l_part, data.dense.h_part, nodes, coefficients, rec.psd_shift, tuple(applications), data)


def grid_arguments(grid):
    """Return the (name, value) arguments that record the reproduced schedule.

    They list the node and application counts, the PSD shift, and for each
    application its start, elapsed time, weight, source flag, per-node steps
    and, when planning omitted some of its nodes, their number.
    """
    arguments = [("k_nodes", len(grid.nodes)), ("applications", len(grid.applications)), ("psd_shift", grid.shift)]
    for index, app in enumerate(grid.applications):
        arguments.extend(((f"application_{index}_start", app.start), (f"application_{index}_elapsed", app.elapsed),
                          (f"application_{index}_weight", app.weight), (f"application_{index}_source", int(app.source))))
        arguments.extend((f"application_{index}_node_{node}_steps", step) for node, step in enumerate(app.steps))
        if app.omitted:
            arguments.append((f"application_{index}_omitted_nodes", len(app.omitted)))
    return arguments


def selected_pf_vector_requirements(circuit, dimension):
    """Return (work, gates, bytes) of applying one decomposed product-formula branch to a D-vector.

    The circuit is the branch after its one requested decompose(), with the
    saved identity phase attached, and Statevector.evolve applies it gate by
    gate. A gate on k qubits forms its 4**k-entry matrix, contracts it with
    the vector in D*2**k complex multiply-adds and reorders the vector, up to
    2*D copies. A nonzero global phase multiplies the D entries, and the
    weighted accumulation ``reference += (scale*coefficient)*evolved`` costs
    2*D + 1. The known arrays are six complex D-vectors and the largest gate
    matrix, 96*D + 16*max 4**k bytes. These units are proxies for the limit
    checks, not timings.
    """
    from qiskit.circuit import Gate
    if dimension != 1 << circuit.num_qubits:
        raise ValueError("PF reference dimension differs from its circuit")
    if circuit.num_clbits or circuit.parameters:
        raise ValueError("PF reference needs a bound unitary circuit")
    work = 2*dimension + 1 + dimension*int(circuit.global_phase != 0)
    largest = 0
    for instruction in circuit.data:
        if not isinstance(instruction.operation, Gate):
            raise ValueError("PF reference applies unitary gates only")
        r = 1 << instruction.operation.num_qubits
        work += dimension*r + r*r + 2*dimension
        largest = max(largest, r*r)
    return work, len(circuit.data), 96*dimension + 16*largest


def grid_reference(plan, grid, *, checks, counts, held_inputs=()):
    """Accumulate the planned finite sum term by term on the padded system, then project once.

    ACL arXiv:2312.03916v2 Eqs.(2),(60)-(61): each application contributes
    weight*exp(shift*elapsed)*sum_j c_j U(elapsed,k_j)*physical_input.
    Coefficients here are the original unshifted quadrature c_j, so PSD
    compensation is applied once per application, including every source node.
    When exp(shift*T) exceeds binary64, the sum holds each compensation
    divided by 2**e (solution_error_budget.psd_recovery_exponent) and 2**e is
    applied to the projected vector at the end. The reference is None when
    that vector is not representable.

    A classical product-formula reference reads the coefficient table at
    most once per distinct k-node with a nonzero coefficient
    (time_independent_terms.stored_pf_nodes). Reading a row in place costs
    one logical visit, and reading the identity coefficient and upper
    dropped mass costs two visits per node (stored_node_read_work), counted
    in the cumulative dense_work with the stored_node_read_work count as a
    diagnostic subset. The table is counted in the memory held by the phase
    that reads it. No Pauli decomposition runs here. Vector actions and
    their phase and accumulation are counted separately.

    A classical product-formula reference applies each saved node action
    with the saved route and step count (host_pf.apply_node), its rotations
    formed from the saved coefficients at the application's elapsed time.
    The route's work (host_pf._route_requirements), the phase path (D + 4:
    negation and the identity-coefficient product, the imaginary argument
    and its exponential, and D products) and the weighted accumulation
    (2*D + 1, host_pf.host_phase_and_accumulation_work) are checked against
    this call's limits before it runs. A saved route that does not fit is
    refused, never replaced by another route.

    A product-formula reference holds, while each saved node action or
    branch runs, the dominant arrays of the Plan and of the caller: dense L
    and H (32*D**2 bytes), the k nodes and coefficients, the application
    schedule (128 bytes per application and node, as selected_grid checks),
    the saved coefficient table of the classical route
    (time_independent_terms._stored_node_table_bytes) or the occurrence-angle
    table of the quantum SELECT with 64 bytes per address for its per-address
    phases and maps, and the caller's input arrays (held_inputs). This is a
    coarse estimate of the dominant allocations, not of Python objects or
    process memory. A classical action adds its rotation buffer, saved route
    scratch and weighted-accumulation reserve.

    The quantum selected_grid reference applies each planned decomposed
    product-formula branch directly to the input vector of its operator
    application, with the work and bytes of selected_pf_vector_requirements.
    Before construction, Qiskit's decompose holds several copies of the
    branch's gates, which its envelope bound covers below.
    """
    from nwqlib.problems.inputs import PhysicalScale
    from .time_independent_terms import (_spectral_host_requirements, spectral_lchs_sum,
        stored_pf_nodes, stored_node_read_work, _stored_node_table_bytes, _BorrowedRows)
    exponent = psd_recovery_exponent(grid.shift, plan.reconstruction.elapsed_time)
    d = plan.reconstruction.encoded_dimension
    pf = plan.method.hamiltonian_evolution_backend in {"trotter", "trotter_error_budgeted"}
    held = 0
    if pf:
        held = (32*d*d + 32*len(grid.nodes) + 128*len(grid.applications)*(len(grid.nodes)+1)
                + sum(array.nbytes for array in held_inputs if array is not None))
        if plan.execution == "classical":
            table = stored_pf_nodes(grid.payload)
            held += _stored_node_table_bytes(table)
        else:
            select = grid.payload.select_data.plan
            held += select.occurrence_angle_tables.array.nbytes + 64*select.padded_node_count
        dense_work(checks, counts, work=0, peak_bytes=held)
    if pf and plan.execution == "classical":
        active = any(app.weight != 0 and app.input_norm != 0 for app in grid.applications)
        read = sorted({
            int(table.grid_to_node[node])
            for node, coefficient in enumerate(grid.coefficients)
            if coefficient != 0
        }) if active else []
        read_work = stored_node_read_work(
            len(table.nodes[index]["union_indices"]) for index in read
        )
        dense_work(checks, counts, work=read_work, peak_bytes=held)
        counts["stored_node_read_work"] += read_work
    setup_peak = held+80*d if pf else held+16*(2*d*d+5*d)
    dense_work(checks, counts, work=d, peak_bytes=setup_peak)
    reference = np.zeros(d, dtype=complex)
    if not pf:
        # The dense exact reference applies every active node evolution to the
        # active application vectors from one Hermitian eigensystem per node
        # (spectral_lchs_sum), admitted before the first eigensystem with its
        # law. Bytes: the phase bytes of the streamed sum, the reference
        # vector and four d-vectors, beside the live L and H.
        # Applications that apply the same k nodes share one streamed sum.
        # Only an omission of some nodes of an application under
        # source_drop_allowance makes another group, with its own eigensystems.
        groups = {}
        for app in grid.applications:
            if app.weight != 0 and app.input_norm != 0:
                scale = app.weight*psd_recovery_part(grid.shift, app.elapsed, exponent)
                if not isfinite(scale):
                    raise ValueError("selected-grid PSD recovery is nonfinite")
                groups.setdefault(app.omitted, []).append((app.elapsed, scale, app.vector))
        sums, work, phase_bytes = [], 0, 0
        for omitted, specifications in (groups or {frozenset(): []}).items():
            nodes = [(k, coefficient) for j, (k, coefficient) in enumerate(zip(grid.nodes, grid.coefficients, strict=True))
                     if coefficient != 0 and j not in omitted]
            identity = (not (np.any(grid.l_part) or np.any(grid.h_part))
                        or not any(elapsed for elapsed, _, _ in specifications))
            eigensystems = 0 if identity or not nodes else (1 if not np.any(grid.l_part) else len(nodes))
            inputs = len({id(vector) for _, _, vector in specifications})
            group_work, group_bytes = _spectral_host_requirements(d, nodes=len(nodes),
                applications=len(specifications), inputs=inputs, eigensystems=eigensystems)
            work, phase_bytes = work+group_work, max(phase_bytes, group_bytes)
            sums.append((nodes, specifications))
        dense_work(checks, counts, work=work, peak_bytes=phase_bytes+80*d)
        for nodes, specifications in sums:
            if specifications and nodes:
                spectral = {"eigh_calls": 0, "matvecs": 0}
                reference += spectral_lchs_sum(grid.l_part, grid.h_part, [k for k, _ in nodes],
                    [coefficient for _, coefficient in nodes], tuple(specifications), counts=spectral)
                counts["eigh_calls"] += spectral["eigh_calls"]
                counts["matvec_attempts"] += spectral["matvecs"]
                counts["matvec_completed"] += spectral["matvecs"]
    order = plan.method.trotter_order
    q = d.bit_length()-1
    stored = grid.payload.host_actions["applications"] if pf and plan.execution == "classical" else None
    for index, app in enumerate(grid.applications if pf else ()):
        if app.weight == 0 or app.input_norm == 0:
            continue
        scale = app.weight*psd_recovery_part(grid.shift, app.elapsed, exponent)
        if not isfinite(scale):
            raise ValueError("selected-grid PSD recovery is nonfinite")
        for node, coefficient in enumerate(grid.coefficients):
            if coefficient == 0 or node in app.omitted:
                continue
            steps = app.steps[node]
            if plan.execution == "classical":
                from .host_pf import (_route_requirements, _rotation_count, _rotation_sequence, apply_node,
                                      host_phase_and_accumulation_work)
                table_node = table.nodes[int(table.grid_to_node[node])]
                rows = _BorrowedRows(table.labels, table_node)
                route = stored[index]["positions"][node]["route"]
                rotations = _rotation_count(len(rows), order)
                total = counts["pf_operations"]+max(1, steps)*rotations
                if exceeds(total, checks.max_pf_operations):
                    raise ValueError("selected-grid reference exceeds max_pf_operations before its node action")
                work, scratch, _ = _route_requirements(d, steps, rotations, route)
                dense_work(
                    checks, counts,
                    work=work+host_phase_and_accumulation_work(d),
                    peak_bytes=held+16*rotations+scratch+16*d,
                )
                counts["pf_operations"] = total
                counts["fixed_pf_attempts"] += 1
                sequence = _rotation_sequence(rows, app.elapsed, max(1, steps), order)
                evolved = apply_node(sequence, -app.elapsed*table_node["identity_coefficient"],
                                     steps, route, app.vector)
                del sequence
            else:
                from qiskit.quantum_info import Statevector
                from .select_synthesis import _build_product_formula_branch
                from .solution_error_budget import unit_angle
                selected = grid.payload.select_data.plan
                # Preconstruction envelope: an occurrence with support s >= 2
                # uses at most 2*(s-1) parity CX, 2*s basis changes and one
                # rotation, at most 4*s - 1 elementary gates, and support one
                # one rotation, so max(1, 4*q - 1)*W*R gates bound a branch
                # of W occurrences per step over R saved repetitions. Bytes:
                # 1024 per gate, because Qiskit's decompose holds the
                # evolution gate's synthesized definition, its DAG, the
                # substituted DAG and the decomposed circuit at once, each
                # allowed 256 bytes per gate as in
                # _dense_synthesis.gatewise_control_size (a 20,000-step
                # two-qubit branch of 1,380,000 decomposed gates raised peak
                # resident memory by 804 bytes per gate with Qiskit 2.6.0),
                # plus the known arrays and one gate matrix of the widest
                # priced gate, 16*4**2 bytes.
                envelope = max(1, 4*q-1)*len(selected.occurrence_schedule)*sum(
                    selected.occurrence_block_repetitions)
                if exceeds(counts["pf_operations"]+envelope, checks.max_pf_operations):
                    raise ValueError("selected-grid reference exceeds max_pf_operations before circuit construction")
                dense_work(
                    checks, counts, work=0,
                    peak_bytes=held+1024*envelope+96*d+256,
                )
                circuit = _build_product_formula_branch(selected, app.slots[node], num_system_qubits=q)
                circuit.global_phase += unit_angle(selected.identity_phases[app.slots[node]])
                decomposed = circuit.decompose()
                del circuit
                work, gates, local_bytes = selected_pf_vector_requirements(decomposed, d)
                if exceeds(counts["pf_operations"]+gates, checks.max_pf_operations):
                    raise ValueError("selected-grid reference exceeds max_pf_operations before its vector action")
                dense_work(checks, counts, work=work, peak_bytes=held+local_bytes+64*gates)
                counts["pf_operations"] += gates
                counts["fixed_pf_attempts"] += 1
                evolved = Statevector(app.vector).evolve(decomposed).data
                del decomposed
            counts["fixed_pf_completed"] += 1
            reference += (scale*coefficient)*evolved
    reference = reference[:plan.problem.dimension]
    if exponent == 0:
        return reference
    return PhysicalScale(mantissa=0.5, exponent=exponent+1).apply_vector(np.ascontiguousarray(reference))
