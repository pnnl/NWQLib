"""Existing selected LCHS data and its exact physical application schedule."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from math import isfinite

import numpy as np

from nwqlib.blocks.selection import SelectedBlock
from nwqlib.core.records import Float64, FrozenArray
from nwqlib.operators.access import _check_bytes
from .references import dense_work
from .solution_error_budget import psd_recovery, psd_recovery_exponent, psd_recovery_part


@dataclass(frozen=True)
class GridApplication:
    """One physical application of the saved finite recipe.

    Attributes:
        start: Time at which the input enters.
        elapsed: Evolution time T - start.
        weight: Duhamel quadrature weight, or one for the initial state.
        vector: Padded physical input vector.
        steps: Saved product-formula step count per k-node, empty for dense nodes.
        source: Whether the input is the constant source.
        input_norm: Physical norm of the input, or None when not representable.
        slots: SELECT address slots of this application in a quantum layout.
    """

    start: float
    elapsed: float
    weight: float
    vector: object
    steps: tuple[int, ...]
    source: bool = False
    input_norm: float | None = None
    slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class SelectedGrid:
    """Saved L, H, k-nodes and coefficients with the Result's application schedule.

    Every field is read from the Plan's own LCHSData and checked against its
    reconstruction, so references and refinement evaluate the recipe that
    was executed.
    """

    l_part: object
    h_part: object
    nodes: object
    coefficients: object
    shift: float
    applications: tuple[GridApplication, ...]
    payload: object


def selected_payload(plan):
    """Match the original selected native data without hashing or reconstructing it."""
    from .selection import LCHSData
    from .periodic import _PeriodicPayload

    data = plan._native.get("native_data")
    if type(data) not in {LCHSData, _PeriodicPayload}:
        raise ValueError("this LCHS Plan has no saved selected numerical data")
    if plan.execution == "quantum":
        selected = {record.content_id:record for record in plan.construction.selections}
        if type(data) is LCHSData:
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
    if type(data) is LCHSData:
        d = plan.reconstruction.encoded_dimension
        if (data.quadrature.l_part.shape != (d,d)
                or data.quadrature.h_part.shape != (d,d) or data.initial.shape != (d,)
                or (data.source is None) != (plan.problem.source is None)
                or data.source is not None and data.source.shape != (d,)
                or data.method != plan.method):
            raise ValueError("selected numerical data differs from its encoded dimensions or Method")
    return data


def _specifications(plan, data):
    """List the physical applications in the order the solve recorded them.

    Each entry is (is_source, start, elapsed, weight, padded input vector,
    input norm). The initial state comes first when it is nonzero, then one
    entry per Duhamel node when the source is nonzero.
    """
    rec = plan.reconstruction
    specs = []
    if rec.initial_scale.mantissa != 0:
        specs.append((False, 0., rec.elapsed_time, 1., data.initial, rec.initial_scale.as_float()))
    if data.source is not None and rec.source_scale.mantissa != 0:
        specs.extend((True, start, rec.elapsed_time-start, weight, data.source, rec.source_scale.as_float())
                     for start, weight in zip(rec.source_nodes, rec.source_weights, strict=True))
    return specs


def host_applications(plan, result, data):
    """Check a classical Result's receipts against the Plan and return its grid applications.

    Each receipt must name the Plan's application at its position, with the
    same start and elapsed time, source role, weight and recovery scale, so
    source roles are validated before any reference or component
    propagation. Per-node product-formula step counts come from the Plan's
    host actions, the selection that execution applied. The receipts do not
    repeat them.
    """
    rec, n = plan.reconstruction, len(data.quadrature.k_nodes)
    specs = _specifications(plan, data)
    if len(result.applications) != len(specs):
        raise ValueError("selected-grid application count differs from its saved source schedule")
    pf = plan.method.hamiltonian_evolution_backend in {"trotter", "trotter_error_budgeted"}
    actions = data.host_actions["applications"] if pf and data.host_actions is not None else None
    if pf and (actions is None or len(actions) != len(specs)
               or any(len(action["positions"]) != n for action in actions)):
        raise ValueError("selected-grid PF schedule needs the Plan's host action for every application and node")
    applications = []
    for index, (application, spec) in enumerate(zip(result.applications, specs, strict=True)):
        source, start, elapsed, weight, vector, norm = spec
        args = {entry.parameter:entry.value.value if isinstance(entry.value, Float64) else entry.value
                for entry in application.arguments}
        if (application.name != f"application_{index}" or application.implementation != plan.construction.kernels[0].implementation
                or args.get("start_time") != start or args.get("elapsed_time") != elapsed
                or args.get("source_application") != int(source) or args.get("weight") != abs(weight)*norm
                or args.get("recovery_scale") != psd_recovery(rec.psd_shift, elapsed)):
            raise ValueError("selected-grid application differs from the recorded physical schedule")
        steps = tuple(position["record"].step_count for position in actions[index]["positions"]) if pf else ()
        applications.append(GridApplication(start, elapsed, weight, vector, steps, source, norm))
    return tuple(applications)


def selected_grid(plan, result, *, checks):
    """Use stored L/H, nodes and coefficients; no spectral/quadrature selection."""
    from .selection import LCHSData
    data = selected_payload(plan)
    if type(data) is not LCHSData:
        raise ValueError("selected-grid vector references require saved dense LCHS data; no periodic densification")
    rec, quad = plan.reconstruction, data.quadrature
    nodes, coefficients = quad.k_nodes, quad.coefficients
    n = len(nodes)
    if (len(coefficients) != n or rec.psd_shift != quad.conversion["psd_shift"]
            or rec.l_norm != quad.l_norm):
        raise ValueError("selected grid differs from the original quadrature/PSD recovery")
    specs = _specifications(plan, data)
    if n*len(specs) > checks.max_node_evaluations:
        raise ValueError("selected grid exceeds max_node_evaluations before reference work")
    _check_bytes(128*len(specs)*(n+1), checks.max_bytes, "selected reference schedule")
    pf = plan.method.hamiltonian_evolution_backend in {"trotter", "trotter_error_budgeted"}
    applications = []
    if plan.execution == "quantum":
        select = None if not pf else data.select_data.plan
        if pf and tuple(select.branch_step_counts) != rec.step_counts:
            raise ValueError("selected-grid PF counts differ from the actual selected schedule")
        layout = data.source_layout
        if layout is None:
            if data.source is not None or len(specs) != 1:
                raise ValueError("selected-grid source layout is unavailable")
            slots_by_app = (tuple(range(n)),)
        else:
            if (layout.source_nodes != rec.source_nodes or layout.source_weights != rec.source_weights
                    or len(layout.applications) != len(specs)):
                raise ValueError("selected-grid source layout differs from its original schedule")
            slots_by_app = tuple(row[3] for row in layout.applications)
        for index, (spec, slots) in enumerate(zip(specs, slots_by_app, strict=True)):
            source, start, elapsed, weight, vector, norm = spec
            if layout is not None:
                row = layout.applications[index]
                if (row[:3] != (start, elapsed, abs(weight)*norm) or len(slots) != n
                        or any(layout.branch_to_node[slot] is None
                            or layout.branch_kinds[layout.branch_to_node[slot]] != ("source" if source else "initial")
                            for slot in slots)):
                    raise ValueError("selected-grid source application differs from its physical input role")
            steps = () if select is None else tuple(select.branch_step_counts[slot] for slot in slots)
            applications.append(GridApplication(start, elapsed, weight, vector, steps, source, norm, slots))
    else:
        applications = host_applications(plan, result, data)
    return SelectedGrid(quad.l_part, quad.h_part, nodes, coefficients, rec.psd_shift, tuple(applications), data)


def grid_arguments(grid):
    """Return the (name, value) arguments that record the reproduced schedule.

    They list the node and application counts, the PSD shift, and for each
    application its start, elapsed time, weight, source flag and per-node steps.
    """
    arguments = [("k_nodes", len(grid.nodes)), ("applications", len(grid.applications)), ("psd_shift", grid.shift)]
    for index, app in enumerate(grid.applications):
        arguments.extend(((f"application_{index}_start", app.start), (f"application_{index}_elapsed", app.elapsed),
                          (f"application_{index}_weight", app.weight), (f"application_{index}_source", int(app.source))))
        arguments.extend((f"application_{index}_node_{node}_steps", step) for node, step in enumerate(app.steps))
    return arguments


def selected_pf_vector_requirements(circuit, dimension):
    """Charge a realized elementary PF branch, without calling to_matrix.

    The circuit is the branch after its one requested decompose(), with
    the saved identity phase already attached. Work uses complex
    multiply-add units plus scalar/entry visits. scalar_work provides the
    alternative unfused scalar count. Neither total prices opaque SDK CPU
    bookkeeping or process RSS.

    For gate j acting on k_j qubits a contraction with the D-vector costs
    D*2**k_j complex multiply-add units (D*(2*2**k_j - 1) literal scalar
    operations), and forming its matrix costs M_j = 4**k_j + s_j visits,
    with s_j the scalar visits of the gate class's matrix expression in
    Qiskit 2.5.2 (H, SX, SXdg 0; CX 0; RX 6; RY 5; RZ 5; RXX 9; RYY 8;
    RZZ 9; RZX 7). One global-phase test per visited definition and, for a
    nonzero phase, D + 3 visits: W_phase = H + (D+3)*H_nz, here
    1 + [phase != 0]*(D+3). The weighted accumulation
    ``reference += (scale*coefficient)*evolved`` costs 2*D + 1. So
    W_vec = D*sum_j 2**k_j + sum_j (4**k_j + s_j) + H + (D+3)*H_nz + 2*D + 1,
    and the admitted work adds up to 2*D reshape/reordering copy visits and
    one census visit per elementary gate, W_admit = W_vec + 2*D*G + G.
    The known local arrays are six complex D-vectors (96*D bytes: the
    current vector, a transposed/reshaped copy, the contraction result, the
    final reordered copy, the reference accumulator and the weighting
    temporary) and one gate matrix at a time, 16*max_j 4**k_j bytes. An
    operation outside this elementary vocabulary is refused rather than
    priced by its arity. These units are admission proxies, not timings or
    equal-cost CPU operations.
    """
    from qiskit.circuit.library import (
        HGate, SXGate, SXdgGate, CXGate,
        RXGate, RYGate, RZGate, RXXGate, RYYGate, RZZGate, RZXGate,
    )
    prices = {
        HGate: (1, 0), SXGate: (1, 0), SXdgGate: (1, 0),
        CXGate: (2, 0), RXGate: (1, 6), RYGate: (1, 5),
        RZGate: (1, 5), RXXGate: (2, 9), RYYGate: (2, 8),
        RZZGate: (2, 9), RZXGate: (2, 7),
    }
    if dimension != 1 << circuit.num_qubits:
        raise ValueError("PF reference dimension differs from its circuit")
    if circuit.num_clbits or circuit.parameters:
        raise ValueError("PF reference needs a bound unitary circuit")
    contractions = scalar_contractions = matrices = 0
    largest_matrix = 0
    gates = 0
    for instruction in circuit.data:
        gate = instruction.operation
        price = prices.get(gate.base_class)
        if price is None or instruction.clbits:
            raise ValueError("PF reference gate needs an explicit matrix-work law")
        arity, scalar_visits = price
        if gate.num_qubits != arity:
            raise ValueError("PF reference gate arity differs from its work law")
        r = 1 << arity
        contractions += dimension * r
        scalar_contractions += dimension * (2 * r - 1)
        matrices += r * r + scalar_visits
        largest_matrix = max(largest_matrix, r * r)
        gates += 1
    phase_work = 1 + (dimension + 3) * int(circuit.global_phase != 0)
    accumulation = 2 * dimension + 1
    traffic = 2 * dimension * gates
    extra = matrices + phase_work + accumulation + traffic + gates
    return dict(
        work=contractions + extra,
        scalar_work=scalar_contractions + extra,
        contraction_work=contractions,
        gate_matrix_work=matrices,
        global_phase_work=phase_work,
        accumulation_work=accumulation,
        copy_work=traffic,
        census_work=gates,
        gate_count=gates,
        local_array_bytes=96 * dimension + 16 * largest_matrix,
    )


def pf_reference_held_payload(grid, *, held_inputs=(), max_work, max_bytes):
    """Return (held bytes, metadata visits) without copying numerical arrays.

    Count the selected LCHS numerical data, its grid schedule, and explicit
    caller-held input arrays. Follow array bases so shared buffers are
    counted once and a view pays for its complete backing allocation.
    The Method, compiled SDK objects, interpreter frames, allocator arenas
    and process RSS are outside this logical payload/native-wrapper law.

    Native allowances are 2048+16f for an f-field dataclass, 128+64n for
    an n-entry frozen mapping, 64+16n for a tuple, 128+16n for a list,
    256+16k for a k-axis
    array descriptor, and 64+4n for an n-character string. Integers use
    32+4*ceil(max(1,bit_length)/30). Float and complex slots use 32 and
    48. A 128-byte slot per visit reserves the temporary inventory set
    and traversal frontier. These are qualified engineering allowances.
    """
    from .selection import LCHSData
    seen = set()
    held = visits = 0

    def add(size):
        nonlocal held
        held += size
        if held + 128*visits > max_bytes:
            raise ValueError("PF reference resident payload and inventory exceed max_bytes")

    def visit(value):
        nonlocal visits
        if visits >= max_work:
            raise ValueError("PF reference payload inventory exceeds max_dense_work")
        visits += 1
        if held + 128*visits > max_bytes:
            raise ValueError("PF reference resident payload and inventory exceed max_bytes")
        if value is None:
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if type(value) is bool:
            add(32)
        elif type(value) is int:
            add(32 + 4*((max(1, abs(value).bit_length())+29)//30))
        elif type(value) is float:
            add(32)
        elif type(value) is complex:
            add(48)
        elif type(value) is str:
            add(64+4*len(value))
        elif isinstance(value, (bytes, bytearray)):
            add(64+len(value))
        elif isinstance(value, memoryview):
            add(128)
            visit(value.obj)
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays need an explicit resident-payload law")
            add(256+16*value.ndim)
            if value.base is None:
                add(value.nbytes)
            else:
                visit(value.base)
        elif isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise TypeError("object scalars need an explicit resident-payload law")
            add(128+value.dtype.itemsize)
        elif isinstance(value, FrozenArray):
            add(128)
            visit(value.array)
        elif isinstance(value, Fraction):
            add(64)
            visit(value.numerator)
            visit(value.denominator)
        elif is_dataclass(value) and not isinstance(value, type):
            record_fields = fields(value)
            add(2048+16*len(record_fields))
            for field in record_fields:
                if isinstance(value, LCHSData) and field.name == "method":
                    continue
                visit(getattr(value, field.name))
        elif isinstance(value, Mapping):
            add(128+64*len(value))
            for key, child in value.items():
                visit(key)
                visit(child)
        elif isinstance(value, (tuple, list)):
            add((128 if isinstance(value, list) else 64)+16*len(value))
            for child in value:
                visit(child)
        else:
            raise TypeError(f"resident payload needs a typed law for {type(value).__name__}")

    visit(grid)
    visit(held_inputs)
    return held, visits


def grid_reference(plan, grid, *, checks, counts, held_inputs=()):
    """Stream the selected finite sum on the padded system, then project once.

    ACL arXiv:2312.03916v2 Eqs.(2),(60)-(61): each application contributes
    weight*exp(shift*elapsed)*sum_j c_j U(elapsed,k_j)*physical_input.
    Coefficients here are the original unshifted quadrature c_j, so PSD
    compensation is applied once per application, including every source node.
    When exp(shift*T) exceeds binary64, the sum holds each compensation
    divided by 2**e (solution_error_budget.psd_recovery_exponent) and 2**e is
    applied to the projected vector at the end. The reference is None when
    that vector is not representable.

    A classical product-formula reference reads the selected coefficient
    table at most once per distinct k-node with a nonzero coefficient
    (time_independent_terms.stored_pf_nodes). A borrowed selected row costs
    one logical visit, and reading the identity coefficient and upper
    dropped mass costs two visits per node (stored_node_read_work), charged
    to the cumulative dense_work with the stored_node_read_work count as a
    diagnostic subset. The table remains in the consuming phase's held
    payload. No Pauli decomposition runs here. Vector actions and their
    phase and accumulation have separate charges.

    A classical product-formula reference applies each saved node action
    with the saved route and step count (host_pf.apply_node), its rotations
    formed from the stored coefficients at the application's elapsed time.
    The route's work (host_pf._route_requirements), the phase path (D + 4:
    negation and the identity-coefficient product, the imaginary argument
    and its exponential, and D products) and the weighted accumulation
    (2*D + 1; host_pf.host_phase_and_accumulation_work) are admitted under
    this invocation's limits before it runs; a saved route that does not fit
    is refused, never reselected.

    The classical product-formula reference holds the selected numerical
    payload and the caller's explicit verification inputs while each saved
    node action runs. That payload includes dense L and H, the coefficient
    and index tables, application records and input vectors. Shared array
    buffers are counted once. Each action adds its rotation buffer, saved
    route scratch and weighted-accumulation reserve to those held bytes.
    Setup and the one-time metadata inventory have separate phase peaks.
    The saved route and step count are replayed without re-selection
    (pf_reference_held_payload).

    The selected-grid quantum reference applies each selected decomposed
    product-formula branch directly to its application vector. Admission
    counts the contraction arity of every executed elementary gate, its
    matrix-entry and scalar-function work, each executed global phase, and
    weighted accumulation. It also reserves the vector-reordering copies.
    Contractions use complex multiply-add size units, while matrix
    formation and vector operations use logical visits. The byte allowance
    covers known arrays and the declared circuit-record payload. Opaque SDK
    storage and runtime are outside that allowance
    (selected_pf_vector_requirements).

    The quantum product-formula reference holds dense L and H and the full
    saved SELECT numerical payload while it builds and applies each branch.
    Held storage includes the occurrence-angle table, phases, mappings,
    stored node coefficients, source layout, saved records and input arrays,
    with shared buffers counted once. Circuit construction adds the two
    logical circuit-record populations in its preconstruction envelope.
    After decomposition, vector admission adds the surviving circuit's
    records and the known vector and elementary-gate arrays. Opaque SDK
    storage and the process heap are outside this known-payload allowance.
    """
    from nwqlib.problems.inputs import PhysicalScale
    from .time_independent_terms import (_spectral_host_requirements, spectral_lchs_sum,
        stored_pf_nodes, stored_node_read_work, _BorrowedRows)
    exponent = psd_recovery_exponent(grid.shift, plan.reconstruction.elapsed_time)
    d = plan.reconstruction.encoded_dimension
    pf = plan.method.hamiltonian_evolution_backend in {"trotter", "trotter_error_budgeted"}
    held = 0
    if pf:
        held, inventory_work = pf_reference_held_payload(
            grid, held_inputs=held_inputs,
            max_work=checks.max_dense_work-counts.get("dense_work", 0),
            max_bytes=checks.max_bytes,
        )
        dense_work(
            checks, counts, work=inventory_work,
            peak_bytes=held+128*inventory_work,
        )
    if pf and plan.execution == "classical":
        table = stored_pf_nodes(grid.payload)
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
        active = [app for app in grid.applications if app.weight != 0 and app.input_norm != 0]
        nodes = [(k, coefficient) for k, coefficient in zip(grid.nodes, grid.coefficients, strict=True)
                 if coefficient != 0]
        specifications = []
        for app in active:
            scale = app.weight*psd_recovery_part(grid.shift, app.elapsed, exponent)
            if not isfinite(scale):
                raise ValueError("selected-grid PSD recovery is nonfinite")
            specifications.append((app.elapsed, scale, app.vector))
        identity = not (np.any(grid.l_part) or np.any(grid.h_part)) or not any(app.elapsed for app in active)
        eigensystems = 0 if identity or not nodes else (1 if not np.any(grid.l_part) else len(nodes))
        inputs = len({id(app.vector) for app in active})
        work, phase_bytes = _spectral_host_requirements(d, nodes=len(nodes), applications=len(active),
            inputs=inputs, eigensystems=eigensystems)
        dense_work(checks, counts, work=work, peak_bytes=phase_bytes+80*d)
        if active and nodes:
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
            if coefficient == 0:
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
                if total > checks.max_pf_operations:
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
                selected = grid.payload.select_data.plan
                # Preconstruction envelope: an occurrence with support s >= 2
                # uses at most 2*(s-1) parity CX, 2*s basis changes and one
                # rotation, at most 4*s - 1 elementary gates, and support one
                # one rotation, so max(1, 4*q - 1)*W*R gates bound a branch
                # of W occurrences per step over R saved repetitions. Bytes:
                # 64 per gate for the undecomposed and the decomposed
                # circuit while both exist (the logical gate-record
                # allowance), the known arrays and one gate matrix of the
                # widest priced gate, 16*4**2 bytes.
                envelope = max(1, 4*q-1)*len(selected.occurrence_schedule)*sum(
                    selected.occurrence_block_repetitions)
                if counts["pf_operations"]+envelope > checks.max_pf_operations:
                    raise ValueError("selected-grid reference exceeds max_pf_operations before circuit construction")
                dense_work(
                    checks, counts, work=0,
                    peak_bytes=held+128*envelope+96*d+256,
                )
                circuit = _build_product_formula_branch(selected, app.slots[node], num_system_qubits=q)
                circuit.global_phase += selected.identity_phases[app.slots[node]]
                decomposed = circuit.decompose()
                del circuit
                requirements = selected_pf_vector_requirements(decomposed, d)
                gates = requirements["gate_count"]
                if counts["pf_operations"]+gates > checks.max_pf_operations:
                    raise ValueError("selected-grid reference exceeds max_pf_operations before its vector action")
                dense_work(checks, counts, work=requirements["work"],
                           peak_bytes=held+requirements["local_array_bytes"]+64*gates)
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
