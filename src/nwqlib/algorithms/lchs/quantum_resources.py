"""Explicit native representatives of an existing LCHS selection.

This samples saved numerical data and actual PREP children, without new phases,
MPS decomposition or a Run. Weighted totals are representative estimates.
The full native inventory comes from prepare(plan).circuits and the shared
circuit inspection.
"""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES
import numpy as np
from nwqlib._validation import integer
from nwqlib.operators.access import _check_bytes


def _native_envelope(limits, width, slots, *, dense=False):
    """Charge one representative to the cumulative gate/table size proxy.

    The proxy makes no promise about SDK CPU time or workspace.

    A representative with ``slots`` gates on ``width`` qubits counts
    slots*(width+1) table entries, or 4**width entries for a dense
    controlled matrix. The build work grows by the entries plus
    slots*(width+1), and 64 bytes per entry and per slot are admitted
    against max_bytes.
    """
    if width>limits['max_qubits']:
        raise ValueError('selected representative exceeds max_qubits')
    entries = (1 << (2*width)) if dense else slots*(width+1)
    work = entries+slots*(width+1)
    limits['used_build_work'] += work
    if limits['used_build_work']>limits['max_build_work']:
        raise ValueError('selected representatives exceed max_build_work')
    _check_bytes(64*entries+64*slots,limits['max_bytes'],'LCHS representative construction arrays')


def _synthesis_envelope(limits, qubit_counts, controls=(), controlled_qubit_counts=()):
    """Charge the exact dense syntheses of one representative, and Qiskit's control of them, to the build work.

    Each entry of ``qubit_counts`` is one call of
    ``_dense_synthesis.dense_unitary_circuit``, and its work law
    (``dense_synthesis_size``) is added to the cumulative build work. Each
    entry of ``controls`` is the ``(gates, instructions, heavy)`` triple of
    one call of Qiskit's control of such a synthesis, sized by
    ``gatewise_control_size``. Each entry of ``controlled_qubit_counts`` is
    one whole-matrix synthesis on that many qubits, controls included
    (``controlled_synthesis_size``). The calls run one at a time and the
    representative keeps their circuits, so the largest working allowance
    plus every kept allowance is admitted against max_bytes.
    """
    from nwqlib.subroutines._dense_synthesis import (controlled_synthesis_size, dense_synthesis_size,
                                                     gatewise_control_size)
    sizes = [dense_synthesis_size(count) for count in qubit_counts]
    sizes += [gatewise_control_size(*counts) for counts in controls if counts[0]]
    sizes += [controlled_synthesis_size(count) for count in controlled_qubit_counts]
    if not sizes:
        return
    limits['used_build_work'] += sum(size[0] for size in sizes)
    if limits['used_build_work']>limits['max_build_work']:
        raise ValueError('selected representatives exceed max_build_work')
    _check_bytes(max(size[1] for size in sizes)+sum(size[2] for size in sizes),limits['max_bytes'],
                 'LCHS representative dense synthesis')


def _product_formula_occurrence_blocks(
    select_data,
    *,
    num_system_qubits: int,
    implementation: str,
):
    """Build one actual resolved Pauli-rotation block per occurrence label."""

    from collections import Counter
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter
    from nwqlib.subroutines.qiskit_compat import controlled
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.backends.resources import SampledBlock
    from .select_synthesis import (_product_formula_branch_terms, _append_multiplexed_pauli_rotation,
                                  _append_structured_pauli_rotation)
    plan = select_data.plan
    num_control_qubits = plan.padded_node_count.bit_length() - 1
    if implementation == "branch_controlled":
        occurrences: Counter[str] = Counter()
        for branch, node in enumerate(plan.branch_to_node):
            if node is None or plan.branch_step_counts[branch] == 0:
                continue
            labels = tuple(label for label, _ in _product_formula_branch_terms(plan, branch))
            schedule = labels if plan.formula_order == 1 else (*labels[:-1], *labels[::-1])
            for label in schedule:
                occurrences[label] += plan.branch_step_counts[branch]
        blocks = []
        for label, multiplicity in occurrences.items():
            # A symbolic angle represents the admitted nonzero rotation class.
            # Zeros/support are counted before synthesis, and no physical branch is sampled.
            step = PauliEvolutionGate(SparsePauliOp(label), time=Parameter(f"theta_{label}")).definition
            step.name = f"pauli_rotation_{label}"
            circuit = QuantumCircuit(num_control_qubits + num_system_qubits)
            gate = controlled(step.to_gate(), num_control_qubits, ctrl_state=(1 << num_control_qubits) - 1)
            circuit.append(gate, range(circuit.num_qubits))
            blocks.append(
                SampledBlock(
                    name=f"select_controlled_pauli_{label}",
                    circuit=circuit,
                    multiplicity=multiplicity,
                )
            )
        return blocks
    representatives: dict[str, tuple[int, int]] = {}
    for index, angles in enumerate(plan.occurrence_angle_tables.array):
        label = plan.occurrence_schedule[index % len(plan.occurrence_schedule)]
        repetitions = plan.occurrence_block_repetitions[index // len(plan.occurrence_schedule)]
        if label in representatives:
            first_index, multiplicity = representatives[label]
            representatives[label] = (first_index, multiplicity + repetitions)
        else:
            representatives[label] = (index, repetitions)

    blocks: list[SampledBlock] = []
    affine_tables = (
        tuple(plan.structured_generator_payload["occurrence_affine_tables"])
        if implementation == "structured" and plan.structured_generator_payload
        else ()
    )
    for label, (index, multiplicity) in representatives.items():
        control = QuantumRegister(num_control_qubits, "lcu_control")
        system = QuantumRegister(num_system_qubits, "system")
        circuit = QuantumCircuit(control, system, name=f"select_occurrence_{label}")
        if implementation == "structured":
            _append_structured_pauli_rotation(
                circuit,
                system_qubits=list(system),
                control_qubits=list(control),
                label=label,
                affine=affine_tables[index],
            )
        else:
            _append_multiplexed_pauli_rotation(
                circuit,
                system_qubits=list(system),
                control_qubits=list(control),
                label=label,
                angles=plan.occurrence_angle_tables.array[index],
            )
        blocks.append(
            SampledBlock(
                name=f"select_occurrence_{label}",
                circuit=circuit,
                multiplicity=multiplicity,
            )
        )
    return blocks


def dense_representative_envelope(q, a, route="gatewise"):
    """Return value-independent counts in explicitly named owner bases.

    This computes a resource model. It builds no representative circuit and
    does not report the operations of an arbitrary transpiled circuit.

    Dense representative resource reporting uses the value-independent upper
    counts of the selected library synthesis and control route. Each record
    identifies its basis, route, owner assumptions and
    ``count_kind="structural_upper_bound"``. It is not an observed circuit
    inventory. Counts for an arbitrary explicitly requested compilation
    continue to require its actual circuit inspection.

    For system width q >= 1 and address width a, the uncontrolled dense
    synthesis census (C_q, U_q, Z_q, H_q) in the basis CX, U, RZ, H is
    (0, 1, 0, 0) at one qubit and, for q >= 2,
    C_q = (25*4**q - 72*2**q + 32)/48, U_q = 7*4**(q-2),
    Z_q = (3*4**q - 12*2**q)/8, H_q = (4**q - 16)/24
    (_dense_synthesis.dense_synthesis_gate_census). Whole-matrix control on
    m = q + a >= 3 qubits is two (m-1)-qubit syntheses and one multiplexed
    RZ (_dense_synthesis.controlled_synthesis_gate_census). Gatewise control
    has the CX bound C_branch = C_q*x(a+1) + H_q*x(a) + Z_q*z(a) + U_q*u(a)
    + p(a) of native.dense_branch_select_cx, with x the ancilla-free
    multi-controlled X count, z, y and u the MCRZ, MCRY and U costs and p
    the branch global phase, and the control constructor populations of
    _dense_synthesis.gatewise_control_counts. A gatewise U bound is not
    available and is reported as None, not zero. Open controls conjugate by
    at most 2*a X gates, which change no CX count. With no address bits the
    route is ``uncontrolled``. A one-system-qubit branch follows the
    gatewise count, as the builder does, and ``auto`` resolves through
    select_dense_control_route. The basis U bound of the whole-matrix and
    uncontrolled routes is the direct translation of U, RZ, H and the open
    control X gates to U, without routing or gate expansion.
    """
    from nwqlib.algorithms.lchs.native import dense_branch_select_cx
    from nwqlib.subroutines._dense_synthesis import (
        DENSE_CONTROL_ROUTES,
        dense_synthesis_gate_census,
        controlled_synthesis_gate_census,
        gatewise_control_counts,
        select_dense_control_route,
    )
    if type(q) is not int or q < 1 or type(a) is not int or a < 0:
        raise ValueError("dense representative widths must be q >= 1 and a >= 0")
    if route not in DENSE_CONTROL_ROUTES:
        raise ValueError("unknown dense control route")
    resolved = "uncontrolled" if a == 0 else select_dense_control_route(route, a)
    if q == 1 and a:
        resolved = "gatewise"
    cx = dense_branch_select_cx(q, a, route)
    result = dict(
        count_kind="structural_upper_bound",
        system_qubits=q,
        address_qubits=a,
        resolved_route=resolved,
        basis_cx_upper_bound=cx,
        open_control_x_upper_bound=2 * a,
        synthesis_gate_upper_bounds=None,
        control_constructor_upper_bounds=None,
        basis_u_upper_bound=None,
    )
    if resolved == "uncontrolled":
        census = dense_synthesis_gate_census(q)
    elif resolved == "whole_matrix":
        census = controlled_synthesis_gate_census(q + a)
    else:
        gates, instructions, heavy = gatewise_control_counts(q, a)
        result["synthesis_gate_upper_bounds"] = dense_synthesis_gate_census(q)
        result["control_constructor_upper_bounds"] = dict(
            gates=gates, instructions=instructions, heavy=heavy
        )
        return result
    result["synthesis_gate_upper_bounds"] = census
    # Direct translation of U/RZ/H/X to U, without routing or gate expansion.
    result["basis_u_upper_bound"] = (
        census["u"] + census["rz"] + census["h"] + 2 * a
    )
    return result


def _dense_select_bound(payload, data):
    """Return the dense SELECT's structural upper-bound record for N physical branches.

    Every physical branch of a homogeneous dense Plan has the same width and
    route, so one bound C per branch gives N*C for SELECT CX. No smallest,
    median or largest |k| branch is built. The record keeps its basis and
    route, and it is not added to the raw representative inventory.
    """
    bound = dense_representative_envelope(data.num_system_qubits, data.num_control_qubits,
                                          payload.method.dense_control_route)
    branches = len(payload.coefficient_plan.coefficients)
    return dict(name="dense_select", multiplicity=branches, bound=bound,
                basis_cx_upper_bound_total=branches*bound["basis_cx_upper_bound"])


def _pf_samples(payload, data, operation):
    """Yield one product-formula occurrence block per Pauli label and the phase blocks.

    Each label's block carries its saved repetition multiplicity. The phase
    part is the combined coefficient and identity diagonal for multiplexed
    or structured SELECT, or per-branch controlled phases and open-control X
    gates for branch-controlled SELECT.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from nwqlib.subroutines.qiskit_compat import controlled
    from .select_synthesis import _product_formula_branch_phase_count
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases

    selected=payload.select_data
    plan=selected.plan
    a,q=data.num_control_qubits,data.num_system_qubits
    # The existing occurrence kernel groups branch-controlled nonzero classes
    # or selects one stored table per label, keeping exact multiplicities.
    labels=len(set(plan.occurrence_schedule))
    tables=len(plan.occurrence_angle_tables)*plan.padded_node_count
    _native_envelope(operation,a+q,max(1,labels)*(8*(1<<a)+8*q))
    _native_envelope(operation,a+q,tables)
    for block in _product_formula_occurrence_blocks(selected,num_system_qubits=q,implementation=payload.selected_select):
        yield block.name,block.circuit,block.multiplicity
    _native_envelope(operation,a+q,8*(1<<a))
    if payload.selected_select=="branch_controlled":
        phases=_product_formula_branch_phase_count(plan)
        masks=sum(2*(a-branch.bit_count()) for branch,node in enumerate(plan.branch_to_node) if node is not None)
        if phases:
            phase=QuantumCircuit(q,name="selected_branch_phase")
            phase.global_phase=Parameter("selected_branch_phase")
            circuit=QuantumCircuit(a+q)
            circuit.append(controlled(phase.to_gate(), a, ctrl_state=(1 << a) - 1),range(a+q))
            yield "branch_phase",circuit,phases
        if masks:
            circuit=QuantumCircuit(1)
            circuit.x(0)
            yield "open_control_x",circuit,masks
    else:
        circuit=QuantumCircuit(a+q)
        combined=tuple(coefficient+identity for coefficient,identity in
                       zip(plan.coefficient_phases,plan.identity_phases,strict=True))
        append_control_diagonal_phases(circuit,list(circuit.qubits[:a]),combined)
        yield "control_diagonal",circuit,1


def _qsp_samples(payload, data, operation):
    """Yield one controlled joint-generator query and the QSP fixed blocks with their counts.

    Multiplicities come from the saved structural law (block-encoding
    queries, projector phases, boundary gates and OAA reflections), so the
    degree-long QSP circuit is never built for this estimate.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import controlled
    from nwqlib.subroutines.qsp import build_control_diagonal_generator_encoding
    from nwqlib.subroutines._semantic import zero_reflection
    from nwqlib.subroutines.qsp.evolution import _append_projector_phase
    from .native import _build_compiled_qsp_part_encoding,_compiled_coefficient_phase_gate

    selected=payload.qsp_plan
    l_part,h_part,_=selected["part_plans"]
    layout,law=selected["layout"],selected["structural_law"]
    ancillas=layout["num_ancillas"]+3
    width=ancillas+data.num_control_qubits+data.num_system_qubits
    # This is one selected child/query template, never the degree-long QSP
    # circuit. No call to prepare_qsp_evolution or phase solving occurs.
    _native_envelope(operation,width,8*(1<<data.num_control_qubits)+8*(1<<data.num_system_qubits))
    # Each active dense-dilation child is synthesized once, in its controlled
    # generator branch or, for a single child, in the controlled query below,
    # and Qiskit controls the synthesized gates there and in the query. On
    # the whole-matrix route the generator branch synthesizes the child's
    # controlled matrix instead (compiled_select_controlled_syntheses).
    active=[child for child,scale in ((l_part,layout["l_diagonal_max_abs"]),(h_part,layout["h_diagonal_max_abs"]))
            if scale>0.]
    route=payload.method.dense_control_route
    from .compiled_selection import compiled_select_controlled_syntheses, compiled_select_dense_controls
    controlled_syntheses=compiled_select_controlled_syntheses(selected,route=route)
    _synthesis_envelope(operation,() if controlled_syntheses else
                        tuple(child.system_qubits+child.num_ancillas for child in active
                              if child.implementation=="dense_dilation"),
                        compiled_select_dense_controls(selected,route=route,queries=(1,)),controlled_syntheses)
    encoding_l=_build_compiled_qsp_part_encoding(l_part)
    encoding_h=None if h_part.implementation=="exact_zero" else _build_compiled_qsp_part_encoding(h_part)
    generator=build_control_diagonal_generator_encoding(encoding_l,encoding_h,
        l_diagonal=selected["l_diagonal"],h_diagonal=selected["h_diagonal"],dense_control_route=route)
    query=QuantumCircuit(width)
    query.append(controlled(generator.circuit.to_gate(),1),[0,*range(3,width)])
    yield "controlled_joint_query",query,law["block_encoding_queries"]
    _native_envelope(operation,width,8*width)
    from nwqlib.subroutines.qsp.evolution import wx_phases_to_reflection
    reflection_phases,global_phase=wx_phases_to_reflection(selected["prepared_evolution"].cos_solution.phases)
    phase=QuantumCircuit(width-1,name="projector_phase")
    _append_projector_phase(phase,phase.qubits[1],list(phase.qubits[2:ancillas-1]),
                            float(reflection_phases[0]),pair_qubit=phase.qubits[0])
    circuit=QuantumCircuit(width)
    circuit.append(controlled(phase.to_gate(),1),range(width))
    yield "controlled_projector_phase",circuit,law["projector_phase_count"]
    coefficient=QuantumCircuit(width)
    gate=_compiled_coefficient_phase_gate(np.asarray(data.original_coefficients),data.padded_term_count)
    if gate is not None:
        coefficient.append(gate,range(ancillas,ancillas+data.num_control_qubits))
    yield "coefficient_phase",coefficient,1
    for name,gate,qubits,key in (("pair_boundary","ch",2,"pair_boundary_h_count"),
                               ("parity_boundary","h",1,"parity_boundary_h_count"),
                               ("odd_pair_sign","cz",2,"odd_pair_sign_count")):
        circuit=QuantumCircuit(qubits)
        getattr(circuit,gate)(*range(qubits))
        yield name,circuit,law[key]
    circuit=QuantumCircuit(1)
    circuit.p(float(global_phase),0)
    yield "parity_scalar_phase",circuit,law["parity_scalar_phase_count"]
    yield "ancilla_reflection",zero_reflection(ancillas),law["oaa_reflection_count"]


def _representatives(plan,payload,operation,bounds):
    """Yield explicit selected PREP, evolution and readout representatives with their multiplicities.

    A dense exact SELECT builds no representative: its structural upper-bound
    record (_dense_select_bound) is appended to ``bounds`` instead.
    """
    from qiskit import QuantumCircuit,ClassicalRegister
    from nwqlib.blocks.lowering import _local_method_context
    from nwqlib.subroutines.lcu.core import prepare_lcu_gate_data
    from .quantum import observed_bits
    context = _local_method_context()
    selected = {record.content_id:record for record in plan.construction.selections}
    blocks = {block.record.signature.name:block for block in plan.blocks}
    rec=plan.reconstruction
    width=len(rec.success_bits)+len(rec.system_bits)
    prepared = {}
    for name in ('initial_prep','coefficient_prep','coefficient_inverse'):
        block=blocks.get(name)
        if block is None:
            if name=='initial_prep' or rec.padded_branches>1:
                raise ValueError('sampling requires actual selected PREP children')
            continue
        if selected.get(block.record.content_id)!=block.record:
            raise ValueError('sampled PREP is not part of this construction')
        _native_envelope(operation,width,max(1,block.record.construction_work or 0))
        if name=='coefficient_inverse':
            if block._base is not blocks['coefficient_prep'] or not block.record.adjoint or block.record.controlled:
                raise ValueError('coefficient inverse does not match the actual selected PREP')
            from nwqlib.subroutines.qiskit_compat import inverse_realized_gate
            base=prepared['coefficient_prep']
            circuit=QuantumCircuit(base.num_qubits)
            circuit.append(inverse_realized_gate(base.to_gate()),circuit.qubits)
        else:
            circuit=block._constructor(block,(),context)
            prepared[name]=circuit
        yield name,circuit,1
    data=prepare_lcu_gate_data(payload.coefficient_plan.coefficients,system_dimension=len(payload.initial),
        max_bytes=payload.method.max_bytes,max_work=payload.method.max_select_work)
    if payload.selected_select=='identity_evolution':
        select=blocks['lchs_select']
        _native_envelope(operation,width,1<<data.num_control_qubits)
        yield 'identity_evolution',select._constructor(select,(),context),1
    elif payload.method.hamiltonian_evolution_backend=='dense_exact':
        bounds.append(_dense_select_bound(payload,data))
    elif payload.select_data is not None:
        yield from _pf_samples(payload,data,operation)
    else:
        yield from _qsp_samples(payload,data,operation)
    # Include readout transformations and measurements in the sampled
    # construction cost, not only the central evolution blocks.
    for setting in rec.settings:
        _native_envelope(operation,width,4*width)
        circuit=QuantumCircuit(width)
        for bit,axis in enumerate(reversed(setting.label)):
            if plan.shots is None or axis not in 'XY':
                continue
            child=blocks.get(f'rotate_{axis.lower()}_{bit}')
            if child is None or selected.get(child.record.content_id)!=child.record:
                raise ValueError('sampled readout needs its exact selected group basis change')
            circuit.compose(child._constructor(child,(),context),qubits=rec.system_bits,inplace=True)
        if plan.shots is not None:
            observed=observed_bits(rec)
            register=ClassicalRegister(len(observed),'readout')
            circuit.add_register(register)
            circuit.measure(observed,register)
        yield 'readout_'+setting.name,circuit,1


# The default caps (13 qubits, 100,000,000 build-work units and 100,000
# inspected operations) are untuned limits of this explicit inspection,
# registered in ENGINEERING_CONSTANTS. Nothing is simulated. A dense exact
# SELECT is reported by its structural upper-bound record
# (dense_representative_envelope, _dense_select_bound), an integer evaluation
# that forms no eigensystem, matrix or circuit and charges no build work.
# A dense-dilation QSP child adds the synthesis of its unitary on q + 1
# qubits and Qiskit's control of it (_dense_synthesis.dense_synthesis_size,
# gatewise_control_size). Product-formula charges grow with the label count
# times 2**a and with the stored angles (template occurrences times
# repetition blocks times 2**a). QSP, PREP and readout charges grow with
# 2**a, 2**q and the width. max_qubits=13 is a policy cap, not a
# consequence of these laws.
def sample_resources(plan, *, max_qubits=13, max_bytes=DEFAULT_MAX_BYTES,
                     max_build_work=100_000_000,max_operations=100_000,transpile_options=None):
    """Inventory selected representative circuits and weight them by multiplicity.

    Returns ``representatives`` (name, saved multiplicity and the operation
    inventory of each representative circuit) and ``weighted_totals`` for each
    readout setting (a sampled group, a mass setting or the exact reduction):
    every core representative's operation counts times its multiplicity,
    plus that setting's readout representative (its group basis change and
    measurements, none for the exact reduction). The weighted
    totals are estimates of the selected construction's operation counts. They
    do not model angle-specific cancellation, cross-block optimization or SDK
    workspace. ``structural_bounds`` lists the records that no raw inventory
    represents: for a dense exact SELECT, its structural upper bound for
    every physical branch (dense_representative_envelope) in the named owner
    bases, never added to a weighted total. max_qubits bounds the Plan width and the width of every
    representative circuit.
    max_build_work bounds summed known gate/table size proxies.
    max_operations bounds all inspected representative operations. No cap
    bounds SDK runtime or process RSS. Transpilation occurs only when
    explicitly requested.
    """
    from nwqlib.core.planning import Plan
    from nwqlib.backends.inspection import inspect_circuit_resources
    from .method import LCHS
    from .selected_grid import selected_payload
    from .selection import LCHSData
    if type(plan) is not Plan or type(plan.method) is not LCHS or plan.execution!='quantum':
        raise ValueError('resource sampling requires the actual quantum LCHS Plan')
    limits=dict(max_qubits=integer(max_qubits,'max_qubits',1),max_bytes=max_bytes,
        max_build_work=integer(max_build_work,'max_build_work',1),used_build_work=0)
    remaining=integer(max_operations,'max_operations',1)
    payload=selected_payload(plan)
    if type(payload) is not LCHSData or payload.source is not None:
        raise ValueError('representative sampling supports homogeneous dense inputs; inspect actual prepared circuits for source/periodic work')
    width=len(plan.reconstruction.success_bits)+len(plan.reconstruction.system_bits)
    _native_envelope(limits,width,1)
    representatives=[]
    core={}
    readout={}
    bounds=[]
    for name,circuit,multiplicity in _representatives(plan,payload,limits,bounds):
        if remaining==0 and len(circuit.data)>0:
            raise ValueError('selected representative inventory exceeds max_operations')
        # The raw representatives are not in cx,u. That basis is only the
        # default target of an explicitly requested transpilation.
        inventory=inspect_circuit_resources(circuit,native_basis=('cx','u') if transpile_options else (),
            label=f'LCHS {name} representative (SELECT={payload.selected_select})',
            max_operations=max(1,remaining),max_bytes=max_bytes,
            transpile_options=transpile_options,basis_label='selected_sample_raw')
        remaining-=inventory['total_operations']
        if remaining<0:
            raise ValueError('selected representative inventory exceeds max_operations')
        representatives.append(dict(name=name,multiplicity=multiplicity,inventory=inventory))
        target=readout.setdefault(name.removeprefix('readout_'),{}) if name.startswith('readout_') else core
        for operation,count in inventory['operations'].items():
            target[operation]=target.get(operation,0)+multiplicity*count
        # Fixed 8192-byte envelope per stored inventory dictionary, registered
        # in ENGINEERING_CONSTANTS.
        _check_bytes(8192*len(representatives),max_bytes,'representative inventories')
    weighted_totals={}
    for setting in plan.reconstruction.settings:
        if setting.name not in readout:
            raise ValueError(f'representative sampling built no readout inventory for setting {setting.name}')
        totals=dict(core)
        for operation,count in readout[setting.name].items():
            totals[operation]=totals.get(operation,0)+count
        weighted_totals[setting.name]=dict(operations=totals,total_operations=sum(totals.values()))
    return dict(representatives=tuple(representatives),weighted_totals=weighted_totals,
                structural_bounds=tuple(bounds))
