"""Selected PF/QSP/MPS data and component laws consumed by LCHS execution."""

from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
from hashlib import sha256
from math import fsum
import struct
from types import MappingProxyType

import numpy as np

from nwqlib.core.records import FrozenArray
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _freeze_array
from nwqlib.problems.inputs import compose_recovery
from .primary_records import METHOD


def freeze_selected(value, *, max_bytes):
    """Snapshot numerical selection arrays once; metadata does no numerical work.

    Writable arrays are copied read-only and their bytes are admitted against
    max_bytes first. Already frozen arrays are shared, not copied, so a Plan
    holds one immutable copy of each selected array.
    """
    copied = 0
    def freeze(item):
        """Return item with every writable array replaced by a read-only snapshot.

        Containers are rebuilt recursively. Mappings become MappingProxyType,
        lists become tuples, dataclasses are rebuilt from their frozen fields
        and NumPy scalars become Python scalars. None and plain int, float,
        complex, str and bool pass unchanged. Any other type raises, so no
        arbitrary object enters a Plan.
        """
        nonlocal copied
        if isinstance(item, FrozenArray):
            return item
        if isinstance(item, np.ndarray):
            if not item.flags.writeable:
                return item
            # The writable source and its read-only snapshot coexist.
            copied += 2*item.nbytes
            _check_bytes(copied, max_bytes, "LCHS selected array snapshots")
            return _freeze_array(item)
        if isinstance(item, Mapping):
            return MappingProxyType({key:freeze(child) for key,child in item.items()})
        if is_dataclass(item):
            return replace(item, **{field.name:freeze(getattr(item,field.name)) for field in fields(item)})
        if isinstance(item,(tuple,list)):
            return tuple(freeze(child) for child in item)
        if isinstance(item,np.generic):
            return item.item()
        if item is None or type(item) in (int,float,complex,str,bool):
            return item
        raise TypeError(f"unsupported LCHS selected numerical data: {type(item).__name__}")
    return freeze(value)


def _feed_selected_value(digest, value):
    """Feed one frozen selected value into ``digest`` as a prefix-free encoding.

    Every value starts with a one-byte kind tag, and every container states
    how many children follow. The byte stream therefore determines the nested
    structure, so two different values cannot produce the same stream. Without
    the counts, ``{'x': {'IZ': 1.0}, 'y': 2.0}`` and
    ``{'x': {'IZ': 1.0, 'y': 2.0}}`` would feed identical bytes.

    - ``A``: an array, or the array of a ``FrozenArray``, as its dtype and
      shape followed by its raw bytes, whose length the dtype and shape fix.
    - ``M``: a mapping, as its entry count and then each key and value in
      sorted key order.
    - ``D``: a dataclass, as its type name, its field count and each field
      name with its value.
    - ``S``: a tuple or list, as its length and then its items.
    - ``f`` and ``c``: a float or complex number, as exact little-endian
      binary64.
    - ``r``: any other scalar that ``freeze_selected`` admits (int, str, bool
      or None), as its type name and repr, ended by ``;``. The repr of a
      string escapes its quote character, so the terminator is unambiguous.

    NumPy scalars are fed as the equal Python scalar.
    """
    if isinstance(value, FrozenArray):
        value = value.array
    if isinstance(value, np.ndarray):
        digest.update(b'A' + (str((value.dtype.str, value.shape)) + ';').encode())
        if value.size:
            digest.update(memoryview(value).cast("B"))
    elif isinstance(value, Mapping):
        digest.update(b'M' + struct.pack('<Q', len(value)))
        for key in sorted(value):
            _feed_selected_value(digest, key)
            _feed_selected_value(digest, value[key])
    elif is_dataclass(value):
        digest.update(b'D')
        _feed_selected_value(digest, type(value).__qualname__)
        members = fields(value)
        digest.update(struct.pack('<Q', len(members)))
        for field in members:
            _feed_selected_value(digest, field.name)
            _feed_selected_value(digest, getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        digest.update(b'S' + struct.pack('<Q', len(value)))
        for child in value:
            _feed_selected_value(digest, child)
    elif isinstance(value, np.generic):
        _feed_selected_value(digest, value.item())
    elif type(value) is float:
        digest.update(b'f' + struct.pack('<d', value))
    elif type(value) is complex:
        digest.update(b'c' + struct.pack('<dd', value.real, value.imag))
    else:
        digest.update(b'r' + (type(value).__name__ + ':' + repr(value) + ';').encode())


def selected_identity(data, problem, method):
    """Bind one selected numerical recipe to its already ingested scientific inputs.

    The SHA-256 covers the Problem and Method identities and every selected
    array, record and scalar in LCHSData, with dtype, shape and exact binary64
    bytes, encoded by ``_feed_selected_value``. The SELECT record or host
    kernel stores this identity, and archive loading recomputes it, so a saved
    payload that differs in any selected number or in its nesting is rejected
    instead of silently executed.
    """
    if data.method != method:
        raise ValueError("selected LCHS data belongs to another Method")
    digest = sha256((problem.content_id+method.content_id).encode())
    _feed_selected_value(digest, (
        data.initial, data.source, data.initial_direction, data.source_direction,
        data.coefficient_plan, data.quadrature, data.select_data, data.select_records,
        data.source_layout, data.qsp_plan, data.initial_mps, data.coefficient_mps,
        data.selected_select, data.host_actions))
    return 'sha256:'+digest.hexdigest()


def select_parameters(method, problem, data):
    """Select every numerical SELECT and PREP product once, before any circuit exists.

    This owns the planning choices made after the coefficient grid is known.
    It resolves the SELECT implementation, lays out constant-source branches,
    builds the product-formula or QSP joint-generator plan, and computes the
    coefficient PREP amplitudes and any requested MPS decompositions. The
    results are frozen into LCHSData, whose content hash (selected_identity)
    is bound to the SELECT record, so execution, archive loading and
    verification reuse these exact numbers and never reselect them.

    A dense_exact selection whose padded slot count exceeds
    max_dense_select_slots is refused here, before native construction. Each
    dense branch is a controlled, classically computed matrix exponential, so
    construction cost and memory grow with both the slot count and the system
    dimension. The refusal names the choices the caller can change. It never
    lowers the k-quadrature, the Duhamel node count or the tolerance to fit,
    because that would silently change the selected approximation.

    For qsp_block_encoding the joint generator is D_L (x) L + D_H (x) H on the
    address register, the effective Hamiltonian of Pocrnic et al.,
    arXiv:2506.20760v2, Section IV, Eqs. (61)-(62). Homogeneous evolution uses
    the k-nodes and ones as diagonals. Source branches scale both diagonals by
    elapsed_time/T, so one evolution for time T gives each branch its own
    elapsed time. When L = 0 the roles swap and H is the only nonzero child.

    Args:
        method: Configured LCHS Method whose choices and limits apply.
        problem: LinearDynamics with its ingested physical input scales.
        data: LCHSData from select_dense, holding the homogeneous coefficient plan.

    Returns:
        A pair of the frozen LCHSData with its selected SELECT products and a
        dict of selected values for reconstruction and resource laws: the
        physical branch count, SELECT ancillas, QSP recovery scale, PREP
        amplitude array and, where applicable, step counts and Duhamel nodes
        and weights.
    """
    from .native import LCHS_QSP_EPSILON_FRACTION, resolve_lcu_select_implementation
    from nwqlib.subroutines._multiplexors import product_formula_select_resource_law
    from .time_independent_terms import generate_lchs_product_formula_select_plan, _build_product_formula_select_plan, _trotter_pauli_decomposition, _admit_pauli_decomposition, LCHSProductFormulaSelectData
    backend = method.hamiltonian_evolution_backend
    dimension = len(data.initial)
    # Stage 1: provisional SELECT route. Product-formula routes are resolved
    # again below, once their plan and its structure certificate exist.
    resolved = resolve_lcu_select_implementation(data.method,he_backend=backend)
    selected_select = resolved['resolved_lcu_select_implementation']
    extra = dict(select_ancillas=0,qsp_recovery=1.,physical_branches=len(data.coefficient_plan.coefficients))
    # Stage 2: constant-source branch layout (initial and Duhamel applications).
    layout = None
    if data.source is not None:
        from .source_selection import select_source_branches
        initial_norm = problem.initial_state.preparation.physical_scale.as_float()
        source_norm = problem.source.preparation.physical_scale.as_float()
        if initial_norm is None or source_norm is None:
            raise ValueError("constant-source branch coefficients need representable input norms")
        layout = select_source_branches(coefficients=data.quadrature.coefficients,k_nodes=data.quadrature.k_nodes,
            initial_norm=initial_norm,source_norm=source_norm,final_time=problem.elapsed_time,
            psd_shift=data.quadrature.conversion['psd_shift'],duhamel_nodes=method.duhamel_nodes,
            compact=selected_select=='branch_controlled',max_bytes=method.max_bytes,
            max_quadrature_work=method.max_quadrature_work)
        data = replace(data,source_layout=layout)
        extra.update(physical_branches=layout.branch_count,source_nodes=layout.source_nodes,source_weights=layout.source_weights)
    # Stage 3: refuse an oversized dense SELECT before any construction.
    coefficients = data.coefficient_plan.coefficients if layout is None else layout.coefficients
    padded_slots = 1 << (len(coefficients)-1).bit_length()
    if backend=='dense_exact' and padded_slots>method.max_dense_select_slots:
        width = (len(coefficients)-1).bit_length()+dimension.bit_length()-1
        raise ValueError(f'LCHS time={problem.elapsed_time}, ||L||={data.quadrature.l_norm:g}, '
            f'approximation_tolerance={method.approximation_tolerance:g} selects {len(coefficients)} branches, '
            f'{padded_slots} padded SELECT slots and {width} total qubits; max_dense_select_slots={method.max_dense_select_slots}. '
            'Choose a shorter time, explicitly change the construction tolerance/limit, or select supported '
            'multiplexed product-formula evolution with its different approximation and cost.')
    if selected_select == 'branch_controlled':
        # native.construct_select passes these coefficients through the LCU
        # intake before it builds any branch. dense_exact always resolves to
        # branch_controlled, and a product-formula request keeps it, so the
        # same call here refuses at planning a Plan that construction would
        # refuse.
        from nwqlib.subroutines.lcu.data import lcu_coefficient_intake
        lcu_coefficient_intake(coefficients,system_dimension=dimension,max_bytes=method.max_bytes,
            max_work=method.max_select_work)
    # Stage 4: branch evolution (product-formula tables or QSP joint generator).
    if backend in ('trotter','trotter_error_budgeted'):
        if layout is None:
            ledger = {}
            selected = generate_lchs_product_formula_select_plan(final_time=problem.elapsed_time,
                method=data.method,_quadrature_plan=data.quadrature,
                max_steps=method.max_trotter_steps,max_bytes=method.max_bytes,max_select_work=method.max_select_work,
                work_ledger=ledger)
            extra['select_work_charged'] = ledger['work']
        else:
            # The homogeneous planner's decomposition admission, with the
            # plan receiving the rest of the same work limit.
            remaining_work = _admit_pauli_decomposition(dimension,max_bytes=method.max_bytes,
                max_select_work=method.max_select_work)
            decomposition = _trotter_pauli_decomposition(data.quadrature.l_part,data.quadrature.h_part)
            ledger = {}
            pf,records,nodes = _build_product_formula_select_plan(decomposition=decomposition,k_values=layout.k_values,
                elapsed_times=layout.elapsed_times,coefficients=layout.coefficients,branch_to_node=layout.branch_to_node,
                method=data.method,address_structure=data.quadrature.coefficient_plan.quadrature.address_structure,
                max_steps=method.max_trotter_steps,
                max_bytes=method.max_bytes,max_select_work=remaining_work,grid_k_values=data.quadrature.k_nodes,
                work_ledger=ledger)
            extra['select_work_charged'] = method.max_select_work-remaining_work+ledger['work']
            from .time_independent_terms import _trotter_budget_quadrature_terms
            summary = _trotter_budget_quadrature_terms(records,coefficients=layout.coefficients,method=data.method)
            selected = LCHSProductFormulaSelectData(coefficients=layout.coefficients,plan=pf,quadrature=summary,
                numerical_psd_premise_satisfied=data.quadrature.numerical_psd_premise_satisfied,nodes=nodes)
            data = replace(data,select_records=records)
        resolved = resolve_lcu_select_implementation(data.method,he_backend=backend,
            candidate_costs=product_formula_select_resource_law(selected.plan),select_plan=selected.plan)
        selected_select = resolved['resolved_lcu_select_implementation']
        data = replace(data,select_data=freeze_selected(selected,max_bytes=method.max_bytes))
        extra['step_counts'] = selected.plan.branch_step_counts
    zero_generator = not np.any(data.quadrature.l_part) and not np.any(data.quadrature.h_part)
    if backend == 'qsp_block_encoding' and zero_generator:
        # exp(-it*0)=I has no encoding scale or QSP polynomial. Keep the
        # actual source PREP and coefficient phase action in the selected leaf.
        selected_select = 'identity_evolution'
    if backend == 'qsp_block_encoding' and not zero_generator:
        from .compiled_selection import _compiled_qsp_part_plans, _compiled_select_qsp_plan
        from nwqlib.subroutines.qsp.evolution import prepare_qsp_evolution
        unitary = data.quadrature.l_norm==0
        first,second = ((data.quadrature.h_part,data.quadrature.l_part) if unitary
            else (data.quadrature.l_part,data.quadrature.h_part))
        # select_work_remaining is the max_select_work that the outer gate's
        # ledger leaves for the remaining construction stages. The QSP phase and
        # polynomial stages below admit their own limits. Readout is funded by
        # max_readout_work.
        l_child,h_child,h_mass,select_work_remaining = _compiled_qsp_part_plans(first,second,
            l_norm=None if unitary else data.quadrature.l_norm,
            max_bytes=method.max_bytes,max_select_work=method.max_select_work)
        parts = (l_child,h_child,h_mass)
        # One evolution of D_L (x) L + D_H (x) H for the full time T gives slot
        # j the branch exp(-i*T*(D_L[j]*L + D_H[j]*H)). With D_L[j] = (t_j/T)*k_j
        # and D_H[j] = t_j/T this is exp(-i*t_j*(k_j*L + H)), the branch of a
        # source application with elapsed time t_j.
        if layout is None:
            diagonal_l,diagonal_h = data.quadrature.k_nodes,np.ones(len(coefficients))
        else:
            diagonal_l,diagonal_h = layout.elapsed_times*layout.k_values/problem.elapsed_time,layout.elapsed_times/problem.elapsed_time
        if unitary:
            # L=0: the first and only joint-generator child is H, weighted by
            # each application's elapsed time. No artificial k integral.
            diagonal_l,diagonal_h = diagonal_h,diagonal_l
        # Synthesis receives its own 0.1*tol component allowance. The kernel's
        # tail and k-quadrature allocations do not cover this evolution error.
        selected = _compiled_select_qsp_plan(part_plans=parts,l_diagonal=diagonal_l,h_diagonal=diagonal_h,
            padded_length=1<<(len(coefficients)-1).bit_length(),evolution_time=problem.elapsed_time,
            epsilon_he=LCHS_QSP_EPSILON_FRACTION*method.approximation_tolerance,max_bytes=method.max_bytes,
            max_degree=method.max_qsp_degree)
        prepared = prepare_qsp_evolution(tau=selected['layout']['alpha']*problem.elapsed_time,
            epsilon=selected['epsilon_he'],expansion=selected['expansion'],max_bytes=method.max_bytes,
            max_degree=method.max_qsp_degree,max_evaluations=method.max_qsp_evaluations,
            limit_name="LCHS.max_qsp_evaluations")
        selected.update(prepared_evolution=prepared,physical_l_diagonal=diagonal_l,physical_h_diagonal=diagonal_h)
        data = replace(data,qsp_plan=freeze_selected(selected,max_bytes=method.max_bytes))
        # The QSP evolution adds parity, pair and signal qubits to the joint
        # generator's block-encoding ancillas (see native.construct_select).
        extra.update(select_ancillas=selected['layout']['num_ancillas']+3,qsp_recovery=prepared.recovery_scale,
            select_work_remaining=select_work_remaining)
    # Stage 5: coefficient PREP amplitudes and requested MPS decompositions.
    if data.source is None:
        amplitudes = data.coefficient_plan.prep_amplitudes(max_bytes=method.max_bytes)
    else:
        from nwqlib.subroutines.lcu.data import _coefficient_bookkeeping
        # Same per-slot PREP law as LCHSCoefficientPlan.prep_amplitudes.
        _check_bytes(96*(1<<(len(coefficients)-1).bit_length()),method.max_bytes,'source coefficient PREP')
        amplitudes = _coefficient_bookkeeping(np.asarray(coefficients),dimension=dimension,coefficient_atol=0,max_bytes=method.max_bytes)[2]
        amplitudes = _freeze_array(amplitudes)
    extra['amplitudes'] = amplitudes
    if data.source is None and method.initial_state_preparation == 'mps_circuit':
        from nwqlib.subroutines.state_preparation.mps import _decompose_normalized_state
        initial_mps = _decompose_normalized_state(data.initial_direction,max_bond_dim=method.initial_state_mps_max_bond_dim,
            threshold=method.initial_state_mps_threshold,max_bytes=method.max_bytes,max_svd_work=method.max_svd_work)
        data = replace(data,initial_mps=initial_mps)
    if data.source is None and method.lcu_state_preparation == 'mps_circuit' and len(coefficients)>1:
        decomposition = data.coefficient_plan.decompose_mps(max_bond_dim=method.lcu_mps_max_bond_dim,
            threshold=method.lcu_mps_threshold,max_bytes=method.max_bytes,max_svd_work=method.max_svd_work)
        data = replace(data,coefficient_mps=decomposition,
            coefficient_plan=replace(data.coefficient_plan,mps_decomposition=decomposition))
    return replace(data,selected_select=selected_select),extra


def construction_work(data,extra,*,method,elapsed):
    """Return SELECT construction work and known peak bytes, without PREP children.

    elapsed is the Problem's elapsed time T, which each branch of a
    dense_exact SELECT without a source evolves over.

    Returns:
        A pair (work, counts). work is the classical construction work in size
        units. counts holds select_cx, the projected CX count of the SELECT
        leaf including any source-input preparations inside it, and
        known_peak_bytes, the peak known array bytes of that construction.
    """
    dimension = len(data.initial)
    physical = extra['physical_branches']
    coefficients = data.coefficient_plan.coefficients if data.source_layout is None else data.source_layout.coefficients
    a = (len(coefficients)-1).bit_length()
    backend = method.hamiltonian_evolution_backend
    route = method.dense_control_route
    whole = False
    if data.selected_select == 'identity_evolution':
        # Only the actual address phase diagonal remains; source PREP is
        # priced below with its real controls and multiplicity.
        cx, work, peak = max(0,(1<<a)-2), max(1,1<<a), 64*(1<<a)
    elif backend == 'dense_exact':
        from nwqlib.subroutines._dense_synthesis import (controlled_synthesis_size, dense_synthesis_size,
                                                         gatewise_control_counts, gatewise_control_size)
        from .native import _whole_matrix, dense_branch_select_cx
        q = dimension.bit_length()-1
        whole = bool(a) and q >= 2 and _whole_matrix(route, a)
        # One controlled exact synthesis per physical branch. Padding slots
        # have no branch.
        cx = physical*dense_branch_select_cx(q, a, route)
        # Branch matrices from node eigensystems (native._dense_branches):
        # one Hermitian eigensystem per distinct node (one for an exactly
        # zero L, none when L and H are both zero), and for each physical
        # nonzero-time branch the product (V * phase) @ V.conj().T
        # (time_independent_terms._spectral_branch_requirements). A
        # zero-time branch is the identity, D**2 fill.
        from .time_independent_terms import _node_key, _spectral_branch_requirements
        layout = data.source_layout
        quad = data.quadrature
        if layout is None:
            calls = tuple((elapsed, float(k)) for k in quad.k_nodes)
        else:
            calls = tuple((float(layout.elapsed_times[branch]), float(layout.k_values[branch]))
                          for branch in range(len(coefficients)) if layout.branch_to_node[branch] is not None)
        identity = not (np.any(quad.l_part) or np.any(quad.h_part))
        timed = tuple(call for call in calls if call[0] != 0)
        # A source layout caches each distinct node's eigensystem; the
        # homogeneous branches stream one node at a time.
        eigensystems = (0 if identity or not timed else 1 if not np.any(quad.l_part)
                        else len({_node_key(k) for _, k in timed}) if layout is not None else len(timed))
        branch_work, frontier = _spectral_branch_requirements(dimension, eigensystems=eigensystems,
            branches=0 if identity else len(timed))
        work = branch_work + (len(calls)-len(timed) if not identity else len(calls))*dimension**2
        # Live bytes while one branch is formed: L and H (32*D**2), the
        # eigensystems cached across the applications of a source layout
        # (16*D**2 + 8*D each) or the current one, and the pre-synthesis
        # matrix frontier. Branches are formed, controlled and appended one
        # at a time; the kept circuits are added below. The queried LAPACK
        # workspace is not charged (a declared known-array workspace). The
        # construction forms no matrix of the full address-plus-system width.
        cached = eigensystems if layout is not None else min(eigensystems, 1)
        peak = 32*dimension**2 + cached*(16*dimension**2 + 8*dimension) + frontier
        if whole:
            # The whole-matrix route (native._whole_matrix) synthesizes each
            # branch's controlled matrix on q + a qubits with
            # _dense_synthesis.controlled_unitary_circuit, one at a time, and
            # the SELECT keeps the circuits. A constant-source branch first
            # multiplies its evolution by the matrix of its input preparation.
            # Each present preparation's matrix is formed once by applying its
            # at most 4*D elementary one- and two-qubit gates to a D-square
            # array, at most 4*D**2 units each (the magnitude and phase trees
            # of _preparation_laws.direct_preparation_cx_bound have at most
            # D - 1 rotations and D - 2 CX each), and every branch adds one
            # D-square product. The two preparation matrices and one product
            # are held at once, 48*D**2 bytes.
            synthesis_work, synthesis_working, synthesis_kept = controlled_synthesis_size(q+a)
            work += physical*synthesis_work
            peak += synthesis_working + physical*synthesis_kept
            if layout is not None:
                # At most two input kinds, the initial state and the source.
                work += 2*16*dimension**3 + physical*dimension**3
                peak += 48*dimension**2
        elif a:
            # qiskit_compat.controlled replaces each branch's distinct D-square
            # unitary by _dense_synthesis.dense_unitary_circuit, and Qiskit
            # then controls every synthesized gate with the a address bits.
            # The branches are synthesized and controlled one at a time, and
            # their controlled circuits are kept in the SELECT. A one-qubit
            # unitary keeps Qiskit's exact definition, one U gate, which
            # Qiskit controls too. Without address bits the single branch
            # stays a UnitaryGate, which a backend that lowers to a gate basis
            # synthesizes within the Run's max_synthesis_work.
            synthesis_work, synthesis_working, synthesis_kept = (
                dense_synthesis_size(dimension.bit_length()-1) if dimension >= 4 else (0, 0, 0))
            control_work, control_working, control_kept = gatewise_control_size(
                *gatewise_control_counts(dimension.bit_length()-1, a))
            work += physical*(synthesis_work+control_work)
            peak += max(synthesis_working, control_working) + physical*(synthesis_kept+control_kept)
    elif data.select_data is not None:
        from .select_synthesis import _branch_controlled_product_formula_resource_law
        from nwqlib.subroutines._multiplexors import product_formula_select_resource_law
        plan = data.select_data.plan
        law = (_branch_controlled_product_formula_resource_law(plan) if data.selected_select=='branch_controlled'
            else product_formula_select_resource_law(plan,implementation=data.selected_select))
        cx = int(law['select_basis_cx_count'])
        entries = len(plan.occurrence_angle_tables)*plan.padded_node_count
        # Work: read every stored angle, then emit each formula occurrence
        # (a 2**a-angle multiplexor plus about 4*log2(D) basis and parity
        # gates), plus one unit per projected CX.
        work = entries+int(law['select_formula_occurrence_count'])*((1<<a)+4*dimension.bit_length())+cx
        # Bytes: 8 per stored float64 angle, a 64-byte-per-entry envelope for
        # the one 2**a-entry multiplexor table being lowered, and 64*D**2 (four
        # complex128 D-by-D arrays) as a system-register envelope.
        peak = 8*entries+64*(1<<a)+64*dimension**2
    else:
        from nwqlib.subroutines._dense_synthesis import (controlled_synthesis_size, dense_synthesis_size,
                                                         gatewise_control_size)
        from .compiled_selection import (compiled_select_controlled_syntheses, compiled_select_cx_projection,
                                         compiled_select_dense_controls, compiled_select_dense_syntheses)
        cx = compiled_select_cx_projection(data.qsp_plan, route=route)['select_cx']
        # Work: every count of the saved structural law (queries, child calls,
        # multiplexed rotations and angles, projector phases, boundary gates and
        # reflections) summed as a size proxy. Bytes: the
        # same 64-byte-per-entry address-table envelope and eight complex128
        # D-by-D arrays for the system-register child constructions.
        work = sum(data.qsp_plan['structural_law'].values())
        peak = 64*(1<<a)+128*dimension**2
        # A dense-dilation child adds its exact syntheses and Qiskit's control
        # of the synthesized gates, made one at a time and kept in the SELECT.
        sizes = [dense_synthesis_size(qubits) for qubits in compiled_select_dense_syntheses(data.qsp_plan, route=route)]
        sizes += [controlled_synthesis_size(qubits)
                  for qubits in compiled_select_controlled_syntheses(data.qsp_plan, route=route)]
        sizes += [gatewise_control_size(*pair)
                  for pair in compiled_select_dense_controls(data.qsp_plan, route=route)]
        if sizes:
            work += sum(size[0] for size in sizes)
            peak += max(size[1] for size in sizes) + sum(size[2] for size in sizes)
    if data.source_layout is not None and not whole:
        from .compiled_selection import _controlled_direct_preparation_cx
        # Branch-controlled SELECT prepares the input inside every physical
        # branch under the a address controls. The other SELECT wirings
        # prepare each present input kind once, under the kind_control bit
        # when both are present (native._append_source_input). On the
        # whole-matrix route a dense branch includes its preparation in its
        # synthesis, whose CX law already covers it.
        multiplicity = physical if data.selected_select=='branch_controlled' else int(data.source_layout.initial_norm>0)+int(data.source_layout.source_norm>0)
        controls = a if data.selected_select=='branch_controlled' else int(data.source_layout.kind_control is not None)
        cx += multiplicity*_controlled_direct_preparation_cx(dimension.bit_length()-1,controls)
    _check_bytes(peak,method.max_bytes,'LCHS selected native array workspace')
    return work,dict(select_cx=cx,known_peak_bytes=peak)


def selected_error_facts(payload, *, method, problem, output, raw):
    """Publish each selected circuit-path error stage as a planning fact.

    One application record is built per physical application (the
    homogeneous term, or each constant-source application with its own slots
    and synthesis bound). Kernel stages come from the provider record in raw.
    The QSP stage uses the compensated recovery bound of the saved phases and
    child encodings. Direct PREP stages are exact zero, and MPS PREP stages
    are unknown until an explicit circuit validation. gamma is the coefficient
    1-norm, times the initial physical norm for homogeneous evolution.
    Source-branch coefficients already include their input norms. A stage
    whose premise fails is published with a reason instead of a value.
    """
    from .analysis import fact as _fact
    from nwqlib.problems.records import Solution
    from .solution_error_budget import _build_lchs_application_record, _stage_manifest, circuit_stage_bounds, psd_recovery_exponent, psd_recovery_scale
    from .time_independent_terms import _trotter_budget_quadrature_terms, _unusable_kernel_stage_names
    backend = payload.method.hamiltonian_evolution_backend
    frame = Solution().frame(problem)
    coefficients = payload.coefficient_plan.coefficients if payload.source_layout is None else payload.source_layout.coefficients
    layout = payload.source_layout
    specs = ((0.,problem.elapsed_time,problem.initial_state.preparation.physical_scale.as_float(),tuple(range(len(coefficients)))),) if layout is None else layout.applications
    applications = []
    for start,elapsed,weight,slots in specs:
        synthesis = None
        if backend in ("trotter", "trotter_error_budgeted"):
            if layout is None:
                synthesis = payload.select_data.quadrature.get("trotter_synthesis_error_bound")
            else:
                records = [payload.select_records[slot] for slot in slots]
                synthesis = _trotter_budget_quadrature_terms(records,coefficients=payload.quadrature.coefficients,
                    method=payload.method)["trotter_synthesis_error_bound"]
        # A nonrepresentable physical input scale preserves unit-output access;
        # component error evidence remains unavailable rather than infinity.
        if weight is None:
            direct = {"initial_state_preparation":getattr(payload.method,"initial_state_preparation","direct") == "direct",
                      "lcu_coefficient_preparation":getattr(payload.method,"lcu_state_preparation","direct") == "direct"}
            return tuple(_fact(name,frame,0. if direct.get(name,False) else None,
                reason="physical input scale is not representable",source=METHOD)
                for name in _stage_manifest(backend=backend,inhomogeneous=layout is not None,circuit=True))
        applications.append(_build_lchs_application_record(raw,final_time=problem.elapsed_time,start_time=start,
            weight=weight,input_norm=1.,backend=backend,synthesis_error_bound=synthesis))
    # The stored coefficients omit the power of two 2**e of an unrepresentable
    # PSD recovery, so gamma composes 2**e with them. gamma stays
    # representable when a small input norm offsets 2**e, and otherwise its
    # stages are unknown.
    growth = psd_recovery_scale(psd_recovery_exponent(payload.quadrature.conversion["psd_shift"], problem.elapsed_time))
    gamma_scale = compose_recovery(fsum(float(abs(c)) for c in coefficients),
        *(() if layout is not None else (problem.initial_state.preparation.physical_scale,)),
        *(() if growth is None else (growth,)))
    gamma = gamma_scale.as_float()
    qsp_error = 0. if payload.selected_select == "identity_evolution" else None
    if payload.qsp_plan is not None:
        from nwqlib.subroutines.qsp.evolution import qsp_evolution_error_terms
        qsp = payload.qsp_plan
        l,h,pruned_h = qsp["part_plans"]
        # ||D (x) (X' - X)|| <= max|D|*||X' - X|| for each child encoding error,
        # and a fully pruned H child contributes max|D_H| times its dropped mass.
        weights = (qsp["layout"]["l_diagonal_max_abs"],qsp["layout"]["h_diagonal_max_abs"])
        errors = tuple(0. if weight==0 else None if part.error_bound is None else weight*part.error_bound
            for weight,part in zip(weights,(l,h),strict=True))
        child_error = None if None in errors else sum(errors)+max(abs(float(value)) for value in qsp["physical_h_diagonal"])*pruned_h
        if qsp["prepared_evolution"] is not None:
            qsp_error = qsp_evolution_error_terms(qsp["prepared_evolution"],evolution_time=problem.elapsed_time,
                child_error_bound=child_error)["compensated_recovery_error_bound"]
    values = circuit_stage_bounds(raw,applications=tuple(applications),backend=backend,inhomogeneous=layout is not None,
        psd_premise_satisfied=payload.quadrature.numerical_psd_premise_satisfied,gamma=gamma,
        delta_lcu=0. if getattr(payload.method,"lcu_state_preparation","direct") == "direct" else None,
        input_preparation_output_error_bound=0. if getattr(payload.method,"initial_state_preparation","direct") == "direct" else None,
        compensated_recovery_error_bound=qsp_error,
        duhamel_quadrature_error_bound=0. if layout is not None and (layout.source_norm==0 or
            payload.selected_select=='identity_evolution' and payload.quadrature.conversion['psd_shift']==0
            and not (np.any(payload.quadrature.l_part) or np.any(payload.quadrature.h_part))) else None,
        unusable_stages=_unusable_kernel_stage_names(raw))
    return tuple(_fact(name,frame,value,reason=reason or "selected component unavailable",source=METHOD)
                 for name,(value,reason) in values.items())
