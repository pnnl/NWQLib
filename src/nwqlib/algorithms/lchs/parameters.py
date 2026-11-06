"""Planned product-formula, QSP and MPS data and the component formulas that LCHS execution uses."""

from nwqlib._limits import ResourceLimitError, exceeds
from dataclasses import fields, is_dataclass, replace
from collections.abc import Mapping
from hashlib import sha256
import struct
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from nwqlib.core.records import FrozenArray
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import OperatorInput, _freeze_array
from nwqlib.problems.inputs import PhysicalScale, StateInput, compose_recovery
from .primary_records import METHOD


# Scalars that freeze_selected keeps unchanged, by exact type. Subclasses,
# such as NumPy scalars, take the full checks.
_PLAIN_SCALARS = (int, float, complex, str, bool)


def freeze_selected(value, *, max_bytes):
    """Make read-only copies of the planned numerical arrays once. Metadata does no numerical work.

    Writable arrays are copied read-only and their bytes are checked against
    max_bytes first. Already frozen arrays are shared, not copied, so a Plan
    holds one immutable copy of each planned array.
    """
    copied = 0
    def freeze(item):
        """Return item with every writable array replaced by a read-only copy.

        Containers are rebuilt recursively. Mappings become MappingProxyType,
        lists become tuples, dataclasses are rebuilt from their frozen fields
        and NumPy scalars become Python scalars. None, plain int, float,
        complex, str and bool, and the immutable operator and state inputs
        pass unchanged. Any other type raises, so no arbitrary object enters
        a Plan. A PhysicalScale is immutable and also passes unchanged.
        """
        nonlocal copied
        if item is None or type(item) in _PLAIN_SCALARS:
            # The common leaves, ahead of the container and array checks they fail.
            return item
        if isinstance(item, (FrozenArray, OperatorInput, StateInput, PhysicalScale)):
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
        raise TypeError(f"unsupported LCHS selected numerical data: {type(item).__name__}")
    return freeze(value)


def _feed_selected_value(digest, value):
    """Feed one frozen planned value into ``digest`` as a prefix-free encoding.

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
    - An operator or state input is the sequence ``("input", representation,
      identity)``. Its identity already hashes its data (or names a supplied
      circuit), so those data are not fed again.
    - ``r``: any other value that ``freeze_selected`` accepts (int, str, bool,
      None or a PhysicalScale, whose repr shows its exact mantissa, integer
      exponent and evidence), as its type name and repr, ended by ``;``. The
      repr of a string escapes its quote character, so the terminator is
      unambiguous.

    NumPy scalars are fed as the equal Python scalar.

    The encoding is collected in one buffer, which is written to ``digest``
    before each array's bytes and at the end, so ``digest`` receives the
    same byte stream in fewer updates.
    """
    buffer = bytearray()
    _encode_selected_value(digest, buffer, value)
    digest.update(buffer)


_COUNT = struct.Struct('<Q')
_FLOAT = struct.Struct('<d')
_COMPLEX = struct.Struct('<dd')
# Per dataclass type: its ``D`` header (type name and field count) and each
# field name with its ``r`` encoding. A dataclass's fields are fixed by its class.
_DATACLASS_HEADERS = {}


def _encode_selected_value(digest, out, value):
    """Append the ``_feed_selected_value`` encoding of ``value`` to ``out``.

    Exact float, int, str, bool, None, tuple, list and complex values take
    the branch that the type checks below select for them. An array writes
    ``out`` and then its own bytes to ``digest``, keeping the stream order.
    """
    kind = type(value)
    if kind is float:
        out += b'f' + _FLOAT.pack(value)
    elif value is None or kind is str or kind is int or kind is bool:
        out += b'r' + (kind.__name__ + ':' + repr(value) + ';').encode()
    elif kind is tuple or kind is list:
        out += b'S' + _COUNT.pack(len(value))
        for child in value:
            _encode_selected_value(digest, out, child)
    elif kind is complex:
        out += b'c' + _COMPLEX.pack(value.real, value.imag)
    else:
        if isinstance(value, FrozenArray):
            value = value.array
        if isinstance(value, (OperatorInput, StateInput)):
            _encode_selected_value(digest, out, ("input", value.reference.representation, value.reference.identity))
        elif isinstance(value, np.ndarray):
            out += b'A' + (str((value.dtype.str, value.shape)) + ';').encode()
            if value.size:
                digest.update(out)
                out.clear()
                digest.update(memoryview(value).cast("B"))
        elif isinstance(value, Mapping):
            out += b'M' + _COUNT.pack(len(value))
            for key in sorted(value):
                _encode_selected_value(digest, out, key)
                _encode_selected_value(digest, out, value[key])
        elif is_dataclass(value):
            header = None if isinstance(value, type) else _DATACLASS_HEADERS.get(type(value))
            if header is None:
                members = fields(value)
                prefix = bytearray(b'D')
                _encode_selected_value(digest, prefix, type(value).__qualname__)
                prefix += _COUNT.pack(len(members))
                names = []
                for field in members:
                    encoded = bytearray()
                    _encode_selected_value(digest, encoded, field.name)
                    names.append((field.name, bytes(encoded)))
                header = bytes(prefix), tuple(names)
                if not isinstance(value, type):
                    _DATACLASS_HEADERS[type(value)] = header
            out += header[0]
            for name, encoded in header[1]:
                out += encoded
                _encode_selected_value(digest, out, getattr(value, name))
        elif isinstance(value, (tuple, list)):
            out += b'S' + _COUNT.pack(len(value))
            for child in value:
                _encode_selected_value(digest, out, child)
        elif isinstance(value, np.generic):
            _encode_selected_value(digest, out, value.item())
        else:
            out += b'r' + (type(value).__name__ + ':' + repr(value) + ';').encode()


def selected_identity(data, problem, method):
    """Return the content hash that ties the planned numerical data to its Problem and Method.

    The SHA-256 covers the content hashes of the Problem and Method and every
    planned array, record and scalar in LCHSData, with dtype, shape and exact
    binary64 bytes, encoded by ``_feed_selected_value``. The SELECT record or
    classical routine saves this hash to identify the planned numerical data.
    """
    if data.method != method:
        raise ValueError("selected LCHS data belongs to another Method")
    digest = sha256((problem.content_id+method.content_id).encode())
    _feed_selected_value(digest, (
        data.generator, data.initial_state, data.source_state, data.dense,
        data.coefficient_plan, data.quadrature, data.select_data, data.select_records,
        data.source_layout, data.qsp_plan, data.initial_mps, data.coefficient_mps,
        data.selected_select, data.host_actions, data.applications))
    return 'sha256:'+digest.hexdigest()


# Share of the combined algorithmic evolution allowance that funds the
# Jacobi-Anger truncation tail; the rest funds phase synthesis. Any share in
# (0, 1) keeps tail + zeta*epsilon_synth within the allowance (Laurent and
# GQET: epsilon_synth = (A - tail)/zeta; shared QSVT: the tail leaves A/2 for
# the fixed per-parity phase residual). One half is a valid simple choice,
# not an optimum: the degree depends on the tail only logarithmically.
COHERENT_TAIL_SHARE = 0.5

# Margin eta of the analytic kernel PREP target |q| <= 1 - eta (M3P witness value, PREP_IMPLEMENTATION_POLICY
# section 6); its G19 + G20 budget is LCHS_QSP_EPSILON_FRACTION * approximation_tolerance, the component
# fraction of the coherent evolution. Both are declared controls, not optima (ENGINEERING_CONSTANTS, M8).
ANALYTIC_PREP_MARGIN = 0.25


def _admit_mps_circuit(size, max_bond_dim, num_layers, method):
    """Admit the whole MPS circuit of a size-entry tensor before its TT-SVD (mps.admit_mps_circuit).

    The TT-SVD and the circuit construction of its route are compared once
    with an explicit ``LCHS.max_svd_work``, so lowering the selected cores
    cannot refuse an accepted Plan.
    """
    from nwqlib.subroutines.state_preparation.mps import admit_mps_circuit
    admit_mps_circuit(max(size-1, 0).bit_length(), max_bond_dim=max_bond_dim, num_layers=num_layers,
                      max_svd_work=method.max_svd_work)


def select_parameters(method, problem, data):
    """Choose every numerical SELECT and PREP quantity once, before any circuit exists.

    This function makes the planning choices that follow once the coefficient
    grid is known.
    It resolves the SELECT implementation, lays out constant-source branches,
    builds the product-formula plan, and computes the
    coefficient PREP amplitudes and any requested MPS decompositions. The
    results are frozen into LCHSData, whose content hash (selected_identity)
    is recorded in the SELECT record, so execution, archive loading and
    verification reuse these exact numbers and never choose them again.
    The qsp_block_encoding backend chooses its coherent route in
    ``quantum.choose_provider`` and prepares it in `_select_coherent`
    instead.

    A dense_exact plan whose final address-slot count exceeds
    max_dense_select_slots is refused here, before circuit construction. A
    composite Gauss grid whose node count alone proves that outcome was
    refused earlier, before its rule was built
    (``providers._build_composite_gauss``). Each
    dense branch is a controlled, classically computed matrix exponential, so
    construction cost and memory grow with both the address-slot count and
    the system dimension. The refusal names the choices the caller can change. It never
    lowers the k-quadrature, the Duhamel node count or the tolerance to fit,
    because that would silently change the planned approximation.

    Args:
        method: Configured LCHS Method whose choices and limits apply.
        problem: LinearDynamics with its ingested physical input scales.
        data: LCHSData from select_dense, holding the homogeneous coefficient plan.

    Returns:
        A pair of the frozen LCHSData with its SELECT data and a dict of
        planned values for reconstruction and resource formulas: the
        physical branch count, PREP amplitude array and, where applicable,
        step counts and Duhamel nodes and weights.
    """
    from .native import resolve_lcu_select_implementation
    from .time_independent_terms import generate_lchs_product_formula_select_plan, _build_product_formula_select_plan, _trotter_pauli_decomposition, _admit_pauli_decomposition, LCHSProductFormulaSelectData
    from .solution_error_budget import signed_shift
    backend = data.backend
    generator = data.generator
    dimension = generator.encoded_dimension
    # Stage 1: provisional SELECT route. Product-formula routes are resolved
    # again below, once their plan and its structure certificate exist.
    selected_select = resolve_lcu_select_implementation(data.method, backend=backend)
    extra = dict(physical_branches=len(data.coefficient_plan.coefficients))
    # Stage 2: constant-source branch layout (initial and Duhamel applications).
    layout = None
    if data.source_state is not None:
        from .source_selection import select_source_branches
        layout = select_source_branches(coefficients=data.quadrature.coefficients,k_nodes=data.quadrature.k_nodes,
            initial_scale=problem.initial_state.preparation.physical_scale,
            source_scale=problem.source.preparation.physical_scale,final_time=problem.elapsed_time,
            shift=signed_shift(data.quadrature.conversion),duhamel_nodes=method.duhamel_nodes,
            scalar_phase=(generator.dissipative_extracted,generator.hamiltonian_extracted),
            compact=selected_select=='branch_controlled',drop_allowance=method.source_drop_allowance,
            max_bytes=method.max_bytes,max_quadrature_work=method.max_quadrature_work)
        data = replace(data,source_layout=layout)
        extra.update(physical_branches=layout.branch_count,source_nodes=layout.source_nodes,source_weights=layout.source_weights)
    # Stage 3: refuse an oversized dense SELECT before any construction.
    # A source layout keeps its coefficient state as amplitudes and phases
    # (source_selection.SourceBranchLayout), which also serve the LCU
    # intake's register and limit checks.
    coefficients = data.coefficient_plan.coefficients if layout is None else layout.amplitudes[:len(layout.phases)]
    padded_slots = 1 << (len(coefficients)-1).bit_length()
    if backend=='dense_exact' and padded_slots>method.max_dense_select_slots:
        from .providers import dense_select_slot_refusal
        raise dense_select_slot_refusal(time=problem.elapsed_time,l_norm=data.quadrature.l_norm,
            tolerance=method.approximation_tolerance,branches=len(coefficients),limit=method.max_dense_select_slots,
            total_qubits=(len(coefficients)-1).bit_length()+dimension.bit_length()-1)
    if selected_select == 'branch_controlled':
        # native.construct_select passes these coefficients through the LCU
        # intake before it builds any branch. dense_exact always resolves to
        # branch_controlled, and a product-formula request keeps it, so the
        # same call here refuses at planning a Plan that construction would
        # refuse.
        from nwqlib.subroutines.lcu.data import lcu_coefficient_intake
        lcu_coefficient_intake(coefficients,system_dimension=dimension,max_bytes=method.max_bytes,
            max_work=method.max_select_work)
    # Stage 4: branch evolution (product-formula tables).
    if backend in ('trotter','trotter_error_budgeted'):
        if layout is None:
            ledger = {}
            dense = data.dense_inputs()
            selected = generate_lchs_product_formula_select_plan(final_time=problem.elapsed_time,
                method=data.method,quadrature=data.quadrature,parts=(dense.l_part,dense.h_part),
                max_steps=method.max_trotter_steps,max_bytes=method.max_bytes,max_select_work=method.max_select_work,
                work_ledger=ledger)
            extra['select_work_charged'] = ledger['work']
        else:
            # The homogeneous planner's decomposition admission, with the
            # plan receiving the rest of the same work limit.
            charged, remaining_work = _admit_pauli_decomposition(dimension,max_bytes=method.max_bytes,
                max_select_work=method.max_select_work)
            dense = data.dense_inputs()
            decomposition = _trotter_pauli_decomposition(dense.l_part,dense.h_part)
            ledger = {}
            # Binary64 slot coefficients relative to 2**scale_exponent, for
            # the recorded layout summary only. A slot below the binary64
            # range is zero here. The SELECT phases, PREP and recovery read
            # the layout's phases, amplitudes and 1-norm instead, and each
            # application's synthesis stage reads the raw coefficients
            # (selected_error_facts).
            slot_coefficients = (layout.coefficient_l1_norm*np.square(layout.amplitudes[:len(layout.phases)])
                                 *np.exp(1j*np.asarray(layout.phases)))
            pf,records,nodes = _build_product_formula_select_plan(decomposition=decomposition,k_values=layout.k_values,
                elapsed_times=layout.elapsed_times,coefficients=slot_coefficients,coefficient_phases=layout.phases,
                branch_to_node=layout.branch_to_node,
                method=data.method,address_structure=data.quadrature.coefficient_plan.quadrature.address_structure,
                max_steps=method.max_trotter_steps,
                max_bytes=method.max_bytes,max_select_work=remaining_work,charged_before=charged,
                grid_k_values=data.quadrature.k_nodes,work_ledger=ledger)
            extra['select_work_charged'] = charged+ledger['work']
            from .time_independent_terms import _trotter_budget_quadrature_terms
            summary = _trotter_budget_quadrature_terms(records,coefficients=slot_coefficients,method=data.method)
            selected = LCHSProductFormulaSelectData(coefficients=slot_coefficients,plan=pf,quadrature=summary,
                numerical_psd_premise_satisfied=data.quadrature.psd_premise_satisfied,nodes=nodes)
            data = replace(data,select_records=records)
        selected_select = resolve_lcu_select_implementation(data.method,backend=backend,select_plan=selected.plan)
        data = replace(data,select_data=freeze_selected(selected,max_bytes=method.max_bytes))
        extra['step_counts'] = selected.plan.branch_step_counts
    # Stage 5: coefficient PREP amplitudes and requested MPS decompositions.
    if data.source_state is None:
        amplitudes = data.coefficient_plan.prep_amplitudes(max_bytes=method.max_bytes)
    else:
        # The layout computed its normalized amplitudes once, within its
        # byte check (source_selection.select_source_branches).
        amplitudes = _freeze_array(np.asarray(layout.amplitudes, dtype=np.complex128))
    extra['amplitudes'] = amplitudes
    if data.source_state is None and method.initial_state_preparation == 'mps_circuit':
        from nwqlib.subroutines.state_preparation.mps import _decompose_normalized_state
        direction = data.dense_inputs().initial_direction
        _admit_mps_circuit(direction.size, method.initial_state_mps_max_bond_dim, method.initial_state_mps_num_layers,
                           method)
        initial_mps = _decompose_normalized_state(direction,
            max_bond_dim=method.initial_state_mps_max_bond_dim,threshold=method.initial_state_mps_threshold,
            max_bytes=method.max_bytes,max_svd_work=None)
        data = replace(data,initial_mps=initial_mps)
    if data.source_state is None and method.lcu_state_preparation == 'mps_circuit' and len(coefficients)>1:
        _admit_mps_circuit(1 << (len(coefficients)-1).bit_length(), method.lcu_mps_max_bond_dim, method.mps_num_layers,
                           method)
        decomposition = data.coefficient_plan.decompose_mps(max_bond_dim=method.lcu_mps_max_bond_dim,
            threshold=method.lcu_mps_threshold,max_bytes=method.max_bytes,max_svd_work=None)
        data = replace(data,coefficient_mps=decomposition,
            coefficient_plan=replace(data.coefficient_plan,mps_decomposition=decomposition))
    return replace(data,selected_select=selected_select),extra


class CoherentContext(NamedTuple):
    """Facts of one coherent planning pass that do not depend on the provider grid, built once (`coherent_context`).

    Every provider candidate of the pass reads them, so the Duhamel rule,
    the child compilation and the input-preparation laws run once per pass.
    The grid-dependent facts stay per candidate: the quadrature summary and
    its kernel stage bounds (``circuit_stage_bounds``), the application
    records that it consumes, and the PREP, lift, omission, node
    normalization, schedules, recovery and error facts of each row.

    Attributes:
        nodes: Gauss-Legendre Duhamel nodes of a nonzero source, else empty.
        weights: Their weights.
        facts: Unomitted ``LCHSApplicationFacts`` of the kernel alternative,
            masses ``w_a nu_a exp(s t_a)``.
        children: ``(l_child, h_child)``, the compiled centered children, None
            for an exact zero; both None for the single-A form or an exactly
            zero evolution.
        a_child: The supplied encoding of a general A (single-A form), else None.
        omitted: Operator-norm bound of a centered part that relative pruning
            removed entirely, else 0.
        charged: Work that compiling the children drew from ``max_select_work``.
        state_preparation_cx: CX of the kind-state preparation R, None when unknown.
        mps_initial: Whether an MPS initial state is priced by its layer law.
        duhamel_bound: The source-quadrature stage (`_duhamel_bound`).
        filter_facts: Application facts of the imaginary-time filter or
            commuting product (masses ``w_a nu_a kappa_a``), or None when
            that alternative does not apply.
        filter_alpha: Its contraction normalization ``alpha_L + delta_L``, or None.
    """

    nodes: tuple
    weights: tuple
    facts: Any
    children: tuple
    a_child: Any
    omitted: float
    charged: int
    state_preparation_cx: int | None
    mps_initial: bool
    duhamel_bound: Any
    filter_facts: Any = None
    filter_alpha: float | None = None


def coherent_context(method, problem, common):
    """Return the `CoherentContext` of one coherent planning pass from its provider-independent facts.

    ``common`` is the ``selection.LCHSCommon`` of the pass. The finite
    solution is the factorized source LCU of ``coherent_selection``:
    u_fin = sum_{a,j} m_a c_j exp(-i t_a (k_j L_0 + H_0)) psi_a with masses
    m_a = w_a nu_a exp(s t_a) (``application_facts``). The masses, the
    Gauss-Legendre Duhamel nodes (``duhamel_quadrature``), the centered
    children of ``compiled_selection`` (relative pruning for a dense A, whose
    fully pruned H keeps its mass as ``omitted``), the kind-state
    preparation CX laws and the source-quadrature bound read only the
    generator, the inputs, the elapsed time and the signed shift s, never
    the kernel or quadrature. The unitary-or-dissipative branch of the child
    compilation is decided by the common ``l_norm``, not by any provider grid.

    When H is an exact multiple of the identity, or the representation gives
    an exact [L, H] = 0, and L has a centered child of known error, the
    imaginary-time filter or commuting product is another finite target
    without a kernel. With L = c_L I + L_0 and the contraction normalization
    alpha = alpha_L + delta_L of the centered child,
    exp(-t_a (L + iH)) = kappa_a exp(-i t_a c_H) exp[-beta (1 - X_a)] with
    beta = T_e alpha and kappa_a = exp[t_a (alpha - c_L)] (AT8-AT13, M7.5),
    so its masses are w_a nu_a kappa_a and no PSD shift is needed. These
    facts are grid-independent too; the alternative's other stages are not.
    An analytic kernel PREP request excludes that alternative, which has no
    kernel register.

    Raises:
        ValueError: If the Duhamel rule exceeds ``max_bytes`` or
            ``max_quadrature_work``, or a child exceeds its limits.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators.access import _check_bytes as check_bytes
    from .compiled_selection import (_compiled_qsp_compact_part_plans, _compiled_qsp_part_plans,
                                     _controlled_direct_preparation_cx, _controlled_state_preparation_cx)
    from .inhomogeneous_theory import duhamel_quadrature
    from .solution_error_budget import signed_shift
    from .source_selection import application_facts
    generator, final_time = common.generator, problem.elapsed_time
    shift = signed_shift(common.conversion)
    nodes = weights = ()
    if common.source_state is not None and common.source_state.preparation.physical_scale.mantissa > 0:
        # Gauss-Legendre rule of the Duhamel integral: the tridiagonal rule and
        # node mapping of select_source_branches, without its A x M slots.
        s = method.duhamel_nodes
        check_bytes(128*s, method.max_bytes, "constant-source Duhamel quadrature")
        if exceeds(s**2+8*s, method.max_quadrature_work):
            raise ResourceLimitError(f"constant-source Duhamel quadrature needs {s**2+8*s} work units (s**2+8*s with "
                             f"s={s}), LCHS.max_quadrature_work={method.max_quadrature_work}. Raise "
                             f"LCHS.max_quadrature_work to at least {s**2+8*s}.")
        nodes, weights = (tuple(float(value) for value in rule) for rule in duhamel_quadrature(final_time, s))
    source_norm = None if common.source_state is None else common.source_state.preparation.physical_scale
    facts = application_facts(kind="homogeneous" if common.source_state is None else "discrete_source",
        final_time=final_time, signed_shift=shift,
        scalars=(generator.dissipative_extracted-shift, generator.hamiltonian_extracted),
        initial_norm=problem.initial_state.preparation.physical_scale, source_norm=source_norm,
        source_nodes=nodes, source_weights=weights)
    # An MPS initial state is priced by its layer law and decomposed only after the selection (_select_coherent).
    mps_initial = common.dense is not None and method.initial_state_preparation == 'mps_circuit'
    # Centered children. An exactly zero L + s*I (unitary) leaves H the only
    # child, as compiled_selection orders them; exact zeros are omitted.
    children, omitted, charged = (None, None), 0.0, 0
    # A general A given only as one supplied encoding is the single-A child (G13): its encoding of A_0 = A
    # (no identity is extracted, so c_L = c_H = 0) and the kernel coefficients k_j - i give k_j L + H. The
    # operator input is kept, which the archive saves with its circuit; the joint owner reads its encoding.
    a_child = generator.single_a
    if not generator.zero_evolution and a_child is None:
        unitary = common.l_norm == 0
        if common.dense is None:
            first, second, omitted, charged = _compiled_qsp_compact_part_plans(generator, unitary=unitary,
                max_bytes=method.max_bytes, max_select_work=method.max_select_work)
        else:
            parts = (common.dense.h_part, common.dense.l_part) if unitary else (common.dense.l_part, common.dense.h_part)
            first, second, omitted, charged = _compiled_qsp_part_plans(*parts,
                l_norm=None if unitary else common.l_norm,
                max_bytes=method.max_bytes, max_select_work=method.max_select_work)
        children = tuple(None if plan.implementation == 'exact_zero' else plan
                         for plan in ((second, first) if unitary else (first, second)))
    # R prepares each input kind once, controlled on the kind bit when both are present.
    kinds = sorted({item.kind for item in facts.applications})
    controls, q = int(len(kinds) == 2), generator.num_qubits

    def preparation_cx(kind):
        if common.dense is None:
            return _controlled_state_preparation_cx(common.initial_state if kind == "initial" else common.source_state,
                                                    controls)
        if kind == "initial" and mps_initial:
            # Each layer has at most q - 1 two-qubit unitaries of at most three CX (select_vector_preparation).
            return 3*method.initial_state_mps_num_layers*max(0, q-1)
        return _controlled_direct_preparation_cx(q, controls)

    costs = [preparation_cx(kind) for kind in kinds]
    filter_facts = alpha = None
    # An exact [L, H] = 0 of the representation (LCHSGenerator.parts_commute); no commutator is computed.
    commuting = generator.parts_commute is not None and children[1] is not None
    if (children[0] is not None and (children[1] is None or commuting) and not omitted
            and method.lcu_state_preparation != "analytic"):
        from nwqlib.subroutines.block_encoding.joint import access_summary
        summary = access_summary(children[0])
        c_l = generator.dissipative_extracted-shift
        alpha = None if summary.error_bound is None else float(np.nextafter(summary.alpha+summary.error_bound, np.inf))
        try:
            filter_facts = None if alpha is None else application_facts(kind=facts.kind, final_time=final_time,
                signed_shift=alpha-c_l, scalars=(c_l, generator.hamiltonian_extracted),
                initial_norm=problem.initial_state.preparation.physical_scale, source_norm=source_norm,
                source_nodes=nodes, source_weights=weights)
        except ApplicabilityError:
            # kappa_a is outside the representable range; the kernel alternative remains.
            filter_facts = None
    return CoherentContext(nodes=nodes, weights=weights, facts=facts, children=children, a_child=a_child,
        omitted=omitted, charged=charged, state_preparation_cx=None if None in costs else sum(costs),
        mps_initial=mps_initial, duhamel_bound=_duhamel_bound(common, problem), filter_facts=filter_facts,
        filter_alpha=alpha)


def coherent_selection(method, problem, data, *, raw, context, acquisition, readout=None, sampling=None):
    """Choose the coherent route of one provider grid by scalar work only, without any preparation.

    The finite solution is the factorized source LCU of
    ``coherent_selection``: u_fin = sum_{a,j} m_a c_j exp(-i t_a (k_j L_0 + H_0))
    psi_a with masses m_a = w_a nu_a exp(s t_a) (``application_facts``), one
    evolution exp(-i T_e S) of the joint generator for the largest elapsed
    time T_e and time factors q_a = t_a/T_e, the PREP amplitudes and the
    slot phases arg(c_j) - t_a (k_j (c_L + s) + c_H). The dense recipe
    evolves L + s I and H, so c_L + s = c_H = 0 there.

    The grid-independent inputs come from ``context`` (`coherent_context`).
    The kernel nodes and raw coefficients come from ``data``'s quadrature,
    and the kernel, k-quadrature, Duhamel and input-preparation stage bounds
    of ``raw`` give ``rest_error``. The objective is the algorithmic frame:
    LCHS_QSP_EPSILON_FRACTION * approximation_tolerance is the combined
    allowance of the truncation tail and phase synthesis, a component
    target and not a total physical guarantee; native gate error stays
    unknown. ``acquisition`` ranks zeta**2 * C for accepted states, C for
    exact readout or fixed shots, and the sufficient quantum work N G C of
    ``sampling`` for an accuracy request (``CoherentObjective``).

    The imaginary-time filter or commuting product of the context is a
    second alternative (``ImaginaryTimeAlternative`` or
    ``CommutingProductAlternative``) whose other stages exclude the kernel
    integral and the k quadrature. ``LCHS.qsp_route`` maps to the coherent
    request (``_coherent_request``). With
    ``lcu_state_preparation="mps_circuit"`` both alternatives price their
    magnitude factors as layered MPS circuits, and an MPS initial state is
    priced by its layer law; no decomposition happens here.

    The same inputs give the same row, so an accuracy request runs this in
    two deterministic passes (``LCHS.sampling_shots``, then planning) with no
    phase synthesis, MPS decomposition, circuit or random draw.

    Returns:
        ``(selection, facts_by_alternative)``: the ``CoherentSelection`` and
        the application facts of each alternative.
    """
    from math import fsum
    from nwqlib.algorithms.protocol import ApplicabilityError
    from .coherent_selection import (CoherentControls, CoherentObjective, KernelGrid, LCHSCoherentAlternative,
                                     select_coherent_route)
    from .native import LCHS_QSP_EPSILON_FRACTION
    from .solution_error_budget import circuit_stage_bounds
    from .time_independent_terms import _unusable_kernel_stage_names
    mps_layers = method.mps_num_layers if method.lcu_state_preparation == 'mps_circuit' else None
    facts, children = context.facts, context.children
    rule = data.quadrature.coefficient_plan.quadrature
    uniform = rule.address_structure.get("kind") == "symmetric_uniform_trapezoid_with_identity_padding"
    kernel = KernelGrid(nodes=tuple(float(k) for k in data.quadrature.k_nodes),
        coefficients=tuple(complex(c) for c in data.quadrature.coefficients),
        half_width=rule.interval_count_each_side if uniform else None,
        spacing=float(rule.base_weights[0]) if uniform else None)
    # The application records read only the PSD shift, the time and the inputs; the stage bounds read this grid's
    # provider bounds, so both are formed per grid.
    records = _application_records(raw, problem=problem, backend=data.backend,
        specs=tuple((item.start_time, item.weight, item.input_norm, None) for item in facts.applications))
    stages = circuit_stage_bounds(raw, applications=records, backend=data.backend,
        inhomogeneous=data.source_state is not None, psd_premise_satisfied=data.quadrature.psd_premise_satisfied,
        gamma=None, delta_lcu=0., input_preparation_output_error_bound=None if context.mps_initial else 0.,
        duhamel_quadrature_error_bound=context.duhamel_bound, unusable_stages=_unusable_kernel_stage_names(raw))
    # The other stages of the physical vector; the direct coefficient PREP
    # contributes 2*gamma*0 and the evolution is the selected row's own term.
    rest = [value for name, (value, _) in stages.items() if name not in ("qsp_synthesis", "lcu_coefficient_preparation")]
    alternatives = [LCHSCoherentAlternative(label="lchs", applications=facts, kernel=kernel, l_child=children[0],
        h_child=children[1], a_child=context.a_child, rest_error=None if None in rest else fsum(rest),
        state_preparation_cx=context.state_preparation_cx, drop_allowance=method.source_drop_allowance,
        omitted_part_bound=context.omitted, coefficient_mps_layers=mps_layers,
        analytic_prep=_analytic_settings(method, data) if method.lcu_state_preparation == "analytic" else None)]
    facts_by_alternative = {"lchs": facts}
    if context.filter_facts is not None:
        from .coherent_selection import CommutingProductAlternative, ImaginaryTimeAlternative
        kernel_free = [value for name, (value, _) in stages.items()
                       if name not in ("qsp_synthesis", "lcu_coefficient_preparation", "kernel_approximation",
                                       "k_quadrature")]
        common = dict(applications=context.filter_facts, l_child=children[0], alpha=context.filter_alpha,
            rest_error=None if None in kernel_free else fsum(kernel_free),
            state_preparation_cx=context.state_preparation_cx,
            drop_allowance=method.source_drop_allowance, coefficient_mps_layers=mps_layers)
        if children[1] is None:
            alternatives.append(ImaginaryTimeAlternative(label="filter", **common))
        else:
            alternatives.append(CommutingProductAlternative(label="product", h_child=children[1],
                                                            commutation=data.generator.parts_commute, **common))
        facts_by_alternative[alternatives[-1].label] = context.filter_facts
    request = _coherent_request(method.qsp_route)
    selection = select_coherent_route(alternatives, objective=CoherentObjective(acquisition=acquisition,
            allowance_frame="algorithmic", allowance=LCHS_QSP_EPSILON_FRACTION*method.approximation_tolerance,
            tail_share=COHERENT_TAIL_SHARE, readout=readout, sampling=sampling),
        controls=CoherentControls(max_bytes=method.max_bytes, max_node_work=method.max_select_work),
        request=request)
    if selection.selected is None:
        # The factorized PREP calls the omission allowance drop_allowance; the Method option is
        # source_drop_allowance.
        hint = ("; set LCHS.source_drop_allowance to cover the omitted applications"
                if "drop_allowance" in selection.reason else "")
        raise ApplicabilityError(f"no coherent LCHS route is admitted within the limits: {selection.reason}{hint}")
    if selection.selected.route == "zero_solution":
        raise ValueError("coherent LCHS selected the exact zero solution, which plan_lchs routes to plan_initial")
    return selection, facts_by_alternative


def _select_coherent(method, problem, choice, *, shots=None):
    """Prepare the selected row of a coherent choice once and synthesize only its phases.

    ``choice`` is the ``quantum.CoherentChoice`` of this planning pass: the
    chosen grid's data, its ``CoherentSelection``, the application facts of
    each alternative and the pass context. No row is selected again. For an
    accuracy request the selected row's sufficient count must equal
    ``shots``, which ``LCHS.sampling_shots`` returned from the same
    deterministic choice, before anything is prepared. Then exactly the
    selected target is synthesized; a selected synthesis that fails raises
    ``CoherentPreparationError`` with its work and cause, and no other row or
    grid is tried. After the route is fixed, an explicit MPS initial state is
    decomposed once (``initial_mps``), and with
    ``lcu_state_preparation="mps_circuit"`` each nontrivial factor of the
    selected PREP is decomposed once (``coefficient_mps``, kernel then
    application).

    Returns:
        The LCHSData with ``qsp_plan`` (the prepared route without its
        preparation handles) and ``applications``, and the planned values
        that ``quantum.plan_coherent_lcu`` reads.
    """
    from .coherent_selection import prepare_selected_coherent_route, without_preparation_handles
    data, selection, context = choice.data, choice.selection, choice.context
    row = selection.selected
    if selection.objective.sampling is not None and row.cost.get("shots") != shots:
        raise ValueError(f"actual shots {shots} differ from the accuracy-selected allocation "
                         f"{row.cost.get('shots')} of {row.label}")
    prepared = prepare_selected_coherent_route(selection)
    facts = choice.facts[row.alternative]
    preparation = prepared.preparation
    mps_layers = method.mps_num_layers if method.lcu_state_preparation == 'mps_circuit' else None
    initial_mps = None
    if data.dense is not None and method.initial_state_preparation == 'mps_circuit':
        from nwqlib.subroutines.state_preparation.mps import _decompose_normalized_state
        _admit_mps_circuit(data.dense.initial_direction.size, method.initial_state_mps_max_bond_dim,
                           method.initial_state_mps_num_layers, method)
        initial_mps = _decompose_normalized_state(data.dense.initial_direction,
            max_bond_dim=method.initial_state_mps_max_bond_dim,threshold=method.initial_state_mps_threshold,
            max_bytes=method.max_bytes,max_svd_work=None)
    coefficient_mps = None
    if mps_layers is not None and preparation is not None:
        # Explicit MPS coefficient PREP (MPS_PREP_POLICY): one TT-SVD of each selected nontrivial magnitude
        # factor, in its selected label order, after the route is fixed; a factor without qubits is the identity.
        from nwqlib.subroutines.state_preparation.mps import _decompose_normalized_state
        factors = ((preparation.kernel_amplitudes, preparation.kernel_qubits),
                   (preparation.application_amplitudes, preparation.application_qubits))
        for amplitudes, width in factors:
            if width:
                _admit_mps_circuit(len(amplitudes), method.lcu_mps_max_bond_dim, mps_layers, method)
        coefficient_mps = tuple(None if not width else _decompose_normalized_state(
            np.asarray(amplitudes, dtype=np.complex128), max_bond_dim=method.lcu_mps_max_bond_dim,
            threshold=method.lcu_mps_threshold, max_bytes=method.max_bytes, max_svd_work=None)
            for amplitudes, width in factors)
    data = replace(data, initial_mps=initial_mps, coefficient_mps=coefficient_mps,
        qsp_plan=freeze_selected(without_preparation_handles(prepared), max_bytes=method.max_bytes),
        applications=freeze_selected(facts, max_bytes=method.max_bytes))
    retained = len(facts.applications)-len(preparation.omitted_applications)
    nodes_per_application = len(data.quadrature.k_nodes) if row.alternative == "lchs" else 1
    # A commuting product keeps the independent success workspaces of its two factors.
    plans = prepared.evolution_plan if isinstance(prepared.evolution_plan, tuple) else (prepared.evolution_plan,)
    # An analytic kernel PREP keeps its work register W from P_c to P_c^dagger (build_source_lcu_circuit).
    work_qubits = 0 if prepared.kernel_preparation is None else prepared.kernel_preparation.plan.work_qubits
    return data, dict(evolution_ancillas=sum(plan.num_ancillas for plan in plans if plan is not None),
        preparation_work_qubits=work_qubits, physical_branches=retained*nodes_per_application,
        select_work_charged=context.charged, source_nodes=context.nodes, source_weights=context.weights)


def _analytic_settings(method, data):
    """Return the analytic kernel PREP settings of ``LCHSCoherentAlternative.analytic_prep`` for the selected grid.

    The analytic PREP (``subroutines.block_encoding.analytic_prep``)
    prepares ``f(k_j)/sqrt(M p_f)``, ``f(k) = exp(-k**2/(8 gamma**2))/sqrt(1 + k**2)``,
    which is the kernel factor ``sqrt(|c_j|/beta_k)`` only for the Low-Somma
    pair: its coefficients are ``h f_hat_2(k_j)/sqrt(2 pi)`` with equal
    weights h (``providers._low_somma_provider_coefficient``), and
    ``|f_hat_2(k)| = sqrt(2/pi) exp(c - 1/(4 gamma**2)) f(k)**2``, so
    ``|c_j|/beta_k = f(k_j)**2/sum_l f(k_l)**2`` with the profile's gamma
    (G18). The phase ``arg c_j = -k_j c`` stays in the slot phase diagonal.
    Any other pair, unequal weights or the unitary reduction is refused
    before any row is planned.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from .native import LCHS_QSP_EPSILON_FRACTION
    plan = data.quadrature.coefficient_plan
    pair = (plan.resolved_lchs_kernel.implementation, plan.resolved_k_quadrature.implementation)
    if pair[0] == "unitary_reduction":
        raise ApplicabilityError("this A has L proportional to the identity, so LCHS reduces to one unitary and has no "
                                 "kernel state for lcu_state_preparation='analytic' to prepare; use "
                                 "lcu_state_preparation='direct'")
    if pair != ("low_somma_f2", "symmetric_uniform_trapezoid") or len(set(plan.base_weights)) != 1:
        raise ApplicabilityError("lcu_state_preparation='analytic' prepares the Low-Somma kernel state on its "
                                 "equal-weight trapezoid grid; use lchs_kernel='low_somma_f2' with "
                                 f"k_quadrature='symmetric_uniform_trapezoid', not {pair[0]!r} with {pair[1]!r}")
    return dict(gamma=float(plan.derived_parameters["gamma"]),
                approximation_tolerance=LCHS_QSP_EPSILON_FRACTION*method.approximation_tolerance,
                margin=ANALYTIC_PREP_MARGIN, max_bytes=method.max_bytes)


def _coherent_request(route):
    """Map ``LCHS.qsp_route`` to the coherent request; ``"auto"`` without parameters is no request."""
    from .coherent_selection import CoherentRequest
    parameters = dict(route.parameters)
    constraints = dict(route=None if route.implementation == "auto" else route.implementation,
                       level=parameters.get("level"), joint=parameters.get("joint"), node=parameters.get("node"))
    if all(value is None for value in constraints.values()):
        return None
    return CoherentRequest(**constraints)


def construction_work(data,extra,*,method,elapsed):
    """Return SELECT construction work and known peak bytes, without PREP children.

    elapsed is the Problem's elapsed time T, which each branch of a
    dense_exact SELECT without a source evolves over.

    Returns:
        A pair (work, counts). work is the classical construction work in size
        units. counts holds select_cx, the projected CX count of the SELECT
        block including any source-input preparations inside it, and
        known_peak_bytes, the declared peak allowance for the numerical arrays
        and circuit data of that construction.
    """
    dimension = data.generator.encoded_dimension
    if data.selected_select == 'coherent_lcu':
        return _coherent_construction_work(data, method=method)
    physical = extra['physical_branches']
    coefficients = data.coefficient_plan.coefficients if data.source_layout is None else data.source_layout.phases
    a = (len(coefficients)-1).bit_length()
    backend = data.backend
    route = method.dense_control_route
    whole = False
    if backend == 'dense_exact':
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
        dense = data.dense_inputs()
        identity = data.generator.zero_evolution
        timed = tuple(call for call in calls if call[0] != 0)
        # A source layout caches each distinct node's eigensystem. The
        # homogeneous branches stream one node at a time.
        eigensystems = (0 if identity or not timed else 1 if not np.any(dense.l_part)
                        else len({_node_key(k) for _, k in timed}) if layout is not None else len(timed))
        branch_work, frontier = _spectral_branch_requirements(dimension, eigensystems=eigensystems,
            branches=0 if identity else len(timed))
        work = branch_work + (len(calls)-len(timed) if not identity else len(calls))*dimension**2
        # Live bytes while one branch is formed: L and H (32*D**2), the
        # eigensystems cached across the applications of a source layout
        # (16*D**2 + 8*D each) or the current one, and the pre-synthesis
        # matrix frontier. Branches are formed, controlled and appended one
        # at a time. The kept circuits are added below. The queried LAPACK
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
            # synthesizes and the Run records.
            synthesis_work, synthesis_working, synthesis_kept = (
                dense_synthesis_size(dimension.bit_length()-1) if dimension >= 4 else (0, 0, 0))
            control_work, control_working, control_kept = gatewise_control_size(
                *gatewise_control_counts(dimension.bit_length()-1, a))
            work += physical*(synthesis_work+control_work)
            peak += max(synthesis_working, control_working) + physical*(synthesis_kept+control_kept)
    else:
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
    if data.source_layout is not None and not whole:
        from .compiled_selection import _controlled_direct_preparation_cx, _controlled_state_preparation_cx
        # Branch-controlled SELECT prepares the input inside every physical
        # branch under the a address controls. The other SELECT wirings
        # prepare each present input kind once, under the kind_control bit
        # when both are present (native._append_source_input). On the
        # whole-matrix route a dense branch includes its preparation in its
        # synthesis, whose CX law already covers it. The dense recipe
        # prepares padded vector directions. A compact recipe prepares each
        # state with its own construction, whose controlled CX is unknown
        # for a supplied circuit.
        layout = data.source_layout
        controls = a if data.selected_select=='branch_controlled' else int(layout.kind_control is not None)
        if data.selected_select=='branch_controlled':
            kinds = [layout.branch_kinds[layout.branch_to_node[slot]] for slot in range(len(coefficients))
                     if layout.branch_to_node[slot] is not None]
        else:
            kinds = sorted(set(layout.branch_kinds))
        states = dict(initial=data.initial_state, source=data.source_state)
        for kind in kinds:
            prep = (_controlled_direct_preparation_cx(data.generator.num_qubits, controls) if data.dense is not None
                    else _controlled_state_preparation_cx(states[kind], controls))
            cx = None if cx is None or prep is None else cx+prep
    _check_bytes(peak,method.max_bytes,'LCHS selected native array workspace')
    return work,dict(select_cx=cx,known_peak_bytes=peak)


def _coherent_construction_work(data, *, method):
    """Return the construction work and known peak bytes of the coherent LCU body, and its CX estimate.

    Work is a size proxy for the limit check, in the convention of the
    compiled QSP law: every count of the selected row's census (original-U
    calls, phase slots, flag toggles, reflections, processing rotations and
    lifted X gates), the stored PREP amplitudes and phases, one unit per
    coefficient-address slot, and the work of each input preparation (its
    StatePreparationSpec work, max(1, q)*D for a dense direct direction, or
    the admitted layered work of an MPS initial state), plus the table work
    of every compact-child, node-recipe and table query that the generator
    queries build, with its actual controls and adjoint (_table_queries),
    the exact syntheses of the dense-dilation children
    (_dense_child_syntheses), for an explicit MPS coefficient PREP each
    factor's layered construction, and for an analytic kernel PREP its MP8
    operation counts and phases. Since the emitters check each of those
    queries against max_select_work, a Plan whose work this total admits
    builds them all. Bytes: 64 per address slot plus eight complex128
    D-by-D arrays for the dense recipe, or otherwise the bytes of the
    largest of those table queries (each query is also checked against
    max_bytes, as its builder checks it), plus those syntheses' working and
    kept allowances and the MPS factors' cores and larger construction
    workspace.
    select_cx is the selected row's attempt CX bound (PREP, its inverse, the
    slot phases, R and the evolution), None when unknown. When a supplied
    operation's definition is not built, the work is None, ``unresolved``
    gives the reason and the known peak excludes the synthesis of that
    definition (_table_queries).
    """
    from .coherent_selection import evolution_factors
    prepared = data.qsp_plan
    preparation, factors = prepared.preparation, evolution_factors(prepared)
    address = preparation.kernel_qubits+preparation.application_qubits
    q, dimension = data.generator.num_qubits, data.generator.encoded_dimension
    work = preparation.data_entries + (1 << address)
    for census in ([prepared.candidate.census] if not factors else [factor.census for factor in factors]):
        work += (sum(count for *_, count in census.get("rows", ())) + sum(count for _, count in census.get("helper_x", ()))
                 + sum(value for key, value in census.items() if key not in ("rows", "helper_x")))
    for kind in sorted({kind for kind in preparation.application_kinds if kind}):
        if data.dense is None:
            work += (data.initial_state if kind == "initial" else data.source_state).preparation.work
        elif kind == "initial" and data.initial_mps is not None:
            from nwqlib.subroutines.state_preparation.mps import admit_layered_construction
            work += admit_layered_construction(data.initial_mps.bond_dimensions, method.initial_state_mps_num_layers,
                max_svd_work=None, max_bytes=method.max_bytes)[0]
        else:
            work += max(1, q)*dimension
    # The emitters pass max_select_work and max_bytes to every table query that they build (native.py), each
    # with its actual controls and adjoint, so the queries' work counts here, each query's bytes are checked as
    # its builder checks them, and the largest query stands for the compact children's tables in the peak
    # (_table_queries).
    query_work, query_bytes, unresolved = _table_queries(prepared, max_bytes=method.max_bytes)
    work += query_work
    peak = 64*(1 << address) + (128*dimension**2 if data.dense is not None else query_bytes)
    synthesis_work, synthesis_bytes = _dense_child_syntheses(prepared)
    work += synthesis_work
    peak += synthesis_bytes
    # The evolution emitters complete each dense-dilation child once
    # (joint._complete_dense_children): a full SVD and the completion, 9 d**3
    # work for d = 2**n, and the built 2d x 2d unitary is held while the
    # queries are built.
    for factor in factors:
        for child in factor.generator.components.values():
            if getattr(child, "implementation", None) == "dense_dilation":
                d = 1 << child.system_qubits
                work += 9*d**3
                peak += 64*d*d
    if prepared.kernel_preparation is not None:
        # The analytic kernel PREP: its MP8 operation counts (analytic_prep._census at the solved degree and
        # rounds) and its stored phases. build_analytic_kernel_query checks its gate copies against max_bytes
        # itself, so they are not part of this known peak.
        analytic = prepared.kernel_preparation
        work += len(analytic.phases.phases) + sum(
            value for key, value in analytic.census["forward"].items()
            if key != "qubits" and not key.endswith("cx") and isinstance(value, int))
    if data.coefficient_mps is not None:
        # Each MPS factor's layered construction (admit_layered_construction) is built once; both factors' cores
        # are held, with the larger construction workspace.
        from nwqlib.subroutines.state_preparation.mps import admit_layered_construction
        built = [admit_layered_construction(item.bond_dimensions, method.mps_num_layers,
                 max_svd_work=None, max_bytes=method.max_bytes)
                 for item in data.coefficient_mps if item is not None]
        work += sum(size for size, _ in built)
        peak += sum(core.nbytes for item in data.coefficient_mps if item is not None for core in item.cores)
        peak += max((size for _, size in built), default=0)
    _check_bytes(peak, method.max_bytes, 'LCHS selected native array workspace')
    # An uncounted supplied query synthesis leaves the work unresolved, never its known part (DESIGN: an unknown
    # cost never becomes zero), and the known peak then excludes the synthesis of that definition.
    return (None if unresolved else work), dict(select_cx=prepared.candidate.cost.get("attempt_cx"),
                                                known_peak_bytes=peak, unresolved=unresolved)


def _query_builds(factor):
    """Return the (controls, adjoint) forms of the generator queries that the emitter builds for one evolution factor.

    The emitter builds one generator query per distinct form of the
    selected row's census rows (of the raw circuit for shared OAA, whose
    other two uses repeat or reverse that built circuit), plus a second
    control state of the controlled query when a direct-involution walk has
    inverse signals (D_- is active on the other processing value).
    """
    from nwqlib.subroutines.qsp.evolution import _shared_census
    rows = factor.census.get("rows", ())
    if factor.route.startswith("shared"):
        rows = _shared_census(factor.degrees["ladder"], factor.signal_access == "generic_lift", "raw", 0)
    builds = sorted({(controls, adjoint) for _, controls, adjoint, _ in rows})
    if factor.signal_access == "direct_involution" and factor.degrees.get("negative_signals"):
        builds.append((1, False))
    return builds


def _table_queries(prepared, *, max_bytes):
    """Return the work, the largest bytes and any unresolved part of the queries that the selected evolution's generator queries build.

    Each generator query of form (c, a) (``_query_builds``) builds the
    queries of its compact children, supplied encodings, node recipes and
    tables with their actual controls and adjoint flags, which
    ``joint._component_query_requirements`` reads with the laws that their
    builders check against max_select_work and max_bytes. Each query is
    also checked against ``max_bytes`` here, as its builder will check it.
    A supplied operation whose definition is not built leaves its part of
    the query synthesis uncounted, and the third value gives the reason;
    the dense unitaries outside such definitions are still counted and
    checked.
    """
    from nwqlib.subroutines.block_encoding.joint import _component_query_requirements
    from .coherent_selection import evolution_factors
    work = largest = 0
    unresolved = None
    for factor in evolution_factors(prepared):
        for controls, adjoint in _query_builds(factor):
            data, query_work, missing = _component_query_requirements(factor.generator, num_controls=controls,
                                                                      adjoint=adjoint, max_bytes=max_bytes)
            work, largest, unresolved = work + query_work, max(largest, data), unresolved or missing
    return work, largest, unresolved


def _dense_child_syntheses(prepared):
    """Return the work and bytes of the exact dense syntheses that the selected evolution's queries make.

    Each generator query of form (c, a) (``_query_builds``) builds every child call of the
    generator's census
    (forward_calls for c = 0, controlled_calls for c = 1), with the child's
    adjoint flag flipped when a = 1. build_block_encoding_query synthesizes
    a dense_dilation child exactly and lets Qiskit control the synthesized
    gates whenever the call has k >= 1 controls, and holds that charge twice
    for an adjoint: dense_synthesis_size(w) plus
    gatewise_control_size(gatewise_control_counts(w, k)) on the child's
    w = n + 1 qubits, the reservation admit_dense_syntheses checks there.
    Work adds over all builds; bytes are the largest working allowance plus
    every kept allowance, since the built queries are held together.
    """
    from nwqlib.subroutines._dense_synthesis import (dense_synthesis_size, gatewise_control_counts,
                                                     gatewise_control_size)
    from .coherent_selection import evolution_factors
    sizes = []
    # A commuting product builds the queries of each of its two evolutions.
    for factor in evolution_factors(prepared):
        generator = factor.generator
        dense = {name: child for name, child in generator.components.items()
                 if getattr(child, "implementation", None) == "dense_dilation"}
        if not dense:
            continue
        for controls, adjoint in _query_builds(factor):
            calls = generator.query_census["forward_calls" if controls == 0 else "controlled_calls"]
            for name, child_controls, child_adjoint, count in calls:
                child = dense.get(name)
                if child is None or not child_controls:
                    continue
                width = child.system_qubits + child.num_ancillas
                copies = 2 if child_adjoint != adjoint else 1
                each = (dense_synthesis_size(width),
                        gatewise_control_size(*gatewise_control_counts(width, child_controls)))
                sizes += count * copies * list(each)
    if not sizes:
        return 0, 0
    return sum(size[0] for size in sizes), max(size[1] for size in sizes) + sum(size[2] for size in sizes)


def _application_records(raw, *, problem, backend, specs):
    """Return one LCHSApplicationRecord per (start, weight, input norm, synthesis bound) application."""
    from .solution_error_budget import _build_lchs_application_record
    return tuple(_build_lchs_application_record(raw, final_time=problem.elapsed_time, start_time=start,
                 weight=weight, input_norm=norm, backend=backend, synthesis_error_bound=synthesis)
                 for start, weight, norm, synthesis in specs)


def _duhamel_bound(payload, problem):
    """Return the source-quadrature stage of a constant-source Plan, None when unavailable or without a source.

    For a compact generator it is the Gauss-Legendre remainder
    (``inhomogeneous_theory._duhamel_quadrature_error_bound``) with the
    original bound B_A of ``LCHSGenerator.generator_norm``, the original
    lower bound of L and the source PhysicalScale, so a declared lower bound
    of L makes it conditional on every route. The dense recipe leaves
    it unavailable, because its ||A|| is computed only by an explicit
    refinement.
    """
    from .inhomogeneous_theory import _duhamel_quadrature_error_bound
    generator = payload.generator
    if problem.source is None:
        return None
    source_scale = problem.source.preparation.physical_scale
    if source_scale.mantissa == 0 or generator.exact_zero:
        return 0.
    if generator.generator_norm is None:
        return None
    return _duhamel_quadrature_error_bound(generator_norm=generator.generator_norm,
        lower_bound=generator.dissipative_facts.lower, source_scale=source_scale,
        final_time=problem.elapsed_time, node_count=payload.method.duhamel_nodes)


def selected_error_facts(payload, *, problem, raw):
    """Report each error stage of the planned circuit path as a planning fact.

    One application record is built per physical application (the
    homogeneous term, or each constant-source application with its own
    address slots and synthesis bound). Kernel stages come from the
    kernel-quadrature record in raw.
    The QSP stage uses the compensated recovery bound of the saved phases and
    child encodings. Direct PREP stages are exact zero, and MPS PREP stages
    are unknown until an explicit circuit validation. gamma is the coefficient
    1-norm, times the initial physical norm for homogeneous evolution.
    The source layout keeps its magnitudes relative to its common scale
    2**scale_exponent, and the homogeneous coefficients relative to
    2**recovery_exponent, which gamma composes back. A stage whose
    assumption fails is reported with a reason instead of a value.
    Component bounds scale by exp(psd_shift*t) with the nonnegative PSD
    shift, which is at least the exact factor exp(s*t) under known decay,
    so they stay upper bounds.

    On the coherent route (``selected_select == "coherent_lcu"``) the
    applications are ``payload.applications``, gamma is the recovery of the
    factorized PREP (the application masses times the kernel one-norm), and
    qsp_synthesis is gamma times the prepared route's algorithmic and child
    terms in the recovered frame. Native gate error is not part of this
    component; the error model keeps it in its unknown native term. When the
    H = 0 imaginary-time filter or the commuting product is selected, the
    kernel and k-quadrature stages are exactly zero, because that finite sum
    contains no kernel integral.

    Input weights w_a*nu_a, gamma, the product-formula bound
    sum_j |c_j|*e_j of the raw coefficients (``_trotter_budget_quadrature_terms``)
    and the source-quadrature bound stay PhysicalScale values until each
    stage is complete, and
    ``solution_error_budget`` converts each stage to binary64 once. A
    positive stage above the binary64 range is therefore unknown with the
    reason ``nonfinite_stage``, a positive stage below it is rounded up to
    a positive binary64 bound, never zero, and a stage whose large and
    small factors multiply to a representable value is reported.

    The source-quadrature stage is ``_duhamel_bound``. Slots or applications
    omitted by the range limit add their bound as the separate component
    ``source_range_omission``.
    """
    from .analysis import fact as _fact
    from nwqlib.problems.records import Solution
    from .solution_error_budget import circuit_stage_bounds, pruning_stage, psd_recovery_scale
    from .time_independent_terms import (_trotter_budget_quadrature_terms, _trotter_pruning_bound,
        _unusable_kernel_stage_names, stored_pf_nodes)
    backend = payload.backend
    frame = Solution().frame(problem)
    layout = payload.source_layout
    dropped = None
    pruning = []
    if payload.selected_select == "coherent_lcu":
        prepared = payload.qsp_plan
        specs = tuple((item.start_time,item.weight,item.input_norm,None) for item in payload.applications.applications)
        gamma = PhysicalScale(mantissa=prepared.physical_recovery[0], exponent=prepared.physical_recovery[1])
        terms = (prepared.error["algorithmic"], prepared.error["child"])
        qsp_error = None if None in terms else sum(terms)
        if prepared.preparation.dropped_bound is not None:
            dropped = PhysicalScale(mantissa=prepared.preparation.dropped_bound[0],
                                    exponent=prepared.preparation.dropped_bound[1])
    else:
        if layout is None:
            l1_norm, exponent = payload.coefficient_plan.coefficient_l1_norm, payload.coefficient_plan.recovery_exponent
            slots_by_application = (tuple(range(len(payload.coefficient_plan.coefficients))),)
            applications = ((0.,1.,problem.initial_state.preparation.physical_scale),)
        else:
            l1_norm, exponent = layout.coefficient_l1_norm, layout.scale_exponent
            slots_by_application = layout.application_slots
            applications = tuple((item.start_time,item.weight,item.input_norm) for item in layout.applications)
            dropped = layout.dropped_bound
        specs = []
        for (start,weight,norm),slots in zip(applications,slots_by_application,strict=True):
            synthesis = None
            if backend in ("trotter", "trotter_error_budgeted"):
                if layout is None:
                    synthesis = payload.select_data.quadrature.get("trotter_synthesis_error_bound")
                else:
                    records = [payload.select_records[slot] for slot in slots]
                    synthesis = _trotter_budget_quadrature_terms(records,coefficients=payload.quadrature.coefficients,
                        method=payload.method)["trotter_synthesis_error_bound"]
            if backend == "trotter":
                if layout is None:
                    nodes = stored_pf_nodes(payload)
                    masses = [nodes.nodes[int(index)]["pruned_l1_mass"] for index in nodes.grid_to_node]
                else:
                    masses = [record.pruned_l1_mass for record in records]
                pruning.append(_trotter_pruning_bound(masses,coefficients=payload.quadrature.coefficients,
                    elapsed_time=problem.elapsed_time-start))
            specs.append((start,weight,norm,synthesis))
        # The stored coefficients omit the power of two 2**e of the homogeneous
        # compensation or of the source layout's common scale, so gamma composes
        # 2**e with them and stays a PhysicalScale.
        growth = psd_recovery_scale(exponent)
        gamma = compose_recovery(l1_norm,
            *(() if layout is not None else (problem.initial_state.preparation.physical_scale,)),
            *(() if growth is None else (growth,)))
        qsp_error = None
    application_records = _application_records(raw,problem=problem,backend=backend,specs=specs)
    values = circuit_stage_bounds(raw,applications=application_records,
        backend=backend,inhomogeneous=payload.source_state is not None,
        psd_premise_satisfied=payload.quadrature.psd_premise_satisfied,gamma=gamma,
        delta_lcu=0. if getattr(payload.method,"lcu_state_preparation","direct") == "direct" else None,
        input_preparation_output_error_bound=0. if getattr(payload.method,"initial_state_preparation","direct") == "direct" else None,
        compensated_recovery_error_bound=qsp_error,duhamel_quadrature_error_bound=_duhamel_bound(payload,problem),
        unusable_stages=_unusable_kernel_stage_names(raw))
    if backend == "trotter":
        values = {**values, "trotter_pruning": pruning_stage(application_records,pruning,
            psd_premise_satisfied=payload.quadrature.psd_premise_satisfied)}
    if payload.selected_select == "coherent_lcu" and payload.qsp_plan.candidate.alternative != "lchs":
        # The imaginary-time filter and the commuting product evolve exp(-t_a (L + iH)) itself: their finite sum
        # has no kernel integral and no k quadrature, so those stages contribute exactly zero on these routes.
        values = {**values, "kernel_approximation": (0., None), "k_quadrature": (0., None)}
    if payload.selected_select == "coherent_lcu" and payload.qsp_plan.kernel_preparation is not None:
        # PREP and its inverse each contribute one state error delta_k of the analytic kernel PREP, so 2 gamma
        # delta_k is a known part of the stage (MATH_TRIAGE Q3, E9). delta_k omits the PREP's unknown emitter,
        # uniform-support and full-gate terms (analytic_prep.prepare_analytic_kernel), so the stage stays unknown.
        from .solution_error_budget import _stage_product
        kernel = payload.qsp_plan.kernel_preparation
        known = _stage_product("lcu_coefficient_preparation", 2.0, gamma, kernel.kernel_error).as_float()
        values = {**values, "lcu_coefficient_preparation": (None,
            f"analytic kernel PREP: known part 2 gamma delta_k = {known!r} ({kernel.qualification})")}
    premise = payload.quadrature.conversion["psd_premise_evidence"]
    facts = tuple(_fact(name,frame,value,reason=reason or "selected component unavailable",source=METHOD,premise=premise)
                  for name,(value,reason) in values.items())
    if dropped is not None:
        from .source_selection import omission_upper_float
        facts += (_fact("source_range_omission",frame,omission_upper_float(dropped),
                        reason="omitted source contribution bound is not representable",source=METHOD,premise=premise),)
    return facts
