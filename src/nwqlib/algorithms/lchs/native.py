"""Circuit constructors for the blocks of an LCHS Program.

They read the numerical data chosen at planning and never select it again.
"""

from __future__ import annotations

from nwqlib.blocks.selection import _no_arguments
from typing import Any

# Algorithmic allowance of the coherent evolution, as a fraction of
# approximation_tolerance, in the recovered evolution frame: the truncation
# tail and the phase synthesis together (parameters._select_coherent splits
# it). It is a component target, not a guarantee that the returned physical
# solution meets approximation_tolerance. Available component bounds keep
# their own premises and scale through the actual LCU.
LCHS_QSP_EPSILON_FRACTION = 0.1

# Additional per-node product-formula allowance as a fraction of approximation_tolerance,
# with the same scope as above. Both fractions are registered in
# ENGINEERING_CONSTANTS.
LCHS_TROTTER_EPSILON_FRACTION = 0.1


def resolve_lcu_select_implementation(method, *, backend: str, select_plan: Any | None = None) -> str:
    """Resolve homogeneous or inhomogeneous SELECT synthesis without a circuit.

    ``backend`` is the resolved Hamiltonian-evolution backend of the
    selected data (``LCHSData.backend``), never the Method's ``"auto"``.
    Decision table, by that backend:

    - qsp_block_encoding: always ``coherent_lcu``, the factorized source LCU
      around one coherent evolution (parameters._select_coherent). Any
      explicit request other than ``auto`` raises.
    - dense_exact: ``branch_controlled``, the only construction for classically
      exponentiated branches. ``multiplexor`` and ``structured`` raise.
    - trotter_error_budgeted: ``multiplexor`` for ``auto``, otherwise the
      explicit request. ``structured`` raises, because branch-dependent step
      counts break the affine address structure.
    - trotter: with an eligible structure certificate, ``auto`` chooses
      ``structured`` only when its projected basis-CX count is strictly lower
      than the multiplexor's. Without a plan or certificate, ``auto`` gives
      ``multiplexor``. An explicit ``structured`` request is kept provisionally
      before a plan exists and raises once a plan lacks an eligible
      certificate. ``multiplexor`` and ``branch_controlled`` requests are
      used as given.

    The planner calls this once before the product-formula plan exists and
    again with the plan (parameters.select_parameters).

    Returns:
        The resolved implementation name.
    """

    requested = method.lcu_select_implementation
    he_backend = backend
    # Eligibility is a construction property. Dense, QSP and variable-step
    # product formulas expose different SELECT implementations.
    if he_backend == "qsp_block_encoding":
        if requested != "auto":
            raise ValueError(
                "hamiltonian_evolution_backend='qsp_block_encoding' uses its "
                "internal coherent LCU; lcu_select_implementation must be 'auto'"
            )
        return "coherent_lcu"
    if he_backend == "dense_exact":
        if requested == "multiplexor":
            raise ValueError(
                "lcu_select_implementation='multiplexor' requires an elementary "
                "product-formula plan, which dense_exact does not provide"
            )
        if requested == "structured":
            raise ValueError(
                "lcu_select_implementation='structured' requires a structure "
                "certificate, which dense_exact does not provide"
            )
        return "branch_controlled"
    if he_backend == "trotter_error_budgeted":
        if requested == "structured":
            raise ValueError(
                "lcu_select_implementation='structured' failed eligibility condition: "
                "fixed-step product formula is required; trotter_error_budgeted is ineligible"
            )
        return "multiplexor" if requested == "auto" else requested
    # trotter
    certificate = (
        select_plan.structure_certificate
        if select_plan is not None and requested in ("auto", "structured")
        else None
    )
    if certificate is None:
        if requested == "structured" and select_plan is not None:
            raise ValueError(
                "lcu_select_implementation='structured' failed eligibility condition: "
                "no structure certificate is available"
            )
        return "multiplexor" if requested == "auto" else requested
    rejection_reasons = tuple(certificate.get("rejection_reasons", ()))
    eligible = bool(certificate.get("eligible", False)) and not rejection_reasons
    if requested == "structured":
        if not eligible:
            details = "; ".join(rejection_reasons) or "certificate is not eligible"
            raise ValueError(
                "lcu_select_implementation='structured' failed eligibility "
                f"condition(s): {details}"
            )
        return "structured"
    if not eligible:
        return "multiplexor"
    from nwqlib.subroutines._multiplexors import (
        product_formula_select_resource_law,
    )

    # Compare costs only after a valid structure certificate exists. Auto
    # chooses structured SELECT only for a strict reduction in the same CX basis.
    multiplexor_cx = int(product_formula_select_resource_law(select_plan)["select_basis_cx_count"])
    structured_cx = int(
        product_formula_select_resource_law(select_plan, implementation="structured")["select_basis_cx_count"]
    )
    return "structured" if structured_cx < multiplexor_cx else "multiplexor"




def construct_preparation(block, arguments, method_context):
    """Build the circuit of one planned PREP block from its saved tensor and optional MPS cores.

    Direct preparation builds the exact normalized state with the direct
    magnitude/phase tree. Its record declares a ``qiskit.direct``
    preparation, so the Run counts its tree among the direct preparations
    of each circuit that calls it before that circuit's charge
    (``lowering._count_definition``). MPS preparation
    reuses the cores chosen at planning, so circuit construction never repeats the TT-SVD.
    """
    _no_arguments(arguments)
    target, decomposition, layers, max_bytes, max_svd_work = block._payload
    if decomposition is None:
        from nwqlib.subroutines.state_preparation.direct import direct_preparation_circuit
        return direct_preparation_circuit(target, register_name="state")
    from nwqlib.subroutines.state_preparation.mps_circuit import build_mps_circuit_state_preparation
    return build_mps_circuit_state_preparation(target,
        max_bond_dim=decomposition.max_bond_dim, threshold=decomposition.threshold,
        num_layers=layers, _selected_decomposition=decomposition,
        max_bytes=max_bytes, max_svd_work=max_svd_work).circuit


def _source_preparations(data):
    """Return the PREP circuits of the unit initial and source directions that SELECT applies.

    These are built inside SELECT and are not declared as PREP blocks, so
    the Run's check of direct preparations does not count them. For the dense recipe each is
    the exact direct tree of a padded direction, which fills the system
    register of padded dimension D that ``selection.select_dense`` checks at
    ``64*D**2 + 128*D`` bytes before planning continues, against ``max_bytes``
    or, by default, the physical memory. When the dissipative part L is
    nonzero, an explicit ``max_spectral_work`` W also requires ``D**3 <= W``
    for the unpadded dimension. A compact generator prepares
    each state with its own construction from ``prepare_qiskit``: the direct
    tree of a vector, one gate per qubit of a product or occupation state,
    or the supplied circuit with its declared global phase, which control
    turns into the relative phase between the initial and source branches.
    """
    from nwqlib.subroutines.state_preparation.direct import _build_normalized_state_preparation
    result = {}
    if data.dense is not None:
        for kind, direction in (("initial", data.dense.initial_direction), ("source", data.dense.source_direction)):
            if direction is not None:
                result[kind] = _build_normalized_state_preparation(direction, input_norm=1., register_name="system").circuit
        return result
    return state_preparations(data.initial_state, data.source_state, max_bytes=data.method.max_bytes)


def state_preparations(initial_state, source_state, *, max_bytes):
    """Return the PREP circuits of the nonzero unit initial and source directions of compact inputs, by kind.

    Each state uses its own construction from ``prepare_qiskit``: the direct
    tree of a vector, one gate per qubit of a product or occupation state,
    or the supplied circuit with its declared global phase. A caller that
    controls these circuits turns that global phase into the relative phase
    between initial and source branches. Neither tree has a size check here
    (`_source_preparations`).
    """
    from nwqlib.problems.inputs import prepare_qiskit
    from nwqlib.subroutines.state_preparation.direct import _build_normalized_state_preparation
    result = {}
    for kind, state in (("initial", initial_state), ("source", source_state)):
        if state is None or state.preparation.physical_scale.mantissa == 0:
            continue
        if state.preparation.implementation == "qiskit.direct":
            result[kind] = _build_normalized_state_preparation(state._direction, input_norm=1.,
                                                               register_name="system").circuit
        else:
            result[kind] = prepare_qiskit(state, max_bytes=max_bytes).circuit
    return result


def construct_select(block, arguments, method_context):
    """Build the circuit of the planned LCHS SELECT block from its saved LCHSData, without choosing anything again.

    SELECT applies sum_j |j><j| (x) e^{i*arg(c_j)} U_j on the coefficient
    address and system registers. The coefficient magnitudes live in the
    separate PREP pair, so PREP, SELECT and inverse PREP form the LCU of ACL
    arXiv:2312.03916v2, Appendix A.3, Lemma 24, with the complex phases moved
    from the two state-preparation oracles into SELECT. The block follows the
    planned route. dense_exact controls classically exponentiated branches,
    and product formulas use the saved multiplexed or structured tables. The
    qsp_block_encoding backend builds its whole LCU body in
    construct_coherent_lcu instead. Branch-controlled product formulas and
    dense source branches control each branch circuit separately. With a
    constant source, input preparation is inside SELECT because initial-state
    branches start from u0 and Duhamel branches from b. Padded layouts apply
    one preparation per input kind, controlled by the layout's kind_control
    address bit when both kinds are present. Branch-controlled and dense
    source branches prepare the input inside each branch.
    """
    _no_arguments(arguments)
    data, elapsed = block._payload
    from qiskit import QuantumCircuit
    from .select_synthesis import (
        _build_product_formula_branch,
        admit_branch_controlled_select,
        build_product_formula_select,
    )
    from .solution_error_budget import unit_angle
    backend = data.backend
    dimension = data.generator.encoded_dimension
    layout = data.source_layout
    # A source layout keeps its coefficient state as amplitudes and phases,
    # because its coefficients need not be representable in binary64. The
    # amplitudes serve the LCU intake's register and limit checks, and the
    # phases are applied as given.
    coefficients = data.coefficient_plan.coefficients if layout is None else layout.amplitudes[:len(layout.phases)]
    phases = None if layout is None else layout.phases

    def lcu_intake():
        # The LCU coefficient intake admits its own bytes and work (owned by
        # subroutines/lcu/core.py). The branch-controlled routes run it before
        # they compute any branch exponential or branch circuit, so a refused
        # coefficient table costs no branch construction.
        from nwqlib.subroutines.lcu.core import prepare_lcu_gate_data
        return prepare_lcu_gate_data(coefficients, system_dimension=dimension,
            max_bytes=data.method.max_bytes, max_work=data.method.max_select_work)

    if backend == "dense_exact" and data.source_layout is None:
        lcu = lcu_intake()
        # Control the branch circuit, not a raw UnitaryGate, whose control
        # would have Qiskit build and check the full address-plus-system
        # matrix of every branch. qiskit_compat.controlled synthesizes the
        # small branch unitary exactly (subroutines/_dense_synthesis.py)
        # before it adds the controls and the coefficient phase. The
        # branches stream: each is formed, controlled and appended before
        # the next is formed.
        return build_branch_select(lcu, _dense_branches(data, ((float(k), elapsed, None)
            for k in data.quadrature.k_nodes)), route=data.method.dense_control_route)
    preparations = {} if data.source_layout is None else _source_preparations(data)
    if backend != "dense_exact" and data.selected_select != "branch_controlled":
        select = build_product_formula_select(data.select_data.plan, num_system_qubits=dimension.bit_length()-1,
            implementation=data.selected_select, max_bytes=data.method.max_bytes)
        prefix = QuantumCircuit(select.num_qubits)
        a = select.num_qubits-(dimension.bit_length()-1)
        _append_source_input(prefix, data, preparations, tuple(range(a)), tuple(range(a, select.num_qubits)))
        prefix.compose(select, inplace=True)
        return prefix
    lcu = lcu_intake()
    count = len(coefficients)
    route = data.method.dense_control_route
    q = dimension.bit_length()-1
    a = (count-1).bit_length()
    if backend == "dense_exact":
        # On the whole-matrix route a dense source branch is one unitary, its
        # evolution times the matrix of its input preparation, so the address
        # controls act on one whole-matrix synthesis. Each preparation's
        # matrix is formed once (parameters.construction_work charges it).
        whole = a >= 1 and q >= 2 and _whole_matrix(route, a)
        preparation_matrices = {}
        if whole:
            from qiskit.quantum_info import Operator
            preparation_matrices = {kind: Operator(preparation).data for kind, preparation in preparations.items()}
        layout = data.source_layout

        def branch_specifications():
            for branch in range(count):
                node = layout.branch_to_node[branch]
                if node is None:
                    yield None, 0., branch
                    continue
                yield float(layout.k_values[branch]), float(layout.elapsed_times[branch]), branch

        def finish(branch, unitary):
            kind = layout.branch_kinds[layout.branch_to_node[branch]]
            if whole:
                from qiskit.circuit.library import UnitaryGate
                circuit = QuantumCircuit(q, name=f"source_{branch}")
                circuit.append(UnitaryGate(unitary @ preparation_matrices[kind], label=f"source_{branch}_U",
                                           check_input=False), circuit.qubits)
                return circuit
            return _compose_branch_circuit(preparation=preparations[kind],
                evolution=None, dense_unitary=unitary, label=f"source_{branch}")

        return build_branch_select(lcu, _dense_branches(data, branch_specifications(), finish, cache=True),
                                   route=route, phases=phases)

    selected = data.select_data.plan
    if a:
        # Qiskit's control unrolls each whole branch, every repetition
        # included, so the SELECT's bytes are checked before any branch is built.
        admit_branch_controlled_select(selected, slots=count, max_bytes=data.method.max_bytes)

    def product_formula_branches():
        for branch in range(count):
            node = branch if data.source_layout is None else data.source_layout.branch_to_node[branch]
            if node is None:
                yield QuantumCircuit(q)
                continue
            circuit = _build_product_formula_branch(selected, branch, num_system_qubits=q)
            # Reduced through its unit phase, so the coefficient phase that
            # build_branch_select adds next is not lost (unit_angle).
            circuit.global_phase += unit_angle(selected.identity_phases[branch])
            if data.source_layout is not None:
                circuit = _compose_branch_circuit(preparation=preparations[data.source_layout.branch_kinds[node]],
                    evolution=circuit, dense_unitary=None, label=f"source_{branch}")
            yield circuit
    return build_branch_select(lcu, product_formula_branches(), route=route, phases=phases)


def _dense_branches(data, specifications, finish=None, *, cache=False):
    """Yield dense_exact branch circuits one at a time from node eigensystems.

    ``specifications`` yields (k, elapsed time, branch) triples, with k None
    for an address padding slot. With ``cache`` enabled, each distinct
    k-node's Hermitian eigensystem (time_independent_terms._node_eigensystem)
    is formed at most once per invocation. When L is exactly zero, any
    required H eigensystem is shared across nodes whether or not ``cache``
    is enabled. The branch matrix is ``(V * exp(-i t lambda)) @ V†``
    (_branch_unitary), with eigenvectors V and eigenvalues lambda, and t the
    elapsed time. It is the identity at zero elapsed time or when L and H
    are both zero. Each branch matrix is released once its circuit is
    yielded, so only the current branch's matrix is in memory while SELECT
    controls and appends it. With ``cache`` (the source layouts, whose nodes
    recur across applications), the eigensystem arrays occupy at most
    ``K*(16*D**2 + 8*D)`` bytes, where K is the number of distinct nonpadding
    k-node keys and D is the encoded system dimension. Each computed
    eigensystem stays in memory for this invocation, with the shared H
    eigensystem used when L is exactly zero. Without ``cache`` and with
    nonzero L, only the current node's eigensystem is in memory and a recurring
    k would be factorized again, so parameters.construction_work counts
    one eigensystem per nonzero-time homogeneous branch.
    ``finish(branch, unitary)`` builds a source branch circuit. Otherwise the
    circuit holds the matrix as one UnitaryGate. The branch unitary skips
    the redundant input check (check_input=False), which does not project it
    onto the unitary group.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from .time_independent_terms import _branch_unitary, _node_eigensystem, _node_key
    dense = data.dense_inputs()
    dimension = data.generator.encoded_dimension
    q = dimension.bit_length()-1
    identity = data.generator.zero_evolution
    zero_l = not np.any(dense.l_part)
    eigensystems = {}
    for k_value, elapsed, branch in specifications:
        if k_value is None:
            yield QuantumCircuit(q)
            continue
        if identity or elapsed == 0:
            unitary = np.eye(dimension, dtype=complex)
        else:
            key = "zero_l" if zero_l else _node_key(k_value)
            eigensystem = eigensystems.get(key)
            if eigensystem is None:
                eigensystem = _node_eigensystem(0. if zero_l else k_value, dense.l_part, dense.h_part)
                if cache or zero_l:
                    eigensystems[key] = eigensystem
            unitary = _branch_unitary(eigensystem, elapsed, dimension)
            del eigensystem
        if finish is None:
            circuit = QuantumCircuit(q)
            circuit.append(UnitaryGate(unitary, check_input=False), circuit.qubits)
        else:
            circuit = finish(branch, unitary)
        del unitary
        yield circuit
        del circuit


def _whole_matrix(route, controls):
    """Whether ``route`` controls a dense branch with ``controls`` address bits through its whole controlled matrix."""
    from nwqlib.subroutines._dense_synthesis import select_dense_control_route
    return select_dense_control_route(route, controls) == "whole_matrix"


def dense_branch_select_cx(num_system_qubits: int, address_bits: int, route: str = "gatewise") -> int:
    """Upper bound on the CX of one controlled dense branch of the dense_exact SELECT.

    It counts the CX of the branch's dense unitary. A source preparation inside a
    constant-source branch is priced separately in
    parameters.construction_work, except on the whole-matrix route, where
    it is part of the branch unitary.

    On the whole-matrix route (``_dense_synthesis.select_dense_control_route``
    for ``route`` and the a address bits), a branch on two or more system
    qubits is one ``controlled_unitary_circuit`` on q + a qubits, at most
    ``(25/96) 4**(q+a) - 2**(q+a) + 4/3`` CX
    (``_dense_synthesis.controlled_synthesis_gate_census``), and its open
    controls add only X gates. A one-qubit branch keeps Qiskit's exact
    definition and takes the gate-wise count below on either route.

    build_branch_select controls each branch through the shared control constructor,
    that is qiskit_compat.controlled over the exact synthesis of
    _dense_synthesis.py. Qiskit's add_control then controls every gate of
    that circuit with the a address bits (qiskit/circuit/_add_control.py,
    apply_basic_controlled_gate), so the count is a sum over the gate counts
    of _dense_synthesis.dense_synthesis_gate_census with these costs after
    decomposition into CX and U gates:

    - CX becomes an X with a + 1 controls, and H becomes an X with a
      controls between one-qubit gates. Their counts are Qiskit 2.6.0's
      synthesis without ancillas, the table MCX_CX_BY_CONTROLS.
    - RZ becomes MCRZ: a CRZ of two CX for a = 1, and for a >= 2 two
      dirty-ancilla X gates on ceil(a/2) and two on floor(a/2) controls
      (_mcphase_cx, pauli_evolution._dirty_mcx_cx).
    - U(theta, phi, lambda) becomes a CU of two CX for a = 1. For a >= 2 it
      becomes MCRZ(lambda), MCRY(theta), MCRZ(phi) and a phase gate on a - 1
      controls, the largest of add_control's cases. MCRY costs 12 CX for
      two controls (two Toffoli gates of QuantumCircuit.mcry's "basic"
      mode, which it selects because its V-chain needs no ancilla for two
      controls), 20 for three (its Gray-code construction, seven CU and six
      CX) and the MCRZ count from four controls on.
    - The branch's global phase, which includes arg(c_j), becomes a phase
      gate on a - 1 controls, _mcphase_cx(a) CX, and is free for a = 1.

    Open controls add only X gates. Inside a branch the multi-controlled X
    gates may instead use idle system qubits as ancillas. For a from 1 to 8
    with one or three idle qubits, clean or already used, every gate kind of
    the synthesis decomposed into at most its ancilla-free count, which the
    formula uses
    (tests/test_lchs_resource_structural_law.py).
    With no address bits the single branch stays a UnitaryGate, whose
    synthesis takes at most the CX of those gate counts. The bound is
    recorded as an estimate, because another Qiskit version can control or
    decompose these gates differently. tests/test_lchs_resource_structural_law.py compares
    it with the transpiled built SELECT.
    """
    from nwqlib.subroutines._dense_synthesis import controlled_synthesis_gate_census, dense_synthesis_gate_census
    census = dense_synthesis_gate_census(num_system_qubits)
    if address_bits == 0:
        return census["cx"]
    if num_system_qubits >= 2 and _whole_matrix(route, address_bits):
        return controlled_synthesis_gate_census(num_system_qubits + address_bits)["cx"]
    cost = _controlled_gate_cx(address_bits)
    return sum(census[name]*cost[name] for name in census) + cost["global_phase"]


def _controlled_gate_cx(address_bits: int) -> dict[str, int]:
    """CX of each gate of the dense synthesis after Qiskit adds address_bits >= 1 controls.

    The keys are the gate kinds of the dense synthesis and the branch's global phase. The
    derivation of each entry is in dense_branch_select_cx.
    """
    from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import _dirty_mcx_cx, _mcphase_cx
    a = address_bits
    if a + 1 > len(MCX_CX_BY_CONTROLS):
        raise ValueError("the dense SELECT CX law covers at most 63 address bits")
    mcrz = 2 if a == 1 else 2*_dirty_mcx_cx((a + 1)//2) + 2*_dirty_mcx_cx(a//2)
    mcry = {2: 12, 3: 20}.get(a, mcrz)
    phase = _mcphase_cx(a)
    # Entry k - 1 of the table is the X gate with k controls.
    return {"cx": MCX_CX_BY_CONTROLS[a], "h": MCX_CX_BY_CONTROLS[a - 1], "rz": mcrz,
            "u": 2 if a == 1 else 2*mcrz + mcry + phase, "global_phase": phase}


def build_branch_select(data, branches, route="gatewise", phases=None):
    """Return the phase-weighted, whole-branch SELECT. Coefficient PREP is a separate block.

    Each branch is controlled on its address through
    ``qiskit_compat.controlled`` on the dense control route ``route``.
    ``branches`` may be a generator, and each branch and its copy are
    released after its controlled circuit is appended, before the next is produced.
    Branch j carries arg(c_j) of the LCU coefficient, or ``phases[j]`` when
    a source layout supplies its phases.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import controlled
    a, q = data.num_control_qubits, data.num_system_qubits
    select = QuantumCircuit(a+q, name="lchs_select")
    for index, (coefficient, branch) in enumerate(zip(data.original_coefficients, branches, strict=True)):
        selected = branch.copy()
        selected.name = f"lchs_node_{index}"
        if phases is not None:
            if phases[index]:
                selected.global_phase += float(phases[index])
        elif abs(coefficient) > 0:
            selected.global_phase += float(np.angle(coefficient))
        if a:
            select.append(controlled(selected.to_gate(), a, ctrl_state=index, route=route), select.qubits)
        else:
            select.compose(selected, inplace=True)
        del branch, selected
    return select


def construct_coherent_lcu(block, arguments, method_context):
    """Build the planned coherent LCU body from its saved prepared route, without choosing or fitting anything again.

    The circuit is ``build_source_lcu_circuit``: the factorized coefficient
    PREP, the kind-state preparation R of the unit inputs, one evolution of
    the joint generator, the slot phase diagonal and the inverse coefficient
    PREP, on [evolution ancillas, PREP work W, kernel, application, physical
    system]. W is the work register of a saved analytic kernel PREP
    (``build_analytic_kernel_query`` of ``prepared.kernel_preparation``,
    which replaces the direct kernel factor) and is empty otherwise.
    With every non-system qubit on zero it maps the physical ``|0>`` to
    ``u_fin / (zeta gamma)``. The evolution is the shared real-QSVT or the
    qubitized emitter with the saved joint generator plan and phases, or the
    identity when the selected row is the exact identity shortcut. The
    inputs use their own constructions (``_source_preparations``) or, for
    an MPS initial state, the cores chosen at planning. The M35 builders
    admit each child query against max_select_work and max_bytes before they
    build it.
    """
    _no_arguments(arguments)
    data = block._payload
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.block_encoding import BlockEncoding
    from nwqlib.subroutines.block_encoding.joint import build_joint_encoding, plan_block_encoding_product
    from nwqlib.subroutines.block_encoding.source_prep import build_source_lcu_circuit
    from nwqlib.subroutines.qsp.evolution import build_qubitized_evolution, build_shared_qsvt_evolution
    from .coherent_selection import evolution_factors
    prepared, method = data.qsp_plan, data.method
    limits = dict(max_bytes=method.max_bytes, max_work=method.max_select_work)
    built = []
    for factor in evolution_factors(prepared):
        if factor.route.startswith("shared"):
            built.append(build_shared_qsvt_evolution(factor.generator, factor.phases,
                evolution_time=factor.evolution_time, amplification=factor.plan.amplification, **limits))
        else:
            built.append(build_qubitized_evolution(factor.generator, factor.phases,
                                                   evolution_time=factor.evolution_time, **limits))
    if not built:
        width = (prepared.preparation.kernel_qubits + prepared.preparation.application_qubits
                 + data.generator.num_qubits)
        evolution = BlockEncoding(circuit=QuantumCircuit(width), alpha=1.0, num_ancillas=0, system_qubits=width,
                                  error_bound=0.0, implementation="identity_evolution", metadata={})
    elif len(built) == 1:
        evolution = built[0]
    else:
        # The commuting product filter * exp(-i T_e S_H): both factors act on [application, physical] with
        # independent success workspaces, so the projected block is B_I B_H (M7.6, plan_block_encoding_product).
        evolution = build_joint_encoding(plan_block_encoding_product(built[0], built[1],
                                                                     target_norm_bounds=(1.0, 1.0)), **limits)
    # The Plan's recovery multiplies the success amplitudes by this zeta.
    if evolution.alpha != prepared.recovery:
        raise RuntimeError("the rebuilt coherent evolution has a different recovery than the planned route")
    kernel_preparation = None
    if prepared.kernel_preparation is not None:
        from nwqlib.subroutines.block_encoding.analytic_prep import build_analytic_kernel_query
        kernel_preparation = build_analytic_kernel_query(prepared.kernel_preparation, **limits)
    preparations = coherent_input_preparations(data)
    return build_source_lcu_circuit(prepared.preparation, evolution, initial=preparations.get("initial"),
                                    source=preparations.get("source"), factors=coherent_coefficient_factors(data),
                                    kernel_preparation=kernel_preparation)


def coherent_coefficient_factors(data):
    """Return the ``(kernel, application)`` MPS factor circuits of an explicit MPS coefficient PREP, or None.

    Each nontrivial factor is the layered MPS circuit of the cores chosen at
    planning for its selected label order, so construction and resource
    sampling never repeat the TT-SVD; a factor without qubits is None, the
    identity.
    """
    if data.coefficient_mps is None:
        return None
    import numpy as np
    from nwqlib.subroutines.state_preparation.mps_circuit import build_mps_circuit_state_preparation
    method, preparation = data.method, data.qsp_plan.preparation
    return tuple(None if decomposition is None else build_mps_circuit_state_preparation(
        np.asarray(amplitudes, dtype=np.complex128), max_bond_dim=decomposition.max_bond_dim,
        threshold=decomposition.threshold, num_layers=method.mps_num_layers, _selected_decomposition=decomposition,
        max_bytes=method.max_bytes, max_svd_work=method.max_svd_work).circuit
        for decomposition, amplitudes in zip(data.coefficient_mps,
                                             (preparation.kernel_amplitudes, preparation.application_amplitudes),
                                             strict=True))


def coherent_input_preparations(data):
    """Return the input-direction preparations by kind that the coherent LCU body's R applies.

    They are ``_source_preparations``, with the MPS circuit of the cores
    chosen at planning for an MPS initial state, so construction and
    resource sampling prepare the same inputs without another decomposition.
    """
    preparations = _source_preparations(data)
    if data.initial_mps is not None:
        from nwqlib.subroutines.state_preparation.mps_circuit import build_mps_circuit_state_preparation
        method = data.method
        preparations["initial"] = build_mps_circuit_state_preparation(data.dense.initial_direction,
            max_bond_dim=data.initial_mps.max_bond_dim, threshold=data.initial_mps.threshold,
            num_layers=method.initial_state_mps_num_layers, _selected_decomposition=data.initial_mps,
            max_bytes=method.max_bytes, max_svd_work=method.max_svd_work).circuit
    return preparations


def _append_source_input(circuit, data, preparations, address, system):
    """Prepare the constant-source input on the system register, if any.

    A layout with only one input kind prepares that input unconditionally.
    With both kinds, the kind_control address bit selects the input for every
    branch, value 0 for the initial direction and value 1 for the source.
    Homogeneous evolution prepares u0 outside SELECT, so nothing is added.
    """
    if data.source_layout is None:
        return
    layout = data.source_layout
    if layout.kind_control is None:
        # One retained input kind: the layout's branches all prepare it.
        circuit.compose(preparations[layout.branch_kinds[0]], qubits=system, inplace=True)
        return
    from nwqlib.subroutines.qiskit_compat import controlled
    for index, kind in enumerate(("initial", "source")):
        circuit.append(controlled(preparations[kind].to_gate(label="prep_"+kind), 1, ctrl_state=index),
            [address[layout.kind_control], *system])


def _compose_branch_circuit(
    *,
    preparation,
    evolution,
    dense_unitary,
    label: str,
):
    """Return one source-layout branch that prepares its input and then evolves it.

    Branch-controlled and dense source layouts prepare the input inside each
    branch, because an initial-state branch starts from u0 and a Duhamel
    branch from b. The evolution is either a product-formula circuit or a
    dense unitary exp(-i*t*(k*L+H)).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    system_qubits = preparation.num_qubits
    circuit = QuantumCircuit(system_qubits, name=label)
    circuit.compose(preparation, qubits=list(range(system_qubits)), inplace=True)
    if evolution is not None:
        circuit.compose(evolution, qubits=list(range(system_qubits)), inplace=True)
    if dense_unitary is not None:
        circuit.append(UnitaryGate(dense_unitary, label=f"{label}_U", check_input=False),
                       list(range(system_qubits)))
    return circuit
