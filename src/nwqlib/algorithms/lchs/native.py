"""Circuit constructors for the blocks of an LCHS Program.

They read the numerical data chosen at planning and never select it again.
"""

from __future__ import annotations
from nwqlib._limits import DEFAULT_MAX_BYTES

from nwqlib.blocks.selection import _no_arguments
from typing import Any

# Additional allowance for the selected Hamiltonian synthesis, not a guarantee
# that the returned physical solution meets approximation_tolerance. Available
# component bounds keep their own premises and scale through the actual LCU.
LCHS_QSP_EPSILON_FRACTION = 0.1

# Additional per-node product-formula allowance as a fraction of approximation_tolerance,
# with the same scope as above. Both fractions are registered in
# ENGINEERING_CONSTANTS.
LCHS_TROTTER_EPSILON_FRACTION = 0.1


def resolve_lcu_select_implementation(method, *, select_plan: Any | None = None) -> str:
    """Resolve homogeneous or inhomogeneous SELECT synthesis without a circuit.

    Decision table, by Hamiltonian-evolution backend:

    - qsp_block_encoding: always ``compiled_qsp``. Any explicit request other
      than ``auto`` raises.
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
    he_backend = method.hamiltonian_evolution_backend
    # Eligibility is a construction property. Dense, QSP and variable-step
    # product formulas expose different SELECT implementations.
    if he_backend == "qsp_block_encoding":
        if requested != "auto":
            raise ValueError(
                "hamiltonian_evolution_backend='qsp_block_encoding' uses its "
                "internal compiled_qsp SELECT; lcu_select_implementation must be 'auto'"
            )
        return "compiled_qsp"
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
    """Lower one selected PREP leaf from its saved tensor and optional MPS cores.

    Direct preparation builds the exact normalized state with the direct
    magnitude/phase tree. Its record declares a ``qiskit.direct``
    preparation, so lowering and the Run compare its 2**q amplitudes with
    ``max_direct_amplitudes`` before this constructor runs. MPS preparation
    reuses the cores chosen at planning, so lowering never repeats the TT-SVD.
    """
    _no_arguments(arguments)
    target, decomposition, layers, max_bytes, max_svd_work = block._payload
    if decomposition is None:
        from nwqlib.subroutines.state_preparation.direct import _build_normalized_state_preparation
        return _build_normalized_state_preparation(target, input_norm=1., register_name="state").circuit
    from nwqlib.subroutines.state_preparation.mps_circuit import build_mps_circuit_state_preparation
    return build_mps_circuit_state_preparation(target,
        max_bond_dim=decomposition.max_bond_dim, threshold=decomposition.threshold,
        num_layers=layers, _selected_decomposition=decomposition,
        max_bytes=max_bytes, max_svd_work=max_svd_work).circuit


def _source_preparations(data):
    """Return exact direct PREP circuits for the unit initial and source directions.

    These trees are built inside SELECT and are not declared as PREP blocks,
    so ``max_direct_amplitudes`` does not apply. Each direction fills the
    system register of padded dimension D, which ``selection.select_dense``
    admits at ``64*D**2 + 128*D`` bytes before planning continues. At the default
    ``max_bytes`` of 10**10 a tree therefore has at most 8192 amplitudes. When
    the dissipative part L is nonzero, the eigensolve also requires
    ``D**3 <= max_spectral_work`` for the unpadded dimension, which by
    default limits a tree to 512 amplitudes.
    """
    from nwqlib.subroutines.state_preparation.direct import _build_normalized_state_preparation
    result = {}
    for kind, direction in (("initial", data.initial_direction), ("source", data.source_direction)):
        if direction is not None:
            result[kind] = _build_normalized_state_preparation(direction, input_norm=1., register_name="system").circuit
    return result


def construct_select(block, arguments, method_context):
    """Lower the selected LCHS SELECT leaf from its saved LCHSData, without reselection.

    SELECT applies sum_j |j><j| (x) e^{i*arg(c_j)} U_j on the coefficient
    address and system registers. The coefficient magnitudes live in the
    separate PREP pair, so PREP, SELECT and inverse PREP form the LCU of ACL
    arXiv:2312.03916v2, Appendix A.3, Lemma 24, with the complex phases moved
    from the two state-preparation oracles into SELECT. The leaf follows the
    selected route: dense_exact controls classically exponentiated branches,
    identity_evolution applies only the coefficient phases, QSP realizes the
    saved joint generator and phases, and product formulas use the saved
    multiplexed or structured tables. Branch-controlled product formulas and
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
    from .select_synthesis import _build_product_formula_branch, build_product_formula_select
    backend = data.method.hamiltonian_evolution_backend
    dimension = len(data.initial)
    coefficients = data.coefficient_plan.coefficients if data.source_layout is None else data.source_layout.coefficients

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
    if data.selected_select == "identity_evolution":
        q = dimension.bit_length()-1
        a = (len(coefficients)-1).bit_length()
        select = QuantumCircuit(a+q)
        address, system = tuple(range(a)),tuple(range(a,a+q))
        phase = _compiled_coefficient_phase_gate(coefficients,1<<a)
        if phase is not None:
            select.append(phase,address)
        _append_source_input(select,data,preparations,address,system)
        return select
    if backend == "qsp_block_encoding":
        select = build_selected_evolution(data.qsp_plan, max_bytes=data.method.max_bytes,
            max_work=data.method.max_select_work, dense_control_route=data.method.dense_control_route)
        # The QSP evolution adds parity, pair and signal qubits to the joint
        # generator's block ancillas. The address register follows them.
        ancillas = data.qsp_plan["layout"]["num_ancillas"] + 3
        q = dimension.bit_length()-1
        address = tuple(range(ancillas, select.num_qubits-q))
        phase_gate = _compiled_coefficient_phase_gate(coefficients, 1 << len(address))
        prefix = QuantumCircuit(select.num_qubits)
        if phase_gate is not None:
            prefix.append(phase_gate, address)
        _append_source_input(prefix, data, preparations, address, tuple(range(select.num_qubits-q, select.num_qubits)))
        prefix.compose(select, inplace=True)
        return prefix
    if backend != "dense_exact" and data.selected_select != "branch_controlled":
        select = build_product_formula_select(data.select_data.plan, num_system_qubits=dimension.bit_length()-1,
            implementation=data.selected_select)
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
                                   route=route)

    def product_formula_branches():
        selected = data.select_data.plan
        for branch in range(count):
            node = branch if data.source_layout is None else data.source_layout.branch_to_node[branch]
            if node is None:
                yield QuantumCircuit(q)
                continue
            circuit = _build_product_formula_branch(selected, branch, num_system_qubits=q)
            circuit.global_phase += selected.identity_phases[branch]
            if data.source_layout is not None:
                circuit = _compose_branch_circuit(preparation=preparations[data.source_layout.branch_kinds[node]],
                    evolution=circuit, dense_unitary=None, label=f"source_{branch}")
            yield circuit
    return build_branch_select(lcu, product_formula_branches(), route=route)


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
    yielded, so only the current branch's matrix is alive while SELECT
    controls and appends it. With ``cache`` (the source layouts, whose nodes
    recur across applications), the eigensystem arrays occupy at most
    ``K*(16*D**2 + 8*D)`` bytes, where K is the number of distinct nonpadding
    k-node keys and D is the encoded system dimension. Each computed
    eigensystem stays alive for this invocation, with the shared H
    eigensystem used when L is exactly zero. Without ``cache`` and with
    nonzero L, only the current node's eigensystem is alive and a recurring
    k would be factorized again, so parameters.construction_work counts
    one eigensystem per nonzero-time homogeneous branch.
    ``finish(branch, unitary)`` builds a source branch circuit. Otherwise the
    circuit holds the matrix as one UnitaryGate. The selected unitary skips
    the redundant input check (check_input=False), which does not project it
    onto the unitary group.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from .time_independent_terms import _branch_unitary, _node_eigensystem, _node_key
    quad = data.quadrature
    dimension = len(data.initial)
    q = dimension.bit_length()-1
    identity = not (np.any(quad.l_part) or np.any(quad.h_part))
    zero_l = not np.any(quad.l_part)
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
                eigensystem = _node_eigensystem(0. if zero_l else k_value, quad.l_part, quad.h_part)
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

    It prices the branch's dense unitary. A source preparation inside a
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
    apply_basic_controlled_gate), so the count is a sum over the gate census
    of _dense_synthesis.dense_synthesis_gate_census with these costs after
    lowering to CX and U gates:

    - CX becomes an X with a + 1 controls, and H becomes an X with a
      controls between one-qubit gates. Their counts are Qiskit 2.5.2's
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
    the census lowered to at most its ancilla-free count, which the law uses
    (tests/test_lchs_resource_structural_law.py).
    With no address bits the single branch stays a UnitaryGate, whose
    synthesis takes at most the census CX. The bound is recorded as an
    estimate, because another Qiskit version can control or lower these
    gates differently. tests/test_lchs_resource_structural_law.py compares
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

    The keys are the census gates and the branch's global phase. The
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


def build_branch_select(data, branches, route="gatewise"):
    """Return the phase-weighted, whole-branch SELECT. Coefficient PREP is a separate block.

    Each branch is controlled on its address through
    ``qiskit_compat.controlled`` on the dense control route ``route``.
    ``branches`` may be a generator: each branch and its copy are released
    after its controlled circuit is appended, before the next is produced.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import controlled
    a, q = data.num_control_qubits, data.num_system_qubits
    select = QuantumCircuit(a+q, name="lchs_select")
    for index, (coefficient, branch) in enumerate(zip(data.original_coefficients, branches, strict=True)):
        selected = branch.copy()
        selected.name = f"lchs_node_{index}"
        if abs(coefficient) > 0:
            selected.global_phase += float(np.angle(coefficient))
        if a:
            select.append(controlled(selected.to_gate(), a, ctrl_state=index, route=route), select.qubits)
        else:
            select.compose(selected, inplace=True)
        del branch, selected
    return select


def build_selected_evolution(selected, *, max_bytes=DEFAULT_MAX_BYTES, max_work=1_000_000_000,
                             dense_control_route="gatewise"):
    """Realize the selected joint generator and saved QSP phases without solving again.

    The joint generator is D_L (x) L + D_H (x) H, with the saved diagonals, and
    QSP evolves it for evolution_time using the phases prepared at planning.
    Its combine qubit controls a dense child on ``dense_control_route``.
    """
    from dataclasses import replace
    import numpy as np
    from nwqlib.subroutines.qsp import build_control_diagonal_generator_encoding
    from nwqlib.subroutines.qsp.evolution import build_qsp_evolution_encoding
    l_plan, h_plan, pruned_h = selected['part_plans']
    joint = build_control_diagonal_generator_encoding(
        _build_compiled_qsp_part_encoding(l_plan, max_bytes=max_bytes, max_work=max_work),
        None if h_plan.implementation == 'exact_zero' else _build_compiled_qsp_part_encoding(h_plan,
            max_bytes=max_bytes, max_work=max_work),
        l_diagonal=selected['l_diagonal'], h_diagonal=selected['h_diagonal'],
        max_work=max_work, max_bytes=max_bytes, dense_control_route=dense_control_route)
    # When pruning removes every H term, the joint generator has no H child to
    # carry that error. The omitted operator D_H (x) H has norm at most
    # max|D_H| times the dropped Pauli mass, which is added to the generator
    # error bound here.
    missing_h = float(np.max(np.abs(selected['physical_h_diagonal'])))*pruned_h
    if missing_h > 0:
        joint = replace(joint,error_bound=None if joint.error_bound is None else joint.error_bound+missing_h)
    return build_qsp_evolution_encoding(joint,evolution_time=selected['evolution_time'],
        epsilon=selected['epsilon_he'],max_work=max_work,max_bytes=max_bytes,
        _prepared=selected['prepared_evolution']).circuit


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
        kind = "initial" if layout.initial_norm > 0 else "source"
        circuit.compose(preparations[kind], qubits=system, inplace=True)
        return
    from nwqlib.subroutines.qiskit_compat import controlled
    for index, kind in enumerate(("initial", "source")):
        circuit.append(controlled(preparations[kind].to_gate(label="prep_"+kind), 1, ctrl_state=index),
            [address[layout.kind_control], *system])


def _compiled_coefficient_phase_gate(
    coefficients, padded_length: int
):
    """Return the diagonal gate sum_j exp(i*arg(c_j))|j><j| on the address register.

    Every nonzero coefficient keeps its exact phase, however small, and zero
    or padding slots get phase zero. None means that all phases are zero.
    """

    import numpy as np
    from qiskit import QuantumCircuit
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases
    coefficient_array = np.zeros(padded_length, dtype=complex)
    coefficient_array[:len(coefficients)] = coefficients
    phases = np.where(coefficient_array != 0.0, np.angle(coefficient_array), 0.0)
    if not np.any(phases != 0.0):
        return None
    phase_circuit = QuantumCircuit(padded_length.bit_length() - 1)
    gate = append_control_diagonal_phases(phase_circuit, phase_circuit.qubits, phases)
    return gate if gate is not None else phase_circuit.to_gate()


def _build_compiled_qsp_part_encoding(plan, *, max_bytes=DEFAULT_MAX_BYTES, max_work=1_000_000_000):
    """Realize one nonzero child from its shared immutable block plan.

    The returned encoding keeps the plan's alpha and error_bound (the
    block-encoding error plus the pruned Pauli mass, in the operator frame of
    L or H). input_scale_factor is the largest kept Pauli coefficient
    magnitude.
    """

    from dataclasses import replace
    from nwqlib.subroutines.block_encoding.core import build_block_encoding_from_plan
    if plan.implementation == "exact_zero":
        raise ValueError("an exact-zero QSP child has no block encoding to realize")
    encoding = build_block_encoding_from_plan(plan, max_bytes=max_bytes, max_work=max_work)
    decomposition = plan.decomposition
    scale = max(abs(complex(term.coefficient)) for term in decomposition.terms)
    return replace(
        encoding,
        alpha=plan.alpha,
        error_bound=plan.error_bound,
        metadata={
            **dict(encoding.metadata),
            **dict(plan.detail),
            "input_scale_factor": float(scale),
            "pauli_pruned_mass_bound": plan.error_bound,
            "error_bound_provenance": "abstract_operator_bound_physical_generator_frame",
            "target_operator_is_hermitian": True,
            "encoded_operator_is_hermitian": True,
        },
    )


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
