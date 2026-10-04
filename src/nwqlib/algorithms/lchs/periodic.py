"""Compact periodic LCHS selection and Strang circuits, without system arrays.

The periodic operator is A = mass*I + diffusion*(2I - S - S^dagger) + i*potential*Z_0
with S the cyclic shift. Its Cartesian parts are L = mass*I +
diffusion*(2I - S - S^dagger), which is PSD analytically with
||L|| <= mass + 4*diffusion, and H = potential*Z_0. No 2**q array is formed.
The k-grid comes from the same providers as the dense route, using the
analytic norm bound for ||L||.
"""

from cmath import phase
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from hashlib import sha256
from math import inf, isfinite, nextafter, sqrt

from nwqlib._preparation_laws import direct_preparation_cx_bound, direct_preparation_controlled_cx_bound
from nwqlib._validation import finite_real
from nwqlib.blocks.records import BlockSemantics, SelectedDefinition
from nwqlib.blocks.selection import SelectedBlock, _no_arguments, _signature, select_preparation
from nwqlib.core.records import InputRef, Source
from nwqlib.evidence.records import Evidence
from nwqlib.ir import BlockCall, Definition, PortMap, Repeat
from nwqlib.operators.access import _check_bytes
from nwqlib.resources.records import ResourceLaw, Workspace
from nwqlib.subroutines.trotterization.error_budget import _smallest_step_count, _upward_float
from .parameters import _feed_selected_value
from .providers import LCHSProblemContext, _resolve_coefficient_context


SOURCE = Source(name="lchs.periodic", version="1", domain="periodic mass/diffusion and alternating imaginary potential",
    reference="A=mass I+diffusion(2I-S-S†)+i potential Z0; analytic PSD; fixed symmetric Strang")


@dataclass(frozen=True)
class _PeriodicPayload:
    """Selected periodic scalars shared by both native leaves and the archive.

    half_angles are the per-address even-bond rotation angles of one half
    step. phases combine each coefficient's argument with its identity
    evolution phase. parameter_id is periodic_identity of the other fields,
    and both leaves record it as their input identity.
    """

    operator: object
    elapsed: float
    steps: int
    coefficient_plan: object
    amplitudes: object
    half_angles: tuple[float, ...]
    phases: tuple[float, ...]
    synthesis_bound: float | None
    parameter_id: str


def periodic_identity(payload, method):
    """Digest every selected field of a periodic payload except parameter_id.

    The SHA-256 starts from the Method identity and feeds each field name and
    value through _feed_selected_value, with type tags and container lengths,
    as selected_identity does for LCHSData. The operator enters as its
    admitted reference identity and its stencil parameters. Selection stores
    the result as parameter_id and in both leaf records, so the Plan identity
    changes with every selected periodic field, including the operator
    parameters.
    """
    digest = sha256(method.content_id.encode())
    for field in fields(payload):
        if field.name != 'parameter_id':
            value = getattr(payload, field.name)
            if field.name == 'operator':
                value = (value.reference.identity, value.periodic_stencil())
            _feed_selected_value(digest, (field.name, value))
    return 'sha256:' + digest.hexdigest()


def _construction_law(q, a, steps):
    """Complete elementary slots, including QFT phases and one outer control.

    Counts are selected occurrences, not optimized native counts. Every step
    uses three H-UCRZ-H rotations and two one-band shifts. A QFT pair has
    q(q-1) controlled phases, 2*floor(q/2) SWAPs and 2q H. Each controlled
    phase has a two-CX/three-phase decomposition. Global phases survive a
    control. They cost a one-qubit phase for the one-control subset.
    """
    p = 1 << a
    shifts = 2 * steps
    qft_cp = shifts * q * (q - 1)
    qft_swaps = shifts * 2 * (q // 2)
    cx = 3 * steps * (p if a else 0) + 2 * qft_cp + 3 * qft_swaps + max(0, p - 2)
    rotations = 3 * steps * p + 2 * steps + shifts * q + p - 1
    phases = 3 * qft_cp
    h = 6 * steps + 2 * shifts * q
    base_cx = cx + 2 * direct_preparation_cx_bound(a, complex_phases=False)
    controlled_cx = (6 * cx + 2 * rotations + 2 * phases + h
                     + 2 * direct_preparation_controlled_cx_bound(a))
    # PREP's scalar tables, Walsh scratch, phase tables, and the polynomial
    # circuit slots are charged before their native builders are imported.
    slots = cx + rotations + phases + h + 4 * p + shifts + 1
    items = 12 * p + q + steps + slots
    payload = 192 * p + 64 * slots + 16 * q
    work = items + payload // 8 + controlled_cx + 4 * p * (a + 1)
    return dict(address_qubits=a, projected_cx=base_cx, projected_controlled_cx=controlled_cx,
        construction_logical_bytes=payload, construction_work=work)




def _magnitude_up(value):
    """Return an exact rational at least ``|value|`` for a complex LCU coefficient with finite parts.

    With ``s = max(|Re c|, |Im c|)`` and ``l = min(|Re c|, |Im c|)``,
    ``|c| = s sqrt(1 + (l/s)**2)``, exactly zero for s = 0. The ratio, its
    square and the sum are each rounded to nearest and stepped one binary64
    number up, and the square root, which IEEE 754 rounds correctly, is
    stepped up as well, so each factor bounds its exact value from above,
    subnormal intermediates included. The ratio lies in [0, 1] and the root's
    argument near [1, 2], so no step leaves the binary64 range, and the
    product with s is formed exactly, so the magnitude need not be a binary64
    number. Ordinary ``abs(c)`` has no such outward guarantee.
    """
    real, imag = abs(value.real), abs(value.imag)
    if not (isfinite(real) and isfinite(imag)):
        raise ValueError(f"periodic LCHS coefficient {value!r} is not finite")
    large, small = max(real, imag), min(real, imag)
    if large == 0:
        return Fraction(0)
    ratio = nextafter(small / large, inf) if small else 0.0
    square = nextafter(ratio * ratio, inf) if ratio else 0.0
    root = nextafter(sqrt(nextafter(1.0 + square, inf) if square else 1.0), inf)
    return Fraction(large) * Fraction(root)


def select_periodic_parameters(operator, *, elapsed, method):
    """Select scalar/branch data from the analytic periodic L norm.

    Each branch generator k*L + H splits into an identity part
    k*(mass + 2*diffusion), applied as an address phase, and three Strang
    terms. They are the even bonds -k*diffusion*X_0, the odd bonds (the
    shift-conjugated copy of that term, identical to it when q = 1, which
    gives the double-wrap edge) and potential*Z_0. The synthesis bound
    T**3*sum_j |c_j|*(2*|k_j|*diffusion + |potential|)**3/(3*r**2) follows
    from Childs et al., Phys. Rev. X 11, 011020,
    doi:10.1103/PhysRevX.11.011020, Proposition 10, Eq. (121),
    after relaxing each nested commutator by ||[X, Y]|| <= 2*||X||*||Y||.
    That relaxation gives t**3*Lambda**3/3 per step with Lambda the sum of
    term norms, and r steps of t = T/r give the bound. The relaxation is
    NWQLib's.

    Derivation of t**3*Lambda**3/3. With term norms a_i, the relaxed
    Eq. (121) is t**3*((1/3)*sum_{i<j} a_i*a_j**2 + (1/6)*sum_{i<j} a_i**2*a_j
    + (2/3)*sum_{i<j<k} a_i*a_j*a_k), and every monomial appears in
    Lambda**3/3 with a coefficient at least as large (1, 1 and 2).
    The even-bond term has norm |k|*diffusion, the odd-bond term the same,
    and potential*Z_0 has norm |potential|, so Lambda = 2*|k|*diffusion + |potential|.

    The weighted branch sum follows from the triangle inequality. The LCU
    applies sum_j c_j*U_j, so replacing each U_j by its r-step Strang
    product S_j changes the operator by at most sum_j |c_j|*||U_j - S_j||,
    which is the bound above for a unit input.

    Step count. Write B = T**3*sum_j |c_j|*Lambda_j**3/3, so the bound is
    B/r**2. With ``trotter_steps`` given, r is that value. With
    ``trotter_synthesis_tolerance`` eps instead, r is the smallest positive
    integer with B/r**2 <= eps, which is r = ceil(sqrt(n)) with
    n = ceil(B/eps) (r = 1 when B = 0). B depends only on the selected grid,
    T and the stencil, so r is chosen after the grid and before the
    construction limits (max_trotter_steps, then the construction law's
    max_bytes and max_select_work) are applied.

    Evaluation. B is an exact rational: T, the diffusion and potential
    coefficients and the nodes k_j are binary64 values read exactly, Lambda_j
    and the cubes are formed exactly, and |c_j| is replaced by the exact
    upper value of ``_magnitude_up``. No intermediate can round to zero or
    overflow, and a float Lambda_j need not be representable for
    T**3*Lambda_j**3 to be. The inversion is exact rational arithmetic on B
    and eps (trotterization.error_budget._smallest_step_count, shared with
    the step rule of Childs et al., Sec. V B), so for r > 1, B/(r - 1)**2
    exceeds eps. The recorded bound is B/r**2 rounded upward, so it is at
    most eps and positive whenever B is. With ``trotter_steps`` given, a B/r**2 above the
    largest binary64 number is recorded as unavailable. The bound is
    sufficient and not tight, so a smaller r may also keep the actual
    synthesis error within eps.
    """
    parameters = operator.periodic_stencil()
    if (method.hamiltonian_evolution_backend != "trotter" or method.trotter_order != 2
            or method.lcu_select_implementation not in {"auto", "structured"}):
        raise ValueError("periodic LCHS requires fixed second-order trotter and auto/structured SELECT")
    if method.initial_state_preparation != "direct" or method.lcu_state_preparation != "direct":
        raise ValueError("periodic LCHS uses direct basis/product/input and coefficient PREP")
    elapsed = finite_real(elapsed, "elapsed time")
    if elapsed <= 0:
        raise ValueError("periodic LCHS requires positive elapsed time")
    q = parameters.num_qubits
    l_norm = parameters.mass + 4 * parameters.diffusion
    if not isfinite(l_norm) or l_norm <= 0:
        raise ValueError("periodic LCHS requires finite positive mass+4*diffusion")
    plan = _resolve_coefficient_context(lchs_kernel=method.lchs_kernel,
        k_quadrature=method.k_quadrature,max_bytes=method.max_bytes,max_quadrature_work=method.max_quadrature_work,
        problem_context=LCHSProblemContext(final_time=elapsed,epsilon=method.approximation_tolerance,
            l_norm=l_norm))
    a = (len(plan.coefficients)-1).bit_length()
    # B = T**3/3 * sum_j |c_j|*Lambda_j**3 with Lambda_j = 2*|k_j|*diffusion + |potential|, exactly.
    diffusion, potential = Fraction(parameters.diffusion), abs(Fraction(parameters.potential))
    exact = Fraction(elapsed)**3/3*sum((_magnitude_up(complex(c))*(2*abs(Fraction(k))*diffusion+potential)**3
        for k,c in zip(plan.nodes,plan.coefficients,strict=True)), Fraction(0))
    steps = method.trotter_steps
    if steps is None:
        # Smallest r >= 1 with B/r**2 <= tolerance.
        steps = _smallest_step_count(exact,method.trotter_synthesis_tolerance,2)
        if steps > method.max_trotter_steps:
            raise ValueError(
                f"periodic Strang synthesis needs {steps} steps for "
                f"trotter_synthesis_tolerance={method.trotter_synthesis_tolerance!r}, "
                f"above max_trotter_steps={method.max_trotter_steps}"
            )
    counts = _construction_law(q,a,steps)
    required_work = counts["construction_work"]
    required_bytes = counts["construction_logical_bytes"]
    if required_work > method.max_select_work or required_bytes > method.max_bytes:
        raise ValueError(
            f"periodic Strang construction for q={q}, address_qubits={a}, steps={steps} "
            f"requires {required_work} work units and {required_bytes} bytes, "
            f"with max_select_work={method.max_select_work} and max_bytes={method.max_bytes}"
        )
    amplitudes = plan.prep_amplitudes(max_bytes=method.max_bytes)
    angles = tuple(-elapsed/steps*k*parameters.diffusion for k in plan.nodes)
    phases = tuple(phase(c)-elapsed*k*(parameters.mass+2*parameters.diffusion)
        for k,c in zip(plan.nodes,plan.coefficients,strict=True))
    if not all(isfinite(value) for value in (*angles,*phases,elapsed/steps*parameters.potential)):
        raise ValueError('periodic LCHS selected rotations and phases must be finite')
    # B/r**2 rounded upward, so a step count selected from the tolerance records a bound at most the
    # tolerance, and an unrepresentable bound for given trotter_steps is unavailable.
    bound = _upward_float(exact/steps**2)
    bound = None if bound == inf else bound
    payload = _PeriodicPayload(operator,elapsed,steps,plan,amplitudes,angles,phases,bound,'')
    return replace(payload,parameter_id=periodic_identity(payload,method)),counts


def _component_facts(payload, *, problem):
    """Publish the periodic circuit-path stages with the analytic PSD premise.

    L is PSD by construction, so there is no shift and every PSD recovery
    factor exp(shift*t) is one. Direct PREP stages are exact zero, and the Strang bound is the
    trotter_synthesis stage.
    """
    from .analysis import fact
    from .solution_error_budget import _build_lchs_application_record,_stage_manifest,circuit_stage_bounds
    from .time_independent_terms import _unusable_kernel_stage_names
    from nwqlib.problems.inputs import compose_recovery
    from nwqlib.problems.records import Solution
    raw = payload.coefficient_plan.record()
    raw.update(psd_shift=0.,approximate_lchs_error_bound=payload.coefficient_plan.quadrature.approximate_lchs_error_bound)
    frame = Solution().frame(problem)
    scale = problem.initial_state.preparation.physical_scale
    if scale.as_float() is None:
        return tuple(fact(name,frame,0. if name in {'initial_state_preparation','lcu_coefficient_preparation'} else None,
            reason='physical input norm is not representable',source=SOURCE)
            for name in _stage_manifest(backend='trotter',inhomogeneous=False,circuit=True))
    application = _build_lchs_application_record(raw,final_time=payload.elapsed,start_time=0.,
        weight=scale.as_float(),input_norm=1.,backend='trotter',synthesis_error_bound=payload.synthesis_bound)
    recovery = compose_recovery(payload.coefficient_plan.coefficient_l1_norm,scale)
    values = circuit_stage_bounds(raw,applications=(application,),backend='trotter',inhomogeneous=False,
        psd_premise_satisfied=True,gamma=recovery.as_float(),delta_lcu=0.,input_preparation_output_error_bound=0.,
        unusable_stages=_unusable_kernel_stage_names(raw))
    return tuple(fact(name,frame,value,reason=reason or 'selected periodic component unavailable',source=SOURCE)
        for name,(value,reason) in values.items())


def _select_periodic_leaf(name,payload,*,basis,phase_only):
    """Declare the address-phase leaf or the one-step Strang leaf with its CX laws.

    The step leaf describes one Strang step, and the Program's Repeat node
    supplies the step count, so resource folding multiplies it once.
    """
    q = payload.operator.periodic_stencil().num_qubits
    a = len(payload.amplitudes).bit_length()-1
    p = 1<<a
    one = _construction_law(q,a,1)
    # Factor the complete one-step law of _construction_law into PREP pair,
    # phase and one step. Repeat supplies the actual step population at the
    # shared fold owner.
    phase_cx,phase_controlled = max(0,p-2),6*max(0,p-2)+2*(p-1)
    cx = phase_cx if phase_only else one['projected_cx']-2*direct_preparation_cx_bound(a,complex_phases=False)-phase_cx
    controlled_cx = phase_controlled if phase_only else one['projected_controlled_cx']-2*direct_preparation_controlled_cx_bound(a)-phase_controlled
    signature = _signature(name,'lchs.periodic.phase' if phase_only else 'lchs.periodic.strang',
        tuple((f'bit_{j}',1) for j in range(a+q)))
    record = SelectedDefinition(signature=signature,implementation=signature.target,
        semantics=BlockSemantics(kind='unknown',input=InputRef(identity=payload.parameter_id,
            representation='selected_periodic_lchs_parameters',source=SOURCE),basis=basis,
            relation='address-dependent coefficient/identity phase' if phase_only else 'one selected symmetric Strang step at each address',
            input_projector='arbitrary coefficient address and system',output_projector='same registers',
            success='unitary SELECT factor',workspace=0,restoration='address preserved; no shift workspace',
            epsilon=None,approximation_metric='operator norm',approximation_evidence='weighted Strang bound at the Plan owner',
            inverse_legal=True,control_legal=True,phase='physical coefficient and identity phases kept'),
        choice=payload.parameter_id,decomposition=None,cost_law=None,construction_work=one['construction_work'],
        cost_context='selected address phase' if phase_only else 'one Strang step; Program Repeat supplies multiplicity',
        resource_laws=tuple(ResourceLaw(metric='cx',basis='cx',value=controlled_cx if control else cx,
            interpretation='upper_bound',evidence=Evidence(kind='external_specification',source=SOURCE),
            controlled=control,adjoint=adjoint,assumptions=('selected elementary slots before native optimization/routing',))
            for control in (False,True) for adjoint in (False,True)),
        workspace=(Workspace(location='host',purpose='workspace',bytes=one['construction_logical_bytes'],source=SOURCE),
                   Workspace(location='host',purpose='workspace',bytes=None,source=SOURCE)))
    return SelectedBlock.bind(record,payload=payload,
        constructor=construct_periodic_phase if phase_only else construct_periodic_step)


def plan_periodic(method,problem,*,output,shots,rng):
    """Select compact homogeneous periodic Strang evolution with physical scalar/vector recovery."""
    from nwqlib.blocks.selection import transform_block
    from nwqlib.problems.inputs import compose_recovery
    from .primary_records import LCHSReconstruction
    from .quantum import finish_quantum_plan,observable_terms,select_vector_preparation
    if problem.source is not None:
        raise ValueError('PeriodicStencil LCHS currently has a homogeneous Strang realization')
    payload,counts = select_periodic_parameters(problem.A,elapsed=problem.elapsed_time,method=method)
    q,a = problem.A.periodic_stencil().num_qubits,counts['address_qubits']
    success,coordinates = tuple(range(a)),tuple(range(a,a+q))
    terms = observable_terms(problem,output,max_bytes=method.max_bytes)
    quad = payload.coefficient_plan.quadrature
    reconstruction = LCHSReconstruction(elapsed_time=problem.elapsed_time,dimension=problem.dimension,encoded_dimension=1<<q,
        initial_scale=problem.initial_state.preparation.physical_scale,
        recovery=compose_recovery(payload.coefficient_plan.coefficient_l1_norm,problem.initial_state.preparation.physical_scale),
        success_bits=success,system_bits=coordinates,terms=terms,
        coefficient_l1_norm=payload.coefficient_plan.coefficient_l1_norm,physical_branches=quad.physical_node_count,
        padded_branches=1<<a,selected_backend='trotter',selected_select='periodic_strang',psd_premise='analytic_periodic',
        l_norm=problem.A.periodic_stencil().mass+4*problem.A.periodic_stencil().diffusion,
        kernel_approximation_bound=quad.approximate_lchs_error_bound,quadrature_bound=quad.quadrature_error_bound,
        step_counts=(payload.steps,)*len(payload.coefficient_plan.nodes))
    prep = problem.initial_state.preparation
    _check_bytes(prep.payload_bytes,method.max_bytes,'periodic input PREP')
    if prep.work > method.max_select_work:
        raise ValueError('periodic input PREP exceeds max_select_work')
    blocks = [select_preparation('initial_prep',problem.initial_state,per_bit=True)]
    definitions = [Definition(id='initial_prep',node=BlockCall(signature='initial_prep',
        ports=tuple(PortMap(port=f'system_{j}',wire=f'bit_{bit}') for j,bit in enumerate(coordinates))))]
    calls = ['initial_prep']
    if a:
        coefficient = select_vector_preparation('coefficient_prep',payload.amplitudes,method=method)
        blocks.extend((coefficient,transform_block('coefficient_inverse',coefficient,adjoint=True)))
        for name in ('coefficient_prep','coefficient_inverse'):
            definitions.append(Definition(id=name,node=BlockCall(signature=name,
                ports=tuple(PortMap(port=f'system_{j}',wire=f'bit_{j}') for j in range(a)))))
        calls.append('coefficient_prep')
    # Keep the identity phase separate from repeated Strang steps, then
    # unprepare the coefficient address before the selected success readout.
    for name,phase_only in (('periodic_phase',True),('periodic_step',False)):
        leaf = _select_periodic_leaf(name,payload,basis=problem.basis,phase_only=phase_only)
        blocks.append(leaf)
        node = BlockCall(signature=name,ports=tuple(PortMap(port=f'bit_{j}',wire=f'bit_{j}') for j in range(a+q)))
        if not phase_only:
            definitions.append(Definition(id='periodic_step_body',node=node))
            node = Repeat(count=payload.steps,body='periodic_step_body')
        definitions.append(Definition(id=name,node=node))
        calls.append(name)
    if a:
        calls.append('coefficient_inverse')
    return finish_quantum_plan(method,problem,reconstruction,output=output,shots=shots,rng=rng,blocks=blocks,
        definitions=definitions,calls=calls,facts=_component_facts(payload,problem=problem),
        select_work=counts['construction_work'],native_data=payload,
        coefficient_plan=payload.coefficient_plan)


def construct_periodic_phase(block,arguments,method_context):
    """Lower the address phase diagonal exp(i*phase_j) for every k-node address.

    phase_j = arg(c_j) - T*k_j*(mass + 2*diffusion) combines the coefficient
    argument with the identity part of k_j*L, so the Strang steps need only
    the three traceless terms.
    """
    _no_arguments(arguments)
    from qiskit import QuantumCircuit
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases
    payload = block._payload
    a = len(payload.amplitudes).bit_length()-1
    circuit = QuantumCircuit(a+payload.operator.periodic_stencil().num_qubits,name='periodic_phase')
    append_control_diagonal_phases(circuit,circuit.qubits[:a],payload.phases)
    return circuit


def construct_periodic_step(block,arguments,method_context):
    """Lower one symmetric Strang step for every address at once.

    The order is Z_0 half step, even-bond half step, odd-bond full step
    (the even-bond rotation conjugated by the cyclic shift), even-bond half
    step, Z_0 half step. Even-bond rotations are H, address-controlled RZ, H
    on site 0, so each address gets its own k-dependent angle.
    """
    _no_arguments(arguments)
    from qiskit import QuantumCircuit
    from nwqlib.subroutines._multiplexors import append_uniformly_controlled_rz
    from nwqlib.subroutines.block_encoding.banded import BandSpecification,build_banded_block_encoding
    payload = block._payload
    parameters = payload.operator.periodic_stencil()
    q,a = parameters.num_qubits,len(payload.amplitudes).bit_length()-1
    step = QuantumCircuit(a+q,name='periodic_strang_step')
    controls,sites = list(step.qubits[:a]),list(step.qubits[a:])
    shift = build_banded_block_encoding(BandSpecification(offsets=(1,),coefficients=(1.,),num_qubits=q)).circuit.to_gate()
    def rx(angles):
        step.h(sites[0])
        append_uniformly_controlled_rz(step,sites[0],controls,angles)
        step.h(sites[0])
    potential_half = payload.elapsed/payload.steps*parameters.potential
    step.rz(potential_half,sites[0])
    rx(payload.half_angles)
    step.append(shift.inverse(),sites)
    rx(tuple(2*value for value in payload.half_angles))
    step.append(shift,sites)
    rx(payload.half_angles)
    step.rz(potential_half,sites[0])
    return step
