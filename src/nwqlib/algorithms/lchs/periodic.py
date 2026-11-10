"""Compact periodic LCHS selection and Strang circuits, without system arrays.

The periodic operator is A = mass*I + diffusion*(2I - S - S^dagger) + i*potential*Z_0
with S the cyclic shift. Its Cartesian parts are L = mass*I +
diffusion*(2I - S - S^dagger), which is PSD analytically with
||L|| <= mass + 4*diffusion, and H = potential*Z_0. No 2**q array is formed.
The k-grid comes from the same kernel and quadrature rules as the dense route,
using the analytic norm bound for ||L||.
"""

from nwqlib._limits import ResourceLimitError, exceeds
from cmath import phase
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from hashlib import sha256
from math import inf, isfinite

from nwqlib._numerics import complex_magnitude_upper
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
from .providers import LCHSProblemContext, _resolve_coefficient_context, lifted_magnitudes
from .solution_error_budget import unit_angle


SOURCE = Source(name="lchs.periodic", version="1", domain="periodic mass/diffusion and alternating imaginary potential",
    reference="A=mass I+diffusion(2I-S-S†)+i potential Z0; analytic PSD; fixed symmetric Strang")

# Kept bytes per Aer native position of a periodic circuit. The older-code sweep
# measured 1.6, 1.1 and 0.98 KB of RSS growth per position at q=10, T=1, 3, 10.
# Registered in ENGINEERING_CONSTANTS.
_NATIVE_POSITION_BYTES = 1024


@dataclass(frozen=True)
class _PeriodicPayload:
    """Planned periodic scalars shared by both circuit blocks and the saved archive.

    half_angles are the per-address even-bond rotation angles of one half
    step. phases combine each raw coefficient's argument with its identity
    evolution phase, which is first reduced through its unit phase
    (``solution_error_budget.unit_angle``) so that a large identity phase
    cannot absorb the coefficient argument. parameter_id is
    periodic_identity of the other fields, and both blocks record it as the
    content hash of their input.
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
    """Hash every planned field of a periodic payload except parameter_id.

    The SHA-256 starts from the Method's content hash and feeds each field
    name and value through _feed_selected_value, with type tags and container
    lengths, as selected_identity does for LCHSData. The operator enters as
    the content hash of its input reference and its stencil parameters.
    Planning saves the result as parameter_id and in both block records, so
    the Plan's content hash changes with every planned periodic field,
    including the operator parameters.
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
    """Complete elementary gate positions, including QFT phases and one outer control.

    Counts are the planned gate occurrences, not optimized Qiskit counts. Every step
    uses three H-UCRZ-H rotations and two one-band shifts. A QFT pair has
    q(q-1) controlled phases, 2*floor(q/2) SWAPs and 2q H. Each controlled
    phase has a two-CX/three-phase decomposition. Global phases survive a
    control. They cost a one-qubit phase for the one-control subset.
    """
    p = 1 << a
    shifts = 2 * steps
    qft_cp = shifts * q * (q - 1)
    qft_swaps = shifts * 2 * (q // 2)
    multiplexor_cx = 3 * steps * (p if a else 0) + max(0, p - 2)
    cx = multiplexor_cx + 2 * qft_cp + 3 * qft_swaps
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
        construction_logical_bytes=payload, construction_work=work,
        native_positions=qft_cp + qft_swaps + multiplexor_cx + rotations + h)


def _leaf_memory(payload, *, phase_only):
    """Return the ``SelectedBlock.bind`` memory estimates of one periodic leaf, for selection and for a reload.

    Each call of a leaf stays in the flattened Aer circuit as top-level native
    positions, counted by operation class from _construction_law and not from
    its ``slots``, because Aer keeps a cp or swap as one position. The address
    phase has p - 1 rz and max(0, p - 2) cx. One Strang step has the rest: the
    2*q*(q-1) cp, 4*floor(q/2) swap and 2q phases of a QFT pair, 6 + 4q h, 3p cx
    and the 3p + 2 rz of its rotations, an upper count because a multiplexor
    omits a zero Walsh angle. The Program's Repeat multiplies a step by r. One
    built definition is taken as one occurrence's size.

    ``_NATIVE_POSITION_BYTES`` is a coarse estimate of the kept bytes per
    position, not a bound on process memory.
    """
    q = payload.operator.periodic_stencil().num_qubits
    a = len(payload.amplitudes).bit_length()-1
    p = 1 << a
    phase = p - 1 + max(0, p - 2)
    positions = phase if phase_only else _construction_law(q, a, 1)["native_positions"] - phase
    return dict(native_bytes=positions*_NATIVE_POSITION_BYTES, definition_bytes=positions*_NATIVE_POSITION_BYTES)


def select_periodic_parameters(operator, *, elapsed, method):
    """Select scalar/branch data from the analytic periodic L norm.

    Planning (``selection._route``) has already refused a Method other than
    fixed second-order trotter with auto/structured SELECT and direct PREP.

    Each branch generator k*L + H splits into an identity part
    k*(mass + 2*diffusion), applied as an address phase, and three Strang
    terms, applied chronologically as potential*Z_0 half, -k*diffusion*E
    half, -k*diffusion*O full, -k*diffusion*E half, potential*Z_0 half.
    E pairs each even site with the next odd site and O = S E S†, with
    S the cyclic shift. E, O and Z_0 are Hermitian involutions, and Z_0
    anticommutes with E and O. The q = 1 case has E = O and preserves the
    double-wrap edge.

    Put a = |k|*diffusion, p = |potential| and g = 0 for q = 1,2 or 1 for
    q >= 3. The ordered second-order theorem of Childs et al., Phys. Rev.
    X 11, 011020, doi:10.1103/PhysRevX.11.011020, Proposition 10, Eq. (121)
    (arXiv:1912.08854v3, Proposition 16, Eq. (152)), gives the branch
    coefficient W_q(a,p) = g*a**3/2 + 4*a**2*p/3 + a*p**2/3.

    In the cell Fourier basis theta = 2*pi*l/2**(q-1), E is sigma_x and
    O its off-diagonal reflection with phases exp(+/-i theta). Thus
    ||E+O|| = 2, ||[E,O]|| = 2*max_l|sin(theta_l)|, and both nested
    matching commutators have norm 4*max_l|sin(theta_l)|. That maximum is
    zero for q = 1,2 and one for q >= 3. In the circuit order H_1=vZ_0,
    H_2=-bE, H_3=-bO, with signed b=k*diffusion and v=potential, the first
    tail is -b(E+O). Anticommutation gives nested norms 16*a**2*p and
    8*a*p**2 there, and 4*g*a**3 for each nested norm at the second term.
    Hence W = (16*a**2*p+4*g*a**3)/12 + (8*a*p**2+4*g*a**3)/24.
    Telescoping r unitary steps bounds each branch by W_q*T**3/r**2.

    The weighted branch sum follows from the triangle inequality. The LCU
    applies sum_j c_j*U_j, so replacing each U_j by its r-step Strang
    product S_j changes the operator by at most sum_j |c_j|*||U_j - S_j||,
    which is the bound above for a unit input.

    Step count. Write B = T**3*sum_j |c_j|*W_q(a_j,p), so the bound is
    B/r**2. With ``Trotter.steps`` given, r is that value. With
    ``Trotter.synthesis_tol`` eps instead, r is the smallest positive
    integer with B/r**2 <= eps, which is r = ceil(sqrt(n)) with
    n = ceil(B/eps) (r = 1 when B = 0). B depends only on the planned grid,
    T and the stencil, so r is chosen after the grid and before the
    construction limits (Trotter.max_steps, then the construction formula's
    max_select_work) are applied. Planning builds no circuit, so the bytes of
    the circuit are compared only at preparation (_leaf_memory), and a plan or
    estimate of a large stencil needs none of that memory.

    Evaluation. B is an exact rational: T, the diffusion and potential
    coefficients and the nodes k_j are binary64 values read exactly, a_j
    and the polynomial are formed exactly, and |c_j| is replaced by the exact
    upper value of ``_numerics.complex_magnitude_upper``. No intermediate can round to zero or
    overflow, and a float a_j need not be representable for T**3*W_q to be.
    The inversion is exact rational arithmetic on B
    and eps (trotterization.error_budget._smallest_step_count, shared with
    the step rule of Childs et al., Sec. V B), so for r > 1, B/(r - 1)**2
    exceeds eps. The recorded bound is B/r**2 rounded upward, so it is at
    most eps and positive whenever B is. With ``Trotter.steps`` given, a
    B/r**2 above the largest binary64 number is recorded as unavailable. The
    bound is sufficient and not tight, so a smaller r may also keep the
    synthesis error itself within eps.

    Diffusion-only branches have B = 0 at q = 1,2 and select one step, while
    nonzero potential keeps both cross terms. Zero diffusion gives W = 0
    at every q, while selection still requires finite positive
    mass+4*diffusion. Signed generators and the identity phase are kept.
    A zero ideal product-formula component does not bound phase/angle,
    execution or readout roundoff, which have separate components.
    """
    parameters = operator.periodic_stencil()
    elapsed = finite_real(elapsed, "elapsed time")
    if elapsed <= 0:
        raise ValueError("periodic LCHS requires positive elapsed time")
    q = parameters.num_qubits
    l_norm = parameters.mass + 4 * parameters.diffusion
    if not isfinite(l_norm) or l_norm <= 0:
        raise ValueError("periodic LCHS requires finite positive mass+4*diffusion")
    plan = _resolve_coefficient_context(lchs_kernel=method.lchs_kernel,
        k_quadrature=method.k_quadrature,max_bytes=method.max_bytes,max_quadrature_work=method.max_quadrature_work,
        problem_context=LCHSProblemContext(final_time=elapsed,epsilon=method.approx_tol,
            l_norm=l_norm))
    a = (len(plan.coefficients)-1).bit_length()
    # B = T**3 * sum_j |c_j|*W_q(a_j,p), exactly relative to stored floats.
    diffusion, potential = Fraction(parameters.diffusion), abs(Fraction(parameters.potential))
    exact = Fraction(elapsed)**3*sum((complex_magnitude_upper(complex(c))*
        ((node_diffusion**3/2 if q >= 3 else 0)
         + 4*node_diffusion**2*potential/3 + node_diffusion*potential**2/3)
        for k,c in zip(plan.nodes,plan.coefficients,strict=True)
        for node_diffusion in (abs(Fraction(k))*diffusion,)), Fraction(0))
    formula = method.ham_evo
    steps = formula.steps
    if steps is None:
        # Smallest r >= 1 with B/r**2 <= tolerance.
        steps = _smallest_step_count(exact,formula.synthesis_tol,2)
        if formula.max_steps is not None and steps > formula.max_steps:
            raise ResourceLimitError(
                f"periodic Strang synthesis needs {steps} steps for "
                f"Trotter.synthesis_tol={formula.synthesis_tol!r}, "
                f"above Trotter.max_steps={formula.max_steps}"
            )
    counts = _construction_law(q,a,steps)
    required_work = counts["construction_work"]
    if exceeds(required_work, method.max_select_work):
        raise ResourceLimitError(
            f"periodic Strang construction for q={q}, address_qubits={a}, steps={steps} "
            f"requires {required_work} work units, above max_select_work={method.max_select_work}. "
            "Raise max_select_work or reduce the construction"
        )
    amplitudes = plan.prep_amplitudes(max_bytes=method.max_bytes)
    angles = tuple(-elapsed/steps*k*parameters.diffusion for k in plan.nodes)
    phases = tuple(phase(c)+unit_angle(-elapsed*k*(parameters.mass+2*parameters.diffusion))
        for k,c in zip(plan.nodes,plan.coefficients,strict=True))
    if not all(isfinite(value) for value in (*angles,*phases,elapsed/steps*parameters.potential)):
        raise ValueError('periodic LCHS selected rotations and phases must be finite')
    # B/r**2 rounded upward, so a step count selected from the tolerance records a bound at most the
    # tolerance, and an unrepresentable bound for given Trotter.steps is unavailable.
    bound = _upward_float(exact/steps**2)
    bound = None if bound == inf else bound
    payload = _PeriodicPayload(operator,elapsed,steps,plan,amplitudes,angles,phases,bound,'')
    return replace(payload,parameter_id=periodic_identity(payload,method)),counts


def _coefficient_mass(plan):
    """Return the raw coefficient one-norm sum_j |c_j| as a PhysicalScale.

    The magnitudes are formed after the exact lift of
    ``providers.lifted_magnitudes`` and the lift is removed as a power of
    two, so a subnormal complex coefficient keeps its magnitude. The
    PREP amplitudes are normalized by the same lifted sum, and
    ``coefficient_l1_norm`` is only its rounded binary64 display.
    """
    from math import fsum
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery
    magnitudes, lift = lifted_magnitudes(plan.coefficients)
    return compose_recovery(fsum(magnitudes), PhysicalScale(mantissa=.5, exponent=1-lift))


def _component_facts(payload, *, problem):
    """Report the periodic circuit-path stages with the analytic PSD assumption.

    L is PSD by construction, so there is no shift and every PSD recovery
    factor exp(shift*t) is one. Direct PREP stages are exact zero, and the
    Strang bound is the trotter_synthesis stage. The input norm and gamma
    stay PhysicalScale values, so each physical stage is composed before its
    one binary64 conversion and stays reported whenever it is
    representable, even if the norm alone is not.
    """
    from .analysis import fact
    from .solution_error_budget import _build_lchs_application_record,circuit_stage_bounds
    from .time_independent_terms import _unusable_kernel_stage_names
    from nwqlib.problems.inputs import compose_recovery
    from nwqlib.problems.records import Solution
    raw = payload.coefficient_plan.record()
    raw.update(psd_shift=0.,approximate_lchs_error_bound=payload.coefficient_plan.quadrature.approximate_lchs_error_bound)
    frame = Solution().frame(problem)
    scale = problem.initial_state.preparation.physical_scale
    application = _build_lchs_application_record(raw,final_time=payload.elapsed,start_time=0.,
        weight=1.,input_norm=scale,backend='trotter',synthesis_error_bound=payload.synthesis_bound)
    values = circuit_stage_bounds(raw,applications=(application,),backend='trotter',inhomogeneous=False,
        psd_premise_satisfied=True,gamma=compose_recovery(_coefficient_mass(payload.coefficient_plan),scale),
        delta_lcu=0.,input_preparation_output_error_bound=0.,unusable_stages=_unusable_kernel_stage_names(raw))
    return tuple(fact(name,frame,value,reason=reason or 'selected periodic component unavailable',source=SOURCE)
        for name,(value,reason) in values.items())


def _select_periodic_leaf(name,payload,*,basis,phase_only):
    """Declare the address-phase block or the one-step Strang block with its CX formulas.

    The step block describes one Strang step, and the Program's Repeat node
    supplies the step count, so the resource total multiplies it once.
    """
    q = payload.operator.periodic_stencil().num_qubits
    a = len(payload.amplitudes).bit_length()-1
    p = 1<<a
    one = _construction_law(q,a,1)
    # Factor the complete one-step law of _construction_law into the PREP
    # pair, the phase and one step. The Program's Repeat node supplies the
    # step count when the Program's resource counts are summed.
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
        constructor=construct_periodic_phase if phase_only else construct_periodic_step,
        **_leaf_memory(payload,phase_only=phase_only))


def plan_periodic(method,problem,*,output,shots,rng):
    """Select compact homogeneous periodic Strang evolution with physical scalar/vector recovery."""
    from nwqlib.blocks.selection import transform_block
    from nwqlib.problems.inputs import compose_recovery
    from .primary_records import LCHSReconstruction
    from .quantum import finish_quantum_plan,observable_terms,select_vector_preparation
    payload,counts = select_periodic_parameters(problem.A,elapsed=problem.elapsed_time,method=method)
    q,a = problem.A.periodic_stencil().num_qubits,counts['address_qubits']
    success,coordinates = tuple(range(a)),tuple(range(a,a+q))
    terms = observable_terms(problem,output,max_bytes=method.max_bytes)
    quad = payload.coefficient_plan.quadrature
    reconstruction = LCHSReconstruction(elapsed_time=problem.elapsed_time,dimension=problem.dimension,encoded_dimension=1<<q,
        initial_scale=problem.initial_state.preparation.physical_scale,
        recovery=compose_recovery(_coefficient_mass(payload.coefficient_plan),problem.initial_state.preparation.physical_scale),
        success_bits=success,system_bits=coordinates,terms=terms,
        coefficient_l1_norm=payload.coefficient_plan.coefficient_l1_norm,physical_branches=quad.physical_node_count,
        padded_branches=1<<a,selected_backend='trotter',selected_select='periodic_strang',psd_premise='analytic_periodic',
        l_norm=problem.A.periodic_stencil().mass+4*problem.A.periodic_stencil().diffusion,
        kernel_approximation_bound=quad.approximate_lchs_error_bound,quadrature_bound=quad.quadrature_error_bound,
        step_counts=(payload.steps,)*len(payload.coefficient_plan.nodes))
    prep = problem.initial_state.preparation
    _check_bytes(prep.payload_bytes,method.max_bytes,'periodic input PREP')
    if exceeds(prep.work, method.max_select_work):
        raise ResourceLimitError(f'periodic input PREP exceeds max_select_work: it needs {prep.work} work units, '
                                 f'max_select_work={method.max_select_work}')
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
    """Build the address phase diagonal exp(i*phase_j) for every k-node address.

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
    """Build one symmetric Strang step for every address at once.

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
