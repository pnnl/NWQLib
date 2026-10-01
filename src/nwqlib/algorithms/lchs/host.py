"""Finite classical LCHS acquisition and exact initial-condition identities."""

from math import fsum
from dataclasses import replace
from typing import NamedTuple
import numpy as np

from nwqlib.artifacts import ArrayOutput, UnavailableOutput
from nwqlib.blocks.kernels import BoundKernel, KernelOutput
from nwqlib.blocks.records import SelectedConstruction, SelectedKernel
from nwqlib.core.planning import Plan, Experiment, ObservationSpec
from nwqlib.core.records import Float64, InputRef
from nwqlib.evidence.error_model import exact_readout_sampling
from nwqlib.execution import KernelApplication, ScalarValue
from nwqlib.evidence.records import Evidence
from nwqlib.ir import Binding, ClassicalStage, Definition, Sequence
from nwqlib.ir.validation import admitted_program
from nwqlib.operators.access import _check_bytes
from nwqlib.problems.records import Solution, StateVector, NormSquared, NormalizedExpectation, QuadraticForm, Samples
from nwqlib.problems.inputs import compose_recovery
from nwqlib.problems.scalars import physical_scalar, physical_vector_statistics
from nwqlib._run_journal import _json_bound
from nwqlib.resources.records import ResourceLaw, Workspace
from .primary_records import METHOD, LCHSReconstruction, LCHSAnalysis
from .analysis import error_model, fact, result_fields
from .solution_error_budget import psd_recovery, psd_recovery_exponent, psd_recovery_part, psd_recovery_scale


def plan_initial(method,problem,*,output,execution,shots,rng,zero):
    """Plan the no-evolution case, zero elapsed time or all physical inputs zero.

    The solution is the initial vector itself, so no quadrature, eigensolve
    or SELECT is selected. A known vector and its norm are answered from the
    ingested input without acquisition. A classical observable of an explicit
    vector uses one recorded host computation. Samples, quantum observables
    and compact states acquire through the actual initial-state PREP. A
    compact zero state has no normalized PREP, so a requested physical zero
    array is materialized once and unit outputs are undefined.
    """
    state = problem.initial_state
    scale = state.preparation.physical_scale
    if isinstance(output,Samples) and scale.mantissa==0:
        raise ValueError('normalized samples are undefined for a zero initial condition')
    rec = LCHSReconstruction(elapsed_time=problem.elapsed_time,dimension=problem.dimension,encoded_dimension=problem.dimension,
        mode='zero' if zero else 'initial',has_source=problem.source is not None,initial_scale=scale,
        source_scale=None if problem.source is None else problem.source.preparation.physical_scale,
        selected_backend='initial_condition',selected_select='no_evolution')
    vector=isinstance(output,(Solution,StateVector))
    observable=isinstance(output,(NormalizedExpectation,QuadraticForm))
    if scale.mantissa==0:
        # A compact zero has no normalized PREP. Materialize its physical
        # vector only when that actual array output was requested.
        if vector and not (isinstance(output,StateVector) and output.normalization=='unit') and state.reference.representation!='vector':
            return _plan_initial_host(method,problem,rec,output=output,execution=execution,shots=shots,rng=rng,
                kind='initial_zero_vector')
        acquire=False
    elif observable and execution=='classical' and state.reference.representation=='vector':
        return _plan_initial_host(method,problem,rec,output=output,execution=execution,shots=shots,rng=rng,
            kind='initial_observable')
    else:
        acquire=isinstance(output,Samples) or observable or (state.reference.representation!='vector' and not isinstance(output,NormSquared))
    if acquire:
        if execution != 'quantum':
            from nwqlib.algorithms.protocol import ApplicabilityError
            raise ApplicabilityError('classical initial-condition output needs an explicit vector; select quantum execution to acquire a prepared state')
        return _initial_acquisition(method,problem,rec,output=output,shots=shots,rng=rng)
    construction = SelectedConstruction(program=admitted_program('LCHS.max_admission_steps',method.max_admission_steps,root='initial',definitions=(Definition(id='initial',node=Sequence()),)),selections=())
    plan = Plan(problem=problem,method=method,output=output,execution=execution,shots=shots,randomness=rng.snapshot(),
        construction=construction,experiments=(),reconstruction=rec,error_model=error_model(problem,output,construction,shots=None),
        assumptions=('initial-condition identity; no evolution or circuit acquisition',))
    return plan._bind()


def _initial_acquisition(method,problem,rec,*,output,shots,rng):
    """Plan readout of the prepared initial state, with recovery by its physical norm."""
    from nwqlib.blocks import select_preparation
    from nwqlib.ir import BlockCall, PortMap
    from nwqlib.problems.inputs import compose_recovery
    from .quantum import finish_quantum_plan, observable_terms, select_vector_preparation
    state = problem.initial_state
    q = max(1,(problem.dimension-1).bit_length())
    if state.reference.representation=='vector':
        from .selection import _pad_vector
        target = _pad_vector(state._direction,1<<q)
        block = select_vector_preparation('initial_prep',target,method=method)
    else:
        block = select_preparation('initial_prep',state,per_bit=True)
    coordinates = tuple(range(q))
    terms = observable_terms(problem,output,max_bytes=method.max_bytes)
    rec = rec.revise(encoded_dimension=1<<q,system_bits=coordinates,recovery=compose_recovery(state.preparation.physical_scale),
        terms=terms,selected_backend='initial_preparation_readout')
    definition = Definition(id='initial_prep',node=BlockCall(signature='initial_prep',
        ports=tuple(PortMap(port=f'system_{j}',wire=f'bit_{j}') for j in range(q))))
    return finish_quantum_plan(method,problem,rec,output=output,shots=shots,rng=rng,blocks=[block],
        definitions=[definition],calls=['initial_prep'])


def analyze_initial(plan,data):
    """Interpret a no-evolution Plan from its acquisition or its ingested input.

    Without acquired chunks, the requested scalar comes from the input's
    physical scale, and the algorithmic_approximation and sampling facts are
    exact zero because no approximation or sampling took place.
    """
    if data.observations.chunks:
        from .analysis import _chunks, analyze_quantum
        chunks=_chunks(plan,data)
        if plan.construction.kernels:
            return analyze_classical(plan,data,chunks)
        return analyze_quantum(plan,data,chunks)
    state,output = plan.problem.initial_state,plan.output
    scale = state.preparation.physical_scale
    norm = scale.squared_as_float()
    numerator = None
    frame = None
    unavailable = None
    if isinstance(output,(NormalizedExpectation,QuadraticForm)):
        frame = 'unit' if isinstance(output,NormalizedExpectation) else 'physical'
        if scale.mantissa==0:
            numerator = 0. if frame=='physical' else None
        else:
            raise ValueError('a nonzero initial observable requires its selected acquisition')
    scalar = isinstance(output,(NormSquared,NormalizedExpectation,QuadraticForm))
    value,unavailable = physical_scalar(output.kind,norm_squared=norm,scale=scale,numerator=numerator,
        numerator_frame=frame or 'physical') if scalar else (None,None)
    if isinstance(output,StateVector) and output.normalization=='unit' and scale.mantissa==0:
        unavailable = 'unit vector is undefined for a zero physical vector'
    return LCHSAnalysis(**result_fields(plan,data),value=value,unavailable=unavailable,norm_squared=norm,physical_scale=scale,
        numerator=numerator,numerator_frame=frame,facts=(fact('algorithmic_approximation',plan.error_model.frame,0.),
            *exact_readout_sampling(plan,data.observations,random_draws=False)))


def _initial_kernel(method,problem,output,kind):
    """Declare actual initial-output work from input metadata, without action."""
    state=problem.initial_state
    dimension=problem.dimension
    outputs=()
    if kind=='initial_observable':
        state.physical_vector()  # Existing immutable data access, no copy.
        from nwqlib.operators.inputs import _scaled_observable_requirements
        workspace,products=_scaled_observable_requirements(output.observable)
        # Norm/subnormal refinement and the original binary vector frame.
        work=products+8*dimension+2
        workspace+=48*dimension
        inputs=(state.reference,output.observable.reference)
        labels=('norm_squared','numerator')
        frames=('physical','unit' if isinstance(output,NormalizedExpectation) else 'physical')
        dependencies=('numpy','scipy') if output.observable.reference.representation in ('csr','csc') else ('numpy',)
    elif kind=='initial_zero_vector':
        if state.preparation.physical_scale.mantissa!=0 or not isinstance(output,(Solution,StateVector)):
            raise ValueError('initial zero materialization requires a physical zero vector output')
        if isinstance(output,StateVector) and output.normalization!='physical':
            raise ValueError('unit zero vector is undefined')
        work,workspace=dimension,32*dimension
        inputs,labels,frames,dependencies=(state.reference,),('norm_squared',),('physical',),('numpy',)
        outputs=(ArrayOutput(name='solution' if isinstance(output,Solution) else 'state_vector',kind='vector',
            basis=problem.basis,frame='physical',global_phase='physical' if isinstance(output,Solution) else output.global_phase),)
    else:
        raise ValueError('unknown initial-output kernel')
    if work>method.max_select_work:
        raise ValueError('initial output computation exceeds max_select_work')
    _check_bytes(workspace,method.max_bytes,'initial output arrays')
    receipt = _initial_application(kind,dimension,work)
    return SelectedKernel(name=kind,implementation=METHOD,inputs=inputs,scalars=labels,scalar_frames=frames,
        outputs=outputs,construction_work=0,invocation_work=work,dependencies=dependencies,
        application_bytes=_json_bound((receipt,),method.max_bytes),
        resource_laws=(ResourceLaw(metric='classical_work',basis='selected_logical',value=work,
            interpretation='upper_bound',evidence=Evidence(kind='proved_relation',source=METHOD),
            assumptions=('actual matrix-vector products and vector scans, or zero-array entries; CPU/RSS unknown',)),),
        workspace=(Workspace(location='host',purpose='workspace',bytes=workspace,source=METHOD),
            Workspace(location='host',purpose='workspace',bytes=None,source=METHOD)))


def _initial_application(kind,dimension,work):
    """Return the one receipt of a no-evolution host kernel, which records its dimension and declared work.

    It is fixed at planning, so the kernel declares its bytes for the Run's
    output reservation.
    """
    return KernelApplication(name=kind,implementation=METHOD,
        arguments=(Binding(parameter='dimension',value=dimension),Binding(parameter='size_units',value=work)))


def _plan_initial_host(method,problem,rec,*,output,execution,shots,rng,kind):
    """Plan one host computation for a no-evolution output of the given kind.

    kind is ``initial_observable`` (an observable of the supplied vector) or
    ``initial_zero_vector`` (a requested physical zero array).
    """
    kernel=_initial_kernel(method,problem,output,kind)
    rec=rec.revise(selected_backend=kind,selected_select='no_evolution',
        classical_work=(('size_units',kernel.invocation_work),))
    program=admitted_program('LCHS.max_admission_steps',method.max_admission_steps,root=kind,definitions=(Definition(id=kind,node=ClassicalStage(implementation=METHOD,boundary='host',kernel=kind)),))
    construction=SelectedConstruction(program=program,selections=(),kernels=(kernel,))
    chosen=Plan(problem=problem,method=method,output=output,execution=execution,shots=shots,randomness=rng.snapshot(),
        construction=construction,experiments=(Experiment(name='initial_output',setting='initial_output',
            observation=ObservationSpec(kind='host_scalars',labels=kernel.scalars)),),reconstruction=rec,
        error_model=error_model(problem,output,construction),
        assumptions=('initial-condition identity; one actual host output computation, no evolution or reference',))
    return chosen._bind(blocks=(bind_host(chosen,None),))


def _execute_initial_output(plan,kernel):
    """Run the declared no-evolution kernel and return its KernelOutput.

    Because u(T) = u(0), the physical norm squared comes directly from the
    ingested scale of u(0). ``initial_observable`` adds the requested
    observable moment of the supplied vector in the declared frame.
    ``initial_zero_vector`` returns the zero array of the original dimension.
    The recorded application states the dimension and the declared work.
    """
    state,output=plan.problem.initial_state,plan.output
    scale=state.preparation.physical_scale
    norm=scale.squared_as_float()
    values=[ScalarValue(label='norm_squared',value=norm,
        unavailable=None if norm is not None else 'physical norm squared is not representable in binary64')]
    arrays=()
    if kernel.name=='initial_observable':
        _,_,statistics=physical_vector_statistics(state.physical_vector(),observable=output.observable,
            numerator_frame=kernel.scalar_frames[-1])
        # The original admitted state owns norm/scale; only the requested
        # observable moment comes from this invocation's framed reduction.
        values.append(statistics[-1])
    else:
        arrays=((kernel.outputs[0].name,np.zeros(plan.problem.dimension,dtype=complex)),)
    application=_initial_application(kernel.name,plan.problem.dimension,kernel.invocation_work)
    return KernelOutput(plan_id=plan.content_id,selected_kernel_id=kernel.content_id,scalars=tuple(values),
        physical_scale=scale,arrays=arrays,applications=(application,))


def plan_classical(method,problem,data,*,output,rng):
    """Declare the bounded finite LCHS/Duhamel host action and requested physical outputs.

    Classical execution evaluates the same selected finite recipe that the
    quantum route would, not an exact reference. Dense nodes form one
    Hermitian eigensystem per node and reuse it for every application
    (time_independent_terms.spectral_lchs_sum). Product-formula nodes use the
    Pauli actions selected once by host_pf.select_actions. The whole repeated work
    and workspace are admitted before the kernel is declared, and the kernel
    input includes selected_identity, so a saved host Plan is tied to these
    exact numbers. The Plan's kernel, k-quadrature and product-formula facts
    are the application sums that execution later records, computed from
    the same selected records (_planned_applications) without a host action.
    Unlike a quantum Plan, it has no preparation or QSP stages.
    """
    from .inhomogeneous_theory import duhamel_quadrature
    source_scale = None if problem.source is None else problem.source.preparation.physical_scale
    source_nodes = source_weights = ()
    if source_scale is not None and source_scale.mantissa!=0:
        # Legendre rule for m nodes: 32*m**2 for the m-by-m Jacobi eigensolve
        # arrays, 32*m for nodes and weights and their mapped copies, and m**3
        # work for the eigensolve.
        _check_bytes(32*method.duhamel_nodes**2+32*method.duhamel_nodes,method.max_bytes,'Duhamel quadrature')
        if method.duhamel_nodes**3>method.max_quadrature_work:
            raise ValueError(
                f'Duhamel quadrature exceeds max_quadrature_work: it needs {method.duhamel_nodes**3} work units '
                f'(duhamel_nodes**3), LCHS.max_quadrature_work={method.max_quadrature_work}. '
                f'Raise LCHS.max_quadrature_work to at least {method.duhamel_nodes**3}.')
        source_nodes,source_weights = duhamel_quadrature(problem.elapsed_time,method.duhamel_nodes)
    rec = LCHSReconstruction(elapsed_time=problem.elapsed_time,dimension=problem.dimension,encoded_dimension=len(data.initial),
        has_source=problem.source is not None,initial_scale=problem.initial_state.preparation.physical_scale,source_scale=source_scale,
        coefficient_l1_norm=data.coefficient_plan.coefficient_l1_norm,physical_branches=len(data.coefficient_plan.coefficients),
        padded_branches=1<<(len(data.coefficient_plan.coefficients)-1).bit_length(),
        selected_backend=method.hamiltonian_evolution_backend,selected_select='finite_classical_quadrature',
        psd_premise='numerical' if data.quadrature.numerical_psd_premise_satisfied else 'unavailable',
        psd_shift=data.quadrature.conversion['psd_shift'],l_norm=data.quadrature.l_norm,
        source_nodes=tuple(map(float,source_nodes)),source_weights=tuple(map(float,source_weights)),
        kernel_approximation_bound=data.quadrature.coefficient_plan.quadrature.approximate_lchs_error_bound,
        quadrature_bound=data.quadrature.coefficient_plan.quadrature.quadrature_error_bound)
    observable = isinstance(output,(NormalizedExpectation,QuadraticForm))
    labels = ('norm_squared','numerator') if observable else ('norm_squared',)
    frames = ('physical','unit' if isinstance(output,NormalizedExpectation) else 'physical') if observable else ('physical',)
    outputs = ()
    if isinstance(output,(Solution,StateVector)):
        outputs = (ArrayOutput(name='solution' if isinstance(output,Solution) else 'state_vector',kind='vector',basis=problem.basis,
            frame='physical' if isinstance(output,Solution) else output.normalization,
            global_phase='physical' if isinstance(output,Solution) else output.global_phase),)
    # Count repeated dense-kernel work across all coefficient and source nodes
    # before admitting the host invocation, not only the size of one matrix.
    if method.hamiltonian_evolution_backend != 'dense_exact':
        from .host_pf import select_actions
        from .parameters import freeze_selected
        actions,counts,workspace = select_actions(data,elapsed=problem.elapsed_time,
            source_nodes=source_nodes,source_weights=source_weights,has_initial=rec.initial_scale.mantissa!=0)
        data = replace(data,host_actions=freeze_selected(actions,max_bytes=method.max_bytes))
    else:
        # Dense host workspace: the live L and H (32*D**2 bytes) and eight
        # complex128 D-vectors, plus the phase bytes of the streamed spectral
        # sum (_spectral_host_requirements).
        starts = ((0.,) if rec.initial_scale.mantissa!=0 else ())+tuple(map(float,source_nodes))
        counts,phase_bytes = _classical_work(data,problem.elapsed_time,starts)
        workspace = 32*len(data.initial)**2+128*len(data.initial)+phase_bytes
    work = counts['size_units']
    if observable:
        from nwqlib.operators.inputs import _scaled_observable_requirements
        observable_bytes,observable_work = _scaled_observable_requirements(output.observable)
        work += observable_work
        counts['size_units'] = work
        counts['observable_actions'] = 1
        workspace += observable_bytes
    rec = rec.revise(classical_work=tuple(counts.items()))
    if work>method.max_select_work:
        raise ValueError('selected classical LCHS dense-kernel work exceeds max_select_work')
    _check_bytes(workspace,method.max_bytes,'classical LCHS arrays')
    # Every component input and application receipt is selected by now. The
    # Plan publishes the same application sums that execution records, and
    # the kernel declares the receipts' JSON size bound, the measure that
    # completion admission uses. The Run reserves it with the outputs, so the
    # receipts do not spend the provider-metadata allowance.
    receipts = tuple(_application(method,problem,rec,index,planned)
        for index,planned in enumerate(_planned_applications(method,problem,rec,data)))
    components = _summed_components(method,problem,[receipt.facts for receipt in receipts])+_duhamel_component(problem,rec)
    from .parameters import selected_identity
    selection = InputRef(identity=selected_identity(data,problem,method),representation='selected_lchs_parameters',source=METHOD)
    kernel = SelectedKernel(name='lchs',implementation=METHOD,
        inputs=tuple({state.reference.identity: state.reference
            for state in (problem.A,problem.initial_state,problem.source) if state is not None}.values())+(selection,),
        scalars=labels,scalar_frames=frames,outputs=outputs,construction_work=counts.get('selection_work',0),
        invocation_work=work-counts.get('selection_work',0),dependencies=('numpy','scipy'),
        application_bytes=_json_bound(receipts,method.max_bytes),
        resource_laws=(ResourceLaw(metric='classical_work',basis='selected_logical',value=work-counts.get('selection_work',0),interpretation='upper_bound',
            evidence=Evidence(kind='proved_relation',source=METHOD),assumptions=('selected vector/rotation/dense-kernel size proxy; CPU and opaque library workspaces are not bounded',)),),
        workspace=(Workspace(location='host',purpose='workspace',bytes=workspace,source=METHOD),
                   Workspace(location='host',purpose='workspace',bytes=None,source=METHOD)))
    program = admitted_program('LCHS.max_admission_steps',method.max_admission_steps,root='lchs',definitions=(Definition(id='lchs',node=ClassicalStage(implementation=METHOD,boundary='host',kernel='lchs')),))
    construction = SelectedConstruction(program=program,selections=(),kernels=(kernel,))
    plan = Plan(problem=problem,method=method,output=output,execution='classical',shots=None,randomness=rng.snapshot(),construction=construction,
        experiments=(Experiment(name='solution',setting='solution',observation=ObservationSpec(kind='host_scalars',labels=labels)),),
        reconstruction=rec,error_model=error_model(problem,output,construction,components),facts=components)
    return plan._bind(blocks=(bind_host(plan,data),),native_data=data,coefficient_plan=data.coefficient_plan if problem.source is None else None)


def _classical_work(data,elapsed,starts):
    """Finite dense-exact host size law of the streamed spectral sum.

    starts holds the start time s of each application, 0 for the initial
    state and the Duhamel node for a source application, each applied over
    its elapsed time T - s. time_independent_terms.spectral_lchs_sum forms
    one Hermitian eigensystem per k node (one for an exactly zero L, none for
    identity actions), projects each of the r distinct input vectors once per
    node and makes one final eigenvector action per node; its work and phase
    bytes are _spectral_host_requirements. 4*D covers the final vector scans.

    Returns:
        The counts recorded in the reconstruction, whose size_units is the
        work, and the phase bytes of the streamed sum beyond its live inputs.
    """
    from .time_independent_terms import _spectral_host_requirements
    dimension,nodes = len(data.initial),data.quadrature.k_nodes
    quad = data.quadrature
    identity = not (np.any(quad.l_part) or np.any(quad.h_part)) or not any(elapsed-start for start in starts)
    eigensystems = 0 if identity else (1 if not np.any(quad.l_part) else len(nodes))
    # The initial and the source vector are the distinct inputs.
    inputs = len({start==0. for start in starts}) if data.source is not None else 1
    work,phase_bytes = _spectral_host_requirements(dimension,nodes=len(nodes),applications=len(starts),
        inputs=inputs,eigensystems=eigensystems)
    matvecs = 0 if identity else len(nodes)*(inputs+1)
    return dict(dimension=dimension,operator_applications=len(starts),evolution_calls=len(starts)*len(nodes),
        dense_expm_calls=0,eigh_calls=eigensystems,matvecs=matvecs,branch_matrix_products=0,
        size_units=work+4*dimension),phase_bytes


def bind_host(plan,data):
    """Bind the Plan's single host kernel to its executor.

    For a no-evolution Plan the kernel is declared again from the Problem and
    output and must equal the saved one, so a loaded archive cannot run a
    kernel whose inputs or work differ. Otherwise the executor evaluates the
    selected finite LCHS sum from data.
    """
    kernel, = plan.construction.kernels
    if plan.reconstruction.mode in ('initial','zero'):
        expected=_initial_kernel(plan.method,plan.problem,plan.output,plan.reconstruction.selected_backend)
        if kernel!=expected:
            raise ValueError('saved initial-output kernel differs from its actual inputs/output/work')
        return BoundKernel._bind(plan,kernel,lambda:_execute_initial_output(plan,kernel))
    return BoundKernel._bind(plan,kernel,lambda:_execute(plan,kernel,data))


class _HostApplication(NamedTuple):
    """One application of the selected finite propagator on the host.

    Attributes:
        start: Time at which the input enters, 0 for the initial state.
        weight: Signed Duhamel quadrature weight, 1 for the initial state.
        source: Whether the input is the constant source b.
        norm: Physical input norm ||u0|| or ||b||, None when it is not
            representable in binary64.
        raw: Quadrature record holding the operator-level bounds that this
            application's facts weight by |weight|*norm and exp(shift*elapsed).
            For a product formula it also holds the per-node step counts and
            the coefficient-weighted synthesis bound.
    """

    start: float
    weight: float
    source: bool
    norm: float | None
    raw: dict

    @property
    def scale(self):
        """Return |weight|*norm, the physical input weight w_a of solution_error_budget, or None."""
        return None if self.norm is None else abs(self.weight)*self.norm


def _planned_applications(method,problem,rec,data):
    """Yield the selected host applications in execution order, without evolving anything.

    The initial state comes first when it is nonzero, then one application
    per Duhamel node, the order of host_pf.select_actions and
    selected_grid._specifications. Every record comes from planning
    selections: the provider bounds and PSD shift in lchs_quadrature_summary,
    and for a product formula each node's stored step record, summed by
    _trotter_budget_quadrature_terms. Planning publishes its component facts
    from these, and execution records the same ones. Yielding one
    application at a time keeps one set of node records alive.
    """
    from .inhomogeneous_theory import _kernel_context
    from .time_independent_terms import lchs_quadrature_summary, _trotter_budget_quadrature_terms
    elapsed = problem.elapsed_time
    norms = (rec.initial_scale.as_float(),None if rec.source_scale is None else rec.source_scale.as_float())
    if data.host_actions is not None:
        raw_base = lchs_quadrature_summary(data.quadrature,method=data.method,he_backend=method.hamiltonian_evolution_backend,
            final_time=elapsed)
        for action in data.host_actions["applications"]:
            raw = dict(raw_base)
            raw.update(_trotter_budget_quadrature_terms([position["record"] for position in action["positions"]],
                coefficients=data.quadrature.coefficients,method=data.method))
            yield _HostApplication(action['start'],action['weight'],action['source'],norms[int(action['source'])],raw)
        return
    if data.source is None:
        # The shared quadrature record, without any node evolution.
        raw = lchs_quadrature_summary(data.quadrature,method=data.method,he_backend='dense_exact',
            final_time=elapsed)
        yield _HostApplication(0.,1.,False,norms[0],raw)
        return
    # The shared record of every Duhamel application. _kernel_context forms
    # no propagator; execution applies the node evolutions to vectors
    # (time_independent_terms.spectral_lchs_sum).
    raw = dict(_kernel_context(final_time=elapsed,method=data.method,_quadrature_plan=data.quadrature).quadrature)
    if rec.initial_scale.mantissa!=0:
        yield _HostApplication(0.,1.,False,norms[0],raw)
    for node,weight in zip(rec.source_nodes,rec.source_weights,strict=True):
        yield _HostApplication(float(node),float(weight),True,norms[1],raw)


def _component_names(method):
    """Return the application-weighted component stages of the selected backend."""
    return ('kernel_approximation','k_quadrature')+(('trotter_synthesis',) if method.hamiltonian_evolution_backend!='dense_exact' else ())


def _application_facts(method,problem,rec,planned):
    """Return the physical component facts of one host application.

    application_stage_bounds multiplies each stored operator-level bound by
    the input weight |w|*||u|| and the PSD recovery exp(shift*elapsed). A
    stage without a usable value is unknown with its stage reason, for
    example missing_stage:trotter_synthesis for a fixed step count.
    """
    from .solution_error_budget import _build_lchs_application_record,application_stage_bounds
    from .time_independent_terms import _unusable_kernel_stage_names
    names = _component_names(method)
    frame = Solution().frame(problem)
    raw = planned.raw
    if planned.scale is None:
        return tuple(fact(name,frame,reason='physical application scale is not representable') for name in names)
    record = _build_lchs_application_record(raw,final_time=problem.elapsed_time,start_time=planned.start,weight=1.,
        input_norm=planned.scale,backend=method.hamiltonian_evolution_backend,
        synthesis_error_bound=raw.get('trotter_synthesis_error_bound'))
    values = application_stage_bounds(raw,applications=(record,),backend=method.hamiltonian_evolution_backend,
        psd_premise_satisfied=rec.psd_premise=='numerical',unusable_stages=_unusable_kernel_stage_names(raw))
    return tuple(fact(name,frame,values[name][0],reason=values[name][1] or 'unavailable component') for name in names)


def _summed_components(method,problem,application_facts):
    """Sum each component over all applications.

    plan_classical passes the facts of the planned applications, and
    analyze_classical passes those of the acquired ones, so both records use
    one rule. A component is concrete only when every application reports a concrete
    value for it. Otherwise it is unknown with the first unknown
    application's reason, since a missing term is not zero.
    """
    frame = Solution().frame(problem)
    components = []
    for name in _component_names(method):
        facts = [item for application in application_facts for item in application if item.fact.quantity==name]
        missing = next((item.fact.reason for item in facts if not isinstance(item.fact.value,Float64)),None)
        if len(facts)!=len(application_facts):
            missing = 'an application did not report this component'
        components.append(fact(name,frame,reason=missing) if missing is not None
                          else fact(name,frame,fsum(item.fact.value.value for item in facts)))
    return tuple(components)


def _duhamel_component(problem,rec):
    """Return the Duhamel-quadrature fact of a constant-source Plan, or nothing.

    A zero source contributes nothing, so the stage is exactly zero. For a
    nonzero source the Gauss-Legendre remainder needs spectral norms of A,
    which only an explicit LCHSRefinement(components=("duhamel",)) evaluates.
    The stage stays unknown with the same reason as the quantum route.
    """
    if problem.source is None:
        return ()
    frame = Solution().frame(problem)
    if rec.source_scale.mantissa==0:
        return (fact('duhamel_quadrature',frame,0.),)
    return (fact('duhamel_quadrature',frame,reason='missing_stage:duhamel_quadrature'),)


def _application(method,problem,rec,index,planned):
    """Return the receipt of one host application, which records its schedule and component facts.

    The arguments (start and elapsed time, source role, weight and recovery
    scale) are the schedule that selected_grid checks against the Plan. The
    per-node step records and routes are in the Plan's host actions, beside
    the stored node table from which rotations are formed at application
    time. Execution applies them there, and selected_grid and
    refinement read the step counts there, so a receipt does not grow with
    the k-node count. Planning builds the same receipts to
    declare their bytes (SelectedKernel.application_bytes).
    """
    raw,start = planned.raw,planned.start
    facts = _application_facts(method,problem,rec,planned)
    arguments = (Binding(parameter='start_time',value=Float64(value=start)),
                 Binding(parameter='elapsed_time',value=Float64(value=problem.elapsed_time-start)),
                 Binding(parameter='source_application',value=int(planned.source)))
    # A recovery exp(shift*elapsed) beyond binary64 has no Float64 value and
    # is omitted, like an unrepresentable weight.
    recovery = psd_recovery(raw['psd_shift'],problem.elapsed_time-start)
    if recovery is not None:
        arguments += (Binding(parameter='recovery_scale',value=Float64(value=recovery)),)
    if planned.scale is not None:
        arguments += (Binding(parameter='weight',value=Float64(value=planned.scale)),)
    return KernelApplication(name=f'application_{index}',implementation=METHOD,arguments=arguments,facts=facts)


def _carried_recovery(recovery,shift,elapsed):
    """Compose into recovery the power of two 2**e that the host arrays omit.

    The arrays hold each PSD recovery exp(shift*elapsed_a) divided by 2**e,
    with e = psd_recovery_exponent(shift, elapsed), which is zero while
    exp(shift*elapsed) is representable. recovery is the initial-state norm
    (homogeneous) or None (constant source), and comes back unchanged when
    e is zero. A carried 2**e keeps a representable physical solution
    available when the growth factor alone overflows.
    """
    growth = psd_recovery_scale(psd_recovery_exponent(shift,elapsed))
    if growth is None:
        return recovery
    return growth if recovery is None else compose_recovery(recovery,growth)


def _evolve(plan,data):
    """Evaluate the selected finite LCHS/Duhamel sum on the padded host vector.

    Returns the evolved unit initial direction (homogeneous) or the physical
    vector (constant source), the recovery still to apply, and one
    KernelApplication per physical application. The recovery is the initial
    norm (homogeneous) or None (constant source), composed with any carried
    power of two 2**e of the PSD recovery, which with a constant source is
    2**e alone (_carried_recovery).
    """
    if data.host_actions is not None:
        return _evolve_pauli_actions(plan,data)
    # Planning selects Pauli actions for every product-formula backend; the
    # remaining host path applies the dense exact node evolutions to vectors
    # from one Hermitian eigensystem per node (spectral_lchs_sum).
    from .time_independent_terms import spectral_lchs_sum
    elapsed = plan.problem.elapsed_time
    shift = plan.reconstruction.psd_shift
    exponent = psd_recovery_exponent(shift,elapsed)
    quad = data.quadrature
    planned_applications = _planned_applications(plan.method,plan.problem,plan.reconstruction,data)
    if data.source is None:
        planned, = planned_applications
        application = _application(plan.method,plan.problem,plan.reconstruction,0,planned)
        recovery = _carried_recovery(plan.reconstruction.initial_scale,shift,elapsed)
        scale = psd_recovery_part(shift,elapsed,exponent) if shift > 0 else 1.
        solution = spectral_lchs_sum(quad.l_part,quad.h_part,quad.k_nodes,quad.coefficients,
            ((elapsed,scale,data.initial_direction),))
        return solution,recovery,(application,)
    specifications,applications = [],[]
    for index,planned in enumerate(planned_applications):
        if planned.norm is None:
            raise ValueError('constant-source classical LCHS needs representable application norms')
        duration = elapsed-planned.start
        specifications.append((duration,planned.weight*psd_recovery_part(shift,duration,exponent),
            data.source if planned.source else data.initial))
        applications.append(_application(plan.method,plan.problem,plan.reconstruction,index,planned))
    solution = spectral_lchs_sum(quad.l_part,quad.h_part,quad.k_nodes,quad.coefficients,tuple(specifications))
    return solution,_carried_recovery(None,shift,elapsed),tuple(applications)


def _evolve_pauli_actions(plan,data):
    """Stream each selected node action and preserve its source-time recovery.

    Each application's rotations are formed from the stored node table at
    its elapsed time, one position at a time (host_pf.application_actions).
    """
    from .host_pf import application_actions, apply_node
    elapsed = plan.problem.elapsed_time
    shift = plan.reconstruction.psd_shift
    exponent = psd_recovery_exponent(shift,elapsed)
    planned_applications = _planned_applications(plan.method,plan.problem,plan.reconstruction,data)
    solution = np.zeros(len(data.initial),dtype=complex)
    applications = []
    table = data.host_actions['nodes']
    for index,(action,planned) in enumerate(zip(data.host_actions['applications'],planned_applications,strict=True)):
        if data.source is None:
            vector = data.initial_direction
        else:
            vector = data.source if planned.source else data.initial
            if planned.norm is None:
                raise ValueError('constant-source classical LCHS needs representable application norms')
        branch = np.zeros_like(solution)
        for coefficient,(_,sequence,phase,steps,route) in zip(data.quadrature.coefficients,
                application_actions(table,action,elapsed=elapsed,order=data.method.trotter_order),strict=True):
            branch += coefficient*apply_node(sequence,phase,steps,route,vector)
        branch *= psd_recovery_part(shift,elapsed-action['start'],exponent)
        solution += action['weight']*branch
        applications.append(_application(plan.method,plan.problem,plan.reconstruction,index,planned))
    recovery = _carried_recovery(plan.reconstruction.initial_scale if data.source is None else None,shift,elapsed)
    return solution,recovery,tuple(applications)


def _execute(plan,kernel,data):
    """Run the declared host kernel and return only the requested outputs.

    The padded vector is cut to the original coordinates before any statistic
    is formed, so dummy coordinates never enter a norm or observable.
    """
    solution,recovery,applications = _evolve(plan,data)
    solution = solution[:plan.problem.dimension]
    observable = plan.output.observable if isinstance(plan.output,(NormalizedExpectation,QuadraticForm)) else None
    scale,direction,statistics = physical_vector_statistics(solution,observable=observable,
        numerator_frame=kernel.scalar_frames[-1] if observable is not None else 'physical',
        need_direction=any(output.frame=='unit' for output in kernel.outputs),recovery=recovery)
    arrays,unavailable = [],[]
    for output in kernel.outputs:
        array = direction if output.frame=='unit' else recovery.apply_vector(solution) if recovery is not None else solution
        if array is None:
            unavailable.append(UnavailableOutput(output=output,reason='requested vector is undefined or unrepresentable in its selected normalization'))
        else:
            arrays.append((output.name,np.asarray(array,dtype=complex)))
    return KernelOutput(plan_id=plan.content_id,selected_kernel_id=kernel.content_id,scalars=statistics,
        physical_scale=scale,arrays=tuple(arrays),applications=applications,unavailable=tuple(unavailable))


def analyze_classical(plan,data,chunks):
    """Interpret one host acquisition and sum its per-application component facts.

    Each listed component is the sum of the acquired applications' own
    facts (_summed_components, the rule planning used), and is unknown unless
    every application supplied a concrete value. The Duhamel component does
    not depend on the acquisition, so the planned fact is kept: exact zero
    for a zero source and otherwise unknown until explicit refinement.
    """
    chunk, = chunks.values()
    values = {entry.label:entry for entry in chunk.values}
    norm = values['norm_squared'].value
    numerator = values.get('numerator')
    scalar = isinstance(plan.output,(NormSquared,NormalizedExpectation,QuadraticForm))
    value,unavailable = physical_scalar(plan.output.kind,norm_squared=norm,scale=chunk.physical_scale,
        numerator=None if numerator is None else numerator.value,numerator_frame='physical' if numerator is None else numerator.frame) if scalar else (None,None)
    if not scalar and chunk.unavailable:
        # A requested vector that is unrepresentable in its frame, as in analyze_quantum.
        unavailable = chunk.unavailable[0].reason
    listed = tuple(term for term in plan.error_model.terms if term.role=='listed_only')
    summed = {}
    if listed:
        summed = {item.fact.quantity:item for item in _summed_components(plan.method,plan.problem,
            [application.facts for application in chunk.applications])}
    components = []
    for term in listed:
        if term.name!='duhamel_quadrature' and term.name not in summed:
            raise ValueError(f'classical LCHS Plan lists component {term.name!r} that its applications do not report')
        components.append(term.fact if term.name=='duhamel_quadrature' else summed[term.name])
    from .solution_error_budget import propagate_physical_error
    bound = propagate_physical_error(plan.output,(item.fact.value.value if isinstance(item.fact.value,Float64) else None for item in components),
        radius=chunk.physical_scale.as_float(),observable_norm=None)
    facts = tuple(components)+(fact('algorithmic_approximation',plan.error_model.frame,bound),
        *exact_readout_sampling(plan,data.observations,random_draws=False))
    return LCHSAnalysis(**result_fields(plan,data),value=value,unavailable=unavailable,norm_squared=norm,
        numerator=None if numerator is None else numerator.value,numerator_frame=None if numerator is None else numerator.frame,
        physical_scale=chunk.physical_scale,artifact=chunk.artifacts[0] if chunk.artifacts else None,
        applications=chunk.applications,facts=facts)
