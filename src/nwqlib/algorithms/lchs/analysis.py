"""Turn LCHS readouts (amplitudes, the exact projected reduction or counts) into the requested output in the original physical coordinates."""

from math import fsum, isfinite, sqrt

from nwqlib._quantum_readout import physical_moment, validate_readout_layout
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.records import Float64
from nwqlib.evidence.error_model import ErrorModel, ErrorTerm, FramedFact, exact_readout_sampling
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.problems.inputs import PhysicalScale, compose_recovery
from nwqlib.problems.records import Solution, StateVector, Samples, NormSquared, NormalizedExpectation, QuadraticForm
from .primary_records import METHOD, LCHSAnalysis
from .quantum import observed_bits


def fact(name,frame,value=None,*,reason="selected error component is unavailable",source=METHOD):
    """Return one error fact in frame, concrete for a value or unknown with reason.

    A concrete value is a numerical evaluation of an ideal bound with zero
    failure probability. None becomes an unknown fact and never a zero.
    """
    fields = dict(quantity=name,unit=frame.unit,scope=frame.scope)
    if value is None:
        return FramedFact(frame=frame,bindings=(),fact=Fact(**fields,availability="unknown",reason=reason))
    if not isfinite(value) or value < 0:
        raise ValueError("LCHS error bounds must be finite and nonnegative")
    return FramedFact(frame=frame,bindings=(),failure_probability=0.,fact=Fact(**fields,
        availability="concrete",value=Float64(value=float(value)),evidence=Evidence(kind="numerical_estimate",source=source),
        assumptions=("ideal selected algorithm relation evaluated numerically; native rounding is separate",)))


def error_model(problem,output,construction,components=(),*,shots=None):
    """Declare the requested-output error terms and the listed physical components.

    The required terms are algorithmic approximation, native floating point
    and sampling in the output frame. Components such as kernel_approximation
    or trotter_synthesis are listed_only physical-solution facts. Analysis
    propagates them into algorithmic_approximation once and never counts
    them twice.
    """
    frame = output.frame(problem)
    required = ("algorithmic_approximation","native_floating_point","sampling")
    terms = tuple(ErrorTerm(name=name,stage="analysis",source=METHOD,formula="requested-output absolute error component",
        fact=fact(name,frame),coverage=name,failure_probability=0. if name=="sampling" and shots is None else None) for name in required)
    terms += tuple(ErrorTerm(name=value.fact.quantity,stage="planning",source=METHOD,
        formula="selected physical-solution component from its numerical owner",fact=value,
        coverage=value.fact.quantity,role="listed_only") for value in components)
    return ErrorModel(output_id=output.content_id,subject_id=problem.content_id,construction_id=construction.content_id,
        frame=frame,required_sources=required,terms=terms,source=METHOD)


def result_fields(plan,data):
    """Return the provenance fields shared by every LCHSAnalysis of this Plan and data."""
    return dict(origin=capture_analysis_origin(analyzer=METHOD,method_id=plan.method.content_id,dependencies=("numpy",)),
        plan_id=plan.content_id,construction_id=plan.construction.content_id,observation_id=data.observations.content_id,
        contribution_ids=tuple(chunk.content_id for chunk in data.observations.chunks))


def _chunks(plan,data):
    """Map each selected experiment to its one completed acquisition.

    Every chunk must match its Plan, Run, realization, setting, observation,
    bindings and a completed trace event, so analysis never combines data
    from another selection or reuses one acquisition for two settings. The
    exact reduction's one point chunk joins its producing receipt's
    trajectory declaration, and its completed event names that one-point
    acquisition.
    """
    from nwqlib.execution import ObservationView
    if data.trace.plan_id!=plan.content_id or len(data.observations.chunks)!=len(plan.experiments):
        raise ValueError("LCHS needs its exact Plan and one completed acquisition per experiment")
    chunks = {}
    completed = {(event.attempt,event.prepared_id,event.observation_id) for event in data.trace.events if event.status=='completed'}
    receipts = {receipt.content_id:receipt for receipt in data.receipts}
    for experiment,chunk in zip(plan.experiments,data.observations.chunks,strict=True):
        if experiment.name in chunks:
            raise ValueError("LCHS settings cannot reuse one acquisition")
        realization = plan.resolve(experiment.name)
        setting,observation = realization.resolved_observation(plan)
        data.trace.validate_observation(chunk)
        if observation.kind=='trajectory':
            receipt = receipts.get(chunk.prepared_id)
            same = (receipt is not None and receipt.observation==observation and chunk.declares_readout_of(receipt)
                and (chunk.attempt,chunk.prepared_id,ObservationView(chunks=(chunk,)).content_id) in completed)
        else:
            same = chunk.observation==observation and (chunk.attempt,chunk.prepared_id,chunk.content_id) in completed
        if (not same or chunk.plan_id!=plan.content_id or chunk.run_id!=data.trace.run_id
                or chunk.realization_id!=realization.content_id or chunk.experiment!=experiment.name
                or chunk.setting!=setting or chunk.bindings!=realization.bindings):
            raise ValueError("LCHS contribution differs from its completed selected acquisition")
        chunks[experiment.name] = chunk
    return chunks


def analyze(plan,data):
    """Dispatch to the no-evolution, classical host or quantum readout analysis."""
    if plan.reconstruction.mode in {'initial','zero'}:
        from .host import analyze_initial
        return analyze_initial(plan,data)
    chunks = _chunks(plan,data)
    if plan.execution=='classical':
        from .host import analyze_classical
        return analyze_classical(plan,data,chunks)
    return analyze_quantum(plan,data,chunks)


def projected_moments(data,chunk):
    """Saved statistics of the exact reduction, validated against its producing receipt.

    The native width is that of the bound declaration (its coordinates and
    valued selectors partition the native wires), so the populations come
    from the declaration and not from the payload.

    The projected kernel bounds the masses it forms from saved amplitudes with
    its scaled two-pass allowance. It resolves only the receipt's
    amplitude-derived masses label. Mass validation uses the resulting
    qualified saved-state budget (``PreparedArtifact.saved_state_error``,
    host phase product included), or the producing receipt's propagated
    window (``saved_state_probability_window``) when another premise remains
    unavailable. Acquisition and publication use the same saved_state_error
    and saved_state_probability_window.
    """
    import json
    from nwqlib._quantum_readout import (PROJECTED_KERNEL, PROJECTED_MASS_EXCLUSIONS, PROJECTED_MOMENTS,
                                         projected_populations, projected_statistics)
    from .primary_records import LCHSProjectedMoments
    receipt = {receipt.content_id:receipt for receipt in data.receipts}.get(chunk.prepared_id)
    if receipt is None or not chunk.declares_readout_of(receipt):
        raise ValueError("LCHS exact reduction needs its producing preparation receipt")
    point = receipt.observation.point(chunk.point)
    if point.kind != "reduction" or point.reducer != PROJECTED_MOMENTS:
        raise ValueError("projected statistics require the projected_moments reducer")
    parameters = json.loads(point.parameters)
    width = len(parameters["coordinates"])+len(parameters["success"])+len(parameters["conditions"])
    stats = projected_statistics(
        chunk, parameters, width,
        delta=receipt.saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0],
        window=receipt.saved_state_probability_window,
    )
    return LCHSProjectedMoments(kernel=PROJECTED_KERNEL,**stats,populations=projected_populations(parameters,width),
        contribution_id=chunk.content_id)


def _sampled(rec,output,chunks):
    """Weighted group moments of the counts settings and the padded physical mass.

    A normalized unpadded output averages each group's weighted parities
    over its success-selected shots. Otherwise every returned shot is kept
    with its success indicator, and the identity coefficient is assigned
    once to the first group. A padded normalized output divides by the
    separately acquired physical-prefix mass.
    """
    import numpy as np
    from nwqlib._quantum_readout import weighted_group_moments
    from nwqlib.operators._pauli import _pauli_masks
    from .primary_records import LCHSGroupMoments
    ns = len(rec.success_bits)
    padded = rec.dimension!=rec.encoded_dimension
    conditional = isinstance(output,NormalizedExpectation) and not padded
    identity = fsum(c for label,c in rec.terms if all(axis=='I' for axis in label))
    coefficients = {}
    for label,c in rec.terms:
        if c!=0 and any(axis!='I' for axis in label):
            coefficients[label] = coefficients.get(label,0.)+c
    records = []
    for index,setting in enumerate(rec.settings):
        chunk = chunks[setting.name]
        histogram = chunk.histogram()
        if histogram.entries and histogram.width!=ns+len(rec.system_bits):
            raise ValueError("LCHS counts differ from their observed success and system bits")
        if histogram.width>64:
            raise ValueError("LCHS sampled readout decodes at most 64 observed bits")
        bits = histogram.indices() if histogram.entries else np.zeros(0,dtype=np.uint64)
        counts = histogram.weights
        success = (bits & np.uint64((1<<ns)-1))==0
        returned = chunk.returned_shots
        if setting.physical_projection:
            physical = success & ((bits>>np.uint64(ns))<np.uint64(rec.dimension))
            selected = sum(map(int,counts[physical]))
            records.append(LCHSGroupMoments(name=setting.name,basis=setting.label,population='physical_prefix',
                returned_shots=returned,selected_shots=selected,mean=selected/returned if returned else None,
                contribution_id=chunk.content_id))
            continue
        labels = rec.groups[index]
        masks = [_pauli_masks(label)[0]|_pauli_masks(label)[1] for label in labels]
        weights = [coefficients[label] for label in labels]
        selected = sum(map(int,counts[success]))
        if not conditional and index==0 and identity!=0:
            masks,weights = masks+[0],weights+[identity]
        masks = [mask<<ns for mask in masks]
        mean = second = variance = None
        if conditional and selected:
            mean,second,variance = weighted_group_moments(bits[success],counts[success],1.,masks,weights)
        elif not conditional and returned:
            mean,second,variance = weighted_group_moments(bits,counts,success.astype(float),masks,weights)
        records.append(LCHSGroupMoments(name=setting.name,labels=labels,basis=setting.label,
            population='success_conditional' if conditional else 'unconditional',returned_shots=returned,
            selected_shots=selected,mean=mean,second_moment=second,variance=variance,contribution_id=chunk.content_id))
    return tuple(records),identity,conditional


def _vector_publication(plan,chunks):
    """Return a quantum Solution's or StateVector's ``(artifact, unavailable, physical_scale, norm_squared)``.

    ``chunks`` maps each selected experiment to its acquisition (``_chunks``).
    The Result publishes the one amplitude acquisition's artifact, if any, whose
    array declaration must be the selected one, the acquisition's first
    unavailability reason, if any, its physical scale, and the square of that
    scale as its norm squared. ``analyze_quantum`` calls this.
    """
    first = chunks[plan.experiments[0].name]
    if len(chunks)!=1 or len(first.artifacts)>1:
        raise ValueError("LCHS vector output needs its one selected amplitude acquisition")
    artifact = first.artifacts[0] if first.artifacts else None
    if artifact is not None and artifact.output!=first.observation.amplitudes.output:
        raise ValueError("LCHS amplitude array differs from its selected physical declaration")
    scale = first.physical_scale
    return (artifact,first.unavailable[0].reason if first.unavailable else None,scale,
            None if scale is None else scale.squared_as_float())


def analyze_quantum(plan,data,chunks):
    """Reduce the acquired LCHS vector, exact reduction or counts in original coordinates."""
    from nwqlib._quantum_readout import (pair_float, pair_ratio, recover_scaled_pair, reduce_sample_arrays,
                                         reduce_setting)
    from nwqlib.core.records import FrozenArray
    from .primary_records import LCHSSamples
    rec,output = plan.reconstruction,plan.output
    first = chunks[plan.experiments[0].name]
    success = tuple((bit,0) for bit in rec.success_bits)
    value = norm = numerator = scale = artifact = reduction = None
    unavailable = None
    samples,groups = None,()
    submitted = returned = selected = None
    if isinstance(output,(Solution,StateVector)):
        artifact,unavailable,scale,norm = _vector_publication(plan,chunks)
        if artifact is not None:
            data.artifact(artifact)
    elif plan.shots is None:
        if len(chunks)!=1 or first.point is None:
            raise ValueError("LCHS exact scalar output needs its one selected reduction")
        reduction = projected_moments(data,first)
        p,q = reduction.physical_mass,reduction.numerator
        recovered = recover_scaled_pair(p,rec.recovery)
        norm = None if recovered is None else pair_float(recovered)
        if p[0]!=0:
            # p represents positive mass m*2**e with 1/2 <= m < 1.
            # For odd e, dividing sqrt(2*m) by two keeps the mantissa in [1/2, 1).
            m,e = p
            root = sqrt(m*2) if e%2 else sqrt(m)
            scale = compose_recovery(PhysicalScale(mantissa=root/2 if e%2 else root,exponent=(e+1)//2 if e%2 else e//2),
                rec.recovery)
        else:
            scale = PhysicalScale(mantissa=0.,exponent=0)
        if isinstance(output,NormSquared):
            value = norm
            if value is None:
                unavailable = 'physical norm squared is unavailable or unrepresentable'
        elif isinstance(output,NormalizedExpectation):
            if p[0]==0:
                unavailable = 'normalized quantity is undefined for zero physical mass'
            else:
                value = pair_ratio(q,p)
                unavailable = None if value is not None else 'normalized expectation is not representable'
            numerator = value
        else:
            recovered = recover_scaled_pair(q,rec.recovery)
            value = None if recovered is None else pair_float(recovered)
            unavailable = None if value is not None else 'physical quadratic form is not representable'
            numerator = value
    else:
        for setting in rec.settings:
            validate_readout_layout(chunks[setting.name],observed=observed_bits(rec),classical=plan.construction.program.classical)
        if isinstance(output,Samples):
            submitted,returned = first.observation.shots,first.returned_shots
            indices,counts,_,selected = reduce_sample_arrays(first,observed=observed_bits(rec),
                coordinates=rec.system_bits,success=success,dimension=rec.dimension)
            # The reducer returns fresh, C-contiguous int64 arrays that nothing else
            # references, so the record adopts them without copying.
            samples = LCHSSamples(indices=FrozenArray._from_owned_canonical_array(indices),
                                  counts=FrozenArray._from_owned_canonical_array(counts))
            del indices,counts
            if selected==0:
                unavailable = 'no observed original-coordinate algorithm-success shots'
        elif isinstance(output,NormSquared):
            setting = rec.settings[0]
            _,mass,_ = reduce_setting(first,observed=observed_bits(rec),success=success,
                coordinates=rec.system_bits if setting.physical_projection else (),
                dimension=rec.dimension if setting.physical_projection else None)
            norm = value = None if mass is None else physical_moment(mass,rec.recovery)
            if value is None:
                unavailable = 'physical norm squared is unavailable or unrepresentable'
        else:
            groups,identity,conditional = _sampled(rec,output,chunks)
            parts = [item for item in groups if item.population!='physical_prefix']
            mass = next((item for item in groups if item.population=='physical_prefix'),None)
            # The physical mass is the unrotated setting's physical fraction,
            # or without padding the first group's success fraction.
            if mass is not None and mass.mean is not None:
                norm = physical_moment(mass.mean,rec.recovery)
            elif rec.dimension==rec.encoded_dimension and parts[0].returned_shots:
                norm = physical_moment(parts[0].selected_shots/parts[0].returned_shots,rec.recovery)
            if any(item.mean is None for item in parts):
                unavailable = 'no observed physical-selection shots'
            elif conditional:
                value = fsum([*(item.mean for item in parts),identity])
            else:
                total = fsum(item.mean for item in parts)
                if isinstance(output,NormalizedExpectation):
                    if mass is None or not mass.mean:
                        unavailable = 'no observed physical-selection shots'
                    else:
                        value = total/mass.mean
                else:
                    value = physical_moment(total,rec.recovery)
                    if value is None:
                        unavailable = 'physical quadratic form is not representable'
            numerator = value
    frame = plan.error_model.frame
    # Propagate only available component evidence into the requested frame.
    # An unavailable contribution does not become a zero total error.
    component_values = tuple(term.fact.fact.value.value if isinstance(term.fact.fact.value,Float64) else None
        for term in plan.error_model.terms if term.role=='listed_only')
    from .solution_error_budget import propagate_physical_error
    propagated = propagate_physical_error(output,component_values,radius=None if scale is None else scale.as_float(),
        observable_norm=fsum(abs(c) for _,c in rec.terms) if rec.terms else None)
    # Exact readout contributes a proved zero sampling error. Counts leave it unknown.
    facts = (fact('algorithmic_approximation',frame,propagated),
             *(exact_readout_sampling(plan,data.observations,random_draws=False) or (fact('sampling',frame),)))
    return LCHSAnalysis(**result_fields(plan,data),value=value,unavailable=unavailable,norm_squared=norm,physical_scale=scale,
        numerator=numerator,numerator_frame='unit' if isinstance(output,NormalizedExpectation) else 'physical' if isinstance(output,QuadraticForm) else None,
        artifact=artifact,samples=samples,reduction=reduction,groups=groups,submitted_shots=submitted,
        returned_shots=returned,selected_shots=selected,facts=facts)
