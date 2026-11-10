"""Turn LCHS readouts (amplitudes, the exact projected reduction or counts) into the requested output in the original physical coordinates."""

from math import fsum, isfinite, log, sqrt

from nwqlib._quantum_readout import physical_moment, validate_readout_layout
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.records import Float64
from nwqlib.evidence.error_model import ErrorModel, ErrorTerm, FramedFact, exact_readout_sampling
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.problems.inputs import PhysicalScale, compose_recovery
from nwqlib.problems.records import Solution, StateVector, Samples, NormSquared, NormalizedExpectation, QuadraticForm
from .primary_records import ASSERTED_L_BOUNDS, METHOD, LCHSAnalysis
from .quantum import observed_bits


# The ideal relations of the concrete components use exact nodes, weights and
# coefficients. Building the stored binary64 coefficient table and executing
# the construction are outside them and fall under native_floating_point.
COMPONENT_ASSUMPTIONS = (
    "ideal selected algorithm relation with exact nodes, weights and coefficients, evaluated numerically",
    "binary64 construction of the coefficient table and execution rounding are under native_floating_point",
)
NATIVE_FLOATING_POINT_REASON = (
    "binary64 construction of the coefficient table (quadrature nodes and weights, kernel values, "
    "Duhamel nodes and weights) and floating-point execution are not bounded")


def component_assumptions(premise):
    """Return the open assumptions of a component bound of a Plan whose PSD premise has the evidence ``premise``.

    A premise the caller asserted stays an open assumption of every bound that
    uses the shift. It never becomes a verified property.
    """
    return COMPONENT_ASSUMPTIONS + ((ASSERTED_L_BOUNDS,) if premise == "user_assertion" else ())


def fact(name,frame,value=None,*,reason=None,source=METHOD,premise=None):
    """Return one error fact in frame, concrete for a value or unknown with reason.

    A concrete value is a numerical evaluation of an ideal bound, with exact
    nodes, weights and coefficients, and zero failure probability, under the
    assumptions of `component_assumptions` for the PSD premise ``premise``. None
    becomes an unknown fact and never a zero. Without a reason, an unknown
    native_floating_point gives NATIVE_FLOATING_POINT_REASON and any other
    component "selected error component is unavailable".
    """
    fields = dict(quantity=name,unit=frame.unit,scope=frame.scope)
    if value is None:
        if reason is None:
            reason = (NATIVE_FLOATING_POINT_REASON if name=="native_floating_point"
                      else "selected error component is unavailable")
        return FramedFact(frame=frame,bindings=(),fact=Fact(**fields,availability="unknown",reason=reason))
    if not isfinite(value) or value < 0:
        raise ValueError("LCHS error bounds must be finite and nonnegative")
    return FramedFact(frame=frame,bindings=(),failure_probability=0.,fact=Fact(**fields,
        availability="concrete",value=Float64(value=float(value)),evidence=Evidence(kind="numerical_estimate",source=source),
        assumptions=component_assumptions(premise)))


def adds_to_total(name):
    """Return whether a listed physical component is summed into algorithmic_approximation.

    trotter_pruning is excluded because it is part of trotter_synthesis: a
    fixed-step certificate bounds both together, and without one the total is
    unknown anyway (solution_error_budget.pruning_stage).
    """
    return name != "trotter_pruning"


def error_model(problem,output,construction,components=(),*,shots=None):
    """Declare the requested-output error terms and the listed physical components.

    The required terms are algorithmic_approximation, native_floating_point
    and sampling, in the output frame. native_floating_point covers the
    binary64 construction of the coefficient table and the floating-point
    execution, and is unknown with NATIVE_FLOATING_POINT_REASON. Components
    such as kernel_approximation or trotter_synthesis are listed_only
    physical-solution facts. Analysis propagates them into
    algorithmic_approximation once and never counts them twice
    (adds_to_total).
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


def _chunks(plan,data,*,partial=False):
    """Map each planned point to its one completed data collection, keyed by the point's key.

    The points are the Plan's experiments, and for the sampled-pair batch each
    ``row`` axis point (`Plan._static_points`), whose key is
    ``pairs[row=i]``. Every observation chunk must match its Plan, Run, built circuit, setting,
    observation, bindings and a completed trace event, so analysis never
    combines data from another Plan or reuses one data collection for two
    points. The exact reduction's one-point chunk joins the trajectory
    declaration of the record that produced it, and its completed event names
    that one-point data collection. An experiment's setting and observation
    are resolved once, since its axis points share them, and each point's
    Realization is formed from its values (`Plan._point_realization`)
    without building its Program again. The points are counted from the
    axis bounds (`Plan._point_count`) and generated one at a time. With
    ``partial`` the data may lack points: each chunk is matched to the point
    its experiment and axis bindings name (`Plan._static_point`), so only
    the points of the data are formed, and checked the same way, and a
    second chunk of one point or a chunk of an unplanned point is still
    refused.
    """
    from nwqlib.execution import ObservationView
    total = plan._point_count()
    count = len(data.observations.chunks)
    if data.trace.plan_id!=plan.content_id or (count>total if partial else count!=total):
        raise ValueError("LCHS needs its exact Plan and one completed acquisition per experiment")
    chunks,readouts = {},{}
    completed = {(event.attempt,event.prepared_id,event.observation_id) for event in data.trace.events if event.status=='completed'}
    receipts = {receipt.content_id:receipt for receipt in data.receipts}
    pairs = (((plan._static_point(chunk.experiment,{item.parameter:item.value for item in chunk.bindings}),chunk)
              for chunk in data.observations.chunks) if partial
             else zip(plan._static_points(),data.observations.chunks,strict=True))
    for point,chunk in pairs:
        if point is None:
            raise ValueError("LCHS contribution differs from its completed selected acquisition")
        if point.key in chunks:
            raise ValueError("LCHS settings cannot reuse one acquisition")
        name = point.experiment.name
        if name not in readouts:
            readouts[name] = plan.resolve(name,bindings=point.bindings).resolved_observation(plan)
        setting,observation = readouts[name]
        realization = plan._point_realization(name,point.bindings)
        data.trace.validate_observation(chunk)
        if observation.kind=='trajectory':
            receipt = receipts.get(chunk.prepared_id)
            same = (receipt is not None and receipt.observation==observation and chunk.declares_readout_of(receipt)
                and (chunk.attempt,chunk.prepared_id,ObservationView(chunks=(chunk,)).content_id) in completed)
        else:
            same = chunk.observation==observation and (chunk.attempt,chunk.prepared_id,chunk.content_id) in completed
        if (not same or chunk.plan_id!=plan.content_id or chunk.run_id!=data.trace.run_id
                or chunk.realization_id!=realization.content_id or chunk.experiment!=name
                or chunk.setting!=setting or chunk.bindings!=realization.bindings):
            raise ValueError("LCHS contribution differs from its completed selected acquisition")
        chunks[point.key] = chunk
    return chunks


def analyze(plan,data):
    """Dispatch to the no-evolution, classical or quantum readout analysis."""
    if plan.reconstruction.mode in {'initial','zero'}:
        from .host import analyze_initial
        return analyze_initial(plan,data)
    chunks = _chunks(plan,data)
    if plan.execution=='classical':
        from .host import analyze_classical
        return analyze_classical(plan,data,chunks)
    return analyze_quantum(plan,data,chunks)


def projected_moments(data,chunk):
    """Saved statistics of the exact reduction, validated against the record that produced it.

    The circuit width is that of the attached declaration (its coordinates and
    valued selectors partition the circuit wires), so the amplitude counts
    (``populations``) come from the declaration and not from the saved data.

    The projected reduction routine bounds the masses it forms from saved amplitudes with
    its scaled two-pass allowance. It resolves only the receipt's
    amplitude-derived masses label. Mass validation uses the resulting
    qualified saved-state budget (``PreparedArtifact.saved_state_error``,
    classical phase product included), or the propagated window of the
    record that produced it (``saved_state_probability_window``) when another
    assumption remains unavailable. Data collection and the reported result
    use the same saved_state_error and saved_state_probability_window.
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


def _coalesced(terms):
    """Return the nonidentity Pauli coefficients summed per label and the summed identity coefficient.

    These are the weights of the counts scores (`_sampled`), so the radius
    (`_counts_radius`) and the sufficient count (`score_ranges`) read the
    same values.
    """
    identity = fsum(c for label,c in terms if all(axis=='I' for axis in label))
    coefficients = {}
    for label,c in terms:
        if c!=0 and any(axis!='I' for axis in label):
            coefficients[label] = coefficients.get(label,0.)+c
    return coefficients,identity


def _sampled(rec,output,chunks):
    """Weighted group moments of the counts settings and the padded physical mass.

    A normalized unpadded output averages each group's weighted parities
    over its shots postselected on success. Otherwise every returned shot is
    kept with its success indicator, and the identity coefficient is assigned
    once to the first group. A padded normalized output divides by the
    separately measured physical-prefix mass.
    """
    import numpy as np
    from nwqlib._quantum_readout import weighted_group_moments
    from nwqlib.operators._pauli import _pauli_masks
    from .primary_records import LCHSGroupMoments
    ns = len(rec.success_bits)
    padded = rec.dimension!=rec.encoded_dimension
    conditional = isinstance(output,NormalizedExpectation) and not padded
    coefficients,identity = _coalesced(rec.terms)
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
            selected = int(counts[physical].sum())
            records.append(LCHSGroupMoments(name=setting.name,basis=setting.label,population='physical_prefix',
                returned_shots=returned,selected_shots=selected,mean=selected/returned if returned else None,
                contribution_id=chunk.content_id))
            continue
        labels = rec.groups[index]
        masks = [_pauli_masks(label)[0]|_pauli_masks(label)[1] for label in labels]
        weights = [coefficients[label] for label in labels]
        selected = int(counts[success].sum())
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

    ``chunks`` maps each planned experiment to its data collection (``_chunks``).
    The Result reports the saved array of the one amplitude data collection,
    if any, whose array declaration must be the planned one, that data
    collection's first unavailability reason, if any, its physical scale, and the square of that
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


_PAIR_SAMPLING = ("Hoeffding bound of the equal-weight mean of independent bounded row means, for the complete "
                  "predeclared row table")


def _sampling_fact(frame,value,failure,*,assumption=_PAIR_SAMPLING,
                   reason="the sampled-pair radius is unavailable for this output"):
    """Return the Hoeffding sampling radius as a fact with its failure probability, or an unknown fact."""
    fields = dict(quantity="sampling",unit=frame.unit,scope=frame.scope)
    if value is None:
        return FramedFact(frame=frame,bindings=(),fact=Fact(**fields,availability="unknown",reason=reason))
    return FramedFact(frame=frame,bindings=(),failure_probability=failure,fact=Fact(**fields,availability="concrete",
        value=Float64(value=float(value)),evidence=Evidence(kind="numerical_estimate",source=METHOD),
        assumptions=(assumption,
                     "binary64 evaluation; no enclosure of the rounded scores, their reduction or the radius")))


_COUNTS_SAMPLING = ("Hoeffding bound for each counts setting's independent shots, summed over settings with the "
                    "failure probability split equally (M7.13-M7.14); a statistical radius of the finite circuit's "
                    "moment, not its bias against the physical solution")


def _counts_radius(rec,output,groups,value,failure,*,returned=None):
    """Return ``(radius, reason)``: the shot radius of a counts estimate in the output frame, or None.

    Counts estimates include a no-evolution readout selected by either estimator.

    Each nonidentity Pauli term of a group setting is measured jointly with
    the others of that group, so the per-shot score of group g is
    ``Y_g = Gamma**2 1_good sum_(l in g) o_l s_l`` with ``|Y_g| <= Gamma**2
    A_g``, ``A_g = sum_(l in g) |o_l|``, the identity coefficient counted once
    in the first unconditional group (M7.13 of the M7 mathematics handoff;
    ``_sampled``). With ``N_g`` independent returned shots and failure share
    ``delta_g`` Hoeffding's Theorem 2 (doi:10.1080/01621459.1963.10500830)
    bounds the group mean within ``Gamma**2 A_g sqrt(2 log(2/delta_g)/N_g)``,
    and a union bound over the settings gives M7.14,
    ``R = sum_g Gamma**2 A_g sqrt(2 log(2/delta_g)/N_g)``,
    ``delta_g = delta/G``. NormSquared uses the indicator in ``[0, Gamma**2]``,
    ``Gamma**2 sqrt(log(2/delta)/(2 N))``. An unpadded normalized output uses
    the success-conditional means over the ``S_g`` selected shots, given
    which those shots are independent, so ``Gamma`` cancels and
    ``R = sum_g A_g sqrt(2 log(2/delta_g)/S_g)``. A padded normalized output
    is the ratio of the unconditional group sum and the physical-prefix
    mass, each covered with its share; the ratio interval exists only when
    the mass interval is positive (``sampled_moments.ratio_interval``), and
    its radius is the larger distance from the estimate to an endpoint.
    """
    from .sampled_moments import hoeffding_radius, ratio_interval
    if isinstance(output,NormSquared):
        if not returned:
            return None,"no returned shots"
        share = sqrt(log(2./failure)/(2.*returned))
        return physical_moment(share,rec.recovery),None
    coefficients,identity = _coalesced(rec.terms)
    identity = abs(identity)
    parts = [item for item in groups if item.population!='physical_prefix']
    mass = next((item for item in groups if item.population=='physical_prefix'),None)
    conditional = bool(parts) and parts[0].population=='success_conditional'
    share = failure/(len(parts)+int(mass is not None))
    radius = []
    for index,item in enumerate(parts):
        shots = item.selected_shots if conditional else item.returned_shots
        bound = fsum(abs(coefficients[label]) for label in item.labels)+(identity if index==0 and not conditional else 0.)
        if not shots:
            return None,"a counts setting has no shots for its population"
        radius.append(hoeffding_radius(bound,share,shots) if bound else 0.)
    total = fsum(radius)
    if conditional:
        return total,None
    if isinstance(output,NormalizedExpectation):
        if mass is None or not mass.returned_shots or value is None:
            return None,"the normalized ratio needs its physical-mass setting"
        numerator = fsum(item.mean for item in parts)
        interval = ratio_interval(numerator,total,mass.mean,sqrt(log(2./share)/(2.*mass.returned_shots)))
        if interval is None:
            return None,"the physical-mass interval is not positive, so the normalized ratio is unresolved"
        return max(value-interval[0],interval[1]-value),None
    return physical_moment(total,rec.recovery),None


def score_ranges(terms,groups):
    """Return ``A_g = sum_(l in g) |o_l|`` of each counts setting, with ``|c_I|`` added to the first, rounded up.

    ``terms`` and ``groups`` are the reconstruction's Pauli terms and the
    member labels of each qubit-wise commuting group setting
    (``quantum.lchs_readout``); an observable without a nonidentity term has
    the one empty group. With the coalesced weights of the counts scores
    (`_coalesced`), the unconditional score of group g is
    ``Y_g = Gamma**2 1_good sum_(l in g) o_l s_l``, plus the identity
    coefficient once in the first group, so ``|Y_g| <= Gamma**2 A_g``
    (M7.13 of the M7 mathematics handoff, the range `_counts_radius` uses).
    Each ``A_g`` is the exact rational sum of those binary64 weights,
    converted to binary64 and raised by one step when the conversion rounded
    down, so it bounds the actual range.
    """
    from fractions import Fraction
    from math import inf, nextafter
    coefficients,identity = _coalesced(terms)
    ranges = []
    for index,labels in enumerate(groups):
        exact = sum((abs(Fraction(coefficients[label])) for label in labels),Fraction())
        if index==0:
            exact += abs(Fraction(identity))
        value = float(exact)
        ranges.append(value if Fraction(value)>=exact else nextafter(value,inf))
    return tuple(ranges)


def sufficient_counts(recovery,*,kind,ranges,tolerance,confidence):
    r"""Return the shots per counts setting that make the coherent sampling radius at most ``tolerance``, or None when a saved Plan cannot hold that count.

    The inversions of the radius laws of `_counts_radius` (ACCURACY_HOOK_POLICY
    AH1/AH2, from M7.13-M7.14 and Hoeffding's Theorem 2,
    doi:10.1080/01621459.1963.10500830). With ``Gamma`` the physical
    recovery, ``delta = 1 - confidence`` and the same N on each of the G
    settings, the failure probability split equally (``delta_g = delta/G``,
    the split of `_counts_radius`), a quadratic form has

        \[R_Q(N)=\sum_g\Gamma^2A_g\sqrt{2\log(2G/\delta)/N}
          =\Gamma^2A\sqrt{2\log(2G/\delta)/N},\qquad A=\sum_gA_g,\]

    so ``R_Q(N) <= epsilon`` exactly when
    ``N >= 2 Gamma**4 A**2 log(2G/delta)/epsilon**2`` (AH1). The norm
    squared is the one indicator in ``[0, Gamma**2]`` with
    ``R_N(N) = Gamma**2 sqrt(log(2/delta)/(2N))``, so
    ``N >= Gamma**4 log(2/delta)/(2 epsilon**2)`` (AH2). The count is the
    ceiling, at least 1. These bound the statistical error of the finite
    circuit's moment, given independent stationary shots within each
    setting; they do not bound its bias against the physical solution.

    ``Gamma**4`` is formed from the PhysicalScale mantissa m and integer
    exponent e as the exact ``m**4 2**(4e)``, never as a binary64 power, and
    the inputs are exact binary64 values. Those inputs change log2 of the
    count by less than 4400, so beyond that margin around the bit length of
    ``core.records.STORABLE_INTEGER_BOUND`` the sign of the exponent alone
    gives None or 1 and the power is not formed. As in
    ``ExpectationMethod.sampling_shots``, the quotient and the logarithm are
    evaluated in 80-digit Decimal arithmetic rounded upward (the logarithm
    is correctly rounded, then raised one unit), so the integer is never
    below the analytic bound.

    Args:
        recovery (PhysicalScale): Gamma of the counts settings.
        kind (str): ``"quadratic_form"`` or ``"norm_squared"``.
        ranges (tuple[float, ...]): A_g of each setting (`score_ranges`);
            ignored for the norm squared, which has one setting.
        tolerance (float): epsilon, positive, in the output's unit.
        confidence (float): 1 - delta, strictly between 0 and 1.

    Returns:
        shots (int | None): N per setting, or None when it is not below
        ``10**4300``, the integers that a saved Plan can hold.
    """
    from decimal import Decimal, ROUND_CEILING, localcontext
    from fractions import Fraction
    from nwqlib.core.records import STORABLE_INTEGER_BOUND
    if kind not in ("quadratic_form","norm_squared"):
        raise ValueError(f"the counts law covers 'quadratic_form' and 'norm_squared', not {kind!r}")
    delta = 1-Fraction(confidence)
    epsilon = Fraction(tolerance)
    total = sum((Fraction(value) for value in ranges),Fraction()) if kind=="quadratic_form" else None
    # A zero recovery or zero score ranges give radius zero for every N.
    if not recovery.mantissa or total==0:
        return 1
    exponent = 4*recovery.exponent
    margin = STORABLE_INTEGER_BOUND.bit_length()+4400
    if exponent>margin:
        return None
    if exponent<-margin:
        return 1
    gamma4 = Fraction(recovery.mantissa)**4*(Fraction(1 << exponent) if exponent >= 0 else Fraction(1,1 << -exponent))
    if kind=="quadratic_form":
        factor = 2*gamma4*total*total/(epsilon*epsilon)
        argument = 2*len(ranges)/delta
    else:
        factor = gamma4/(2*epsilon*epsilon)
        argument = 2/delta
    with localcontext() as context:
        # 80 significant digits, as in ExpectationMethod.sampling_shots.
        context.prec = 80
        context.rounding = ROUND_CEILING
        required = (Decimal(factor.numerator)/Decimal(factor.denominator)
                    *(Decimal(argument.numerator)/Decimal(argument.denominator)).ln().next_plus())
        if required>Decimal(STORABLE_INTEGER_BOUND-1):
            return None
        return max(1,int(required.to_integral_value(rounding=ROUND_CEILING)))


def _pair_rows(plan,chunks):
    """Return the moments, indices and score sums of the rows whose chunks are present with all m shots, and the returned shots.

    Each point ``pairs[row=i]`` of the Plan's one experiment is row i. The
    present data collections, each matched to its planned point by
    `_chunks`, are reduced in increasing order of their ``row`` binding,
    without visiting the rows that have none. A row's counts give the signed scores of
    `sampled_moments.row_scores` with the branch bit 0 and system qubit j at
    bit 1 + j. The estimator's sample unit is a complete row of m shots. A row
    with fewer returned shots is incomplete and adds nothing to the
    completed-row Hoeffding population, while its returned shots and attempt
    still count as exposure. Its attempt is final, so continuing the Run does
    not measure it again. A missing value is neither zero nor a completed
    sample. Rows without a chunk are skipped, which
    only `sampled_pair_progress` passes. A product-formula recipe
    multiplies each row's magnitude by its saved factor, |gamma| of its
    member or mu_x*mu_y of its PTSC branches, and the score ranges by the
    recipe's bound on that factor; a negative member's sign is already in
    its circuit's readout phase (`sampled_recipes`). A sampled QSP row
    multiplies its magnitude by zeta_b**2 of its bucket (1 for a both-zero
    row) and the ranges by the bank bound F; its success flags follow the
    system bits, and only shots with every flag zero score (B16-B18 of the
    finite-bank relations in `sampled_qsp`, `sampled_moments.row_scores`), so the mean still
    divides by every returned shot.
    """
    from nwqlib._quantum_readout import validate_readout_layout
    from .sampled_moments import add_rows, empty_pair_moments, row_scores
    rec = plan.reconstruction
    rows,recipe,bank = rec.pair_rows,rec.pair_recipe,rec.pair_qsp
    q = rec.encoded_dimension.bit_length()-1
    flags = 0 if bank is None else bank.flag_qubits
    good = ((1<<flags)-1)<<(q+1)
    classical = plan.construction.program.classical
    magnitudes = rows.score_magnitudes()
    if recipe is not None:
        magnitudes = magnitudes*recipe.row_factors()
    if bank is not None:
        magnitudes = magnitudes*bank.row_factors()
    done,scores,ids,returned = [],[],[],0
    present = sorted((({item.parameter:item.value for item in chunk.bindings}["row"],chunk) for chunk in chunks.values()),
                     key=lambda pair:pair[0])
    for index,chunk in present:
        validate_readout_layout(chunk,observed=tuple(range(q+1+flags)),classical=classical)
        histogram = chunk.histogram()
        returned += chunk.returned_shots
        if chunk.returned_shots!=rows.repeats or not histogram.entries:
            continue
        term = int(rows.term.array[index])
        label = rec.terms[term][0] if term>=0 else "I"*q
        parity = sum(1<<(1+j) for j,axis in enumerate(reversed(label)) if axis!="I")
        scores.append(row_scores(histogram.indices(),histogram.weights,branch_mask=1,parity_mask=parity,
            magnitude=float(magnitudes[index]),signed_range=float(rows.signed_range.array[index]),good_mask=good))
        done.append(index)
        ids.append(chunk.content_id)
    factor = recipe.score_factor if recipe is not None else bank.score_factor if bank is not None else 1.
    return add_rows(empty_pair_moments(rows,factor),done,scores,ids),done,scores,returned


def sampled_pair_progress(run):
    """Return the raw sums of the measured rows of a sampled-pair LCHS Run, without any confidence statement.

    Use it on a Run of a Plan with `LCHS(estimator="sampled_pairs")`
    that stopped before every row was measured, for example one reopened with
    `load_run`. It reads the Run's saved data, checks every present row's data
    collection as the complete analysis does, and returns which rows are
    completed and missing with the signed sums of their scores. It changes no
    Run state and saves nothing. It gives no estimate, confidence radius or
    normalized value, because those need every predeclared row, and rows lost
    to failed or outcome-dependent acquisitions are not a random sample of the
    rows. Continuing the Run (`run.wait()` or `run.resume()`) measures only the
    rows without a finished attempt, and its Result is then the only publisher
    of confidence. The estimator's sample unit is a complete row of `repeats`
    shots. A row whose finished attempt returned fewer shots is incomplete and
    is listed as missing. It adds nothing to the completed-row Hoeffding
    population, and its returned shots and attempt still count as exposure. Its
    attempt is final, so continuing does not measure it again, and that Run
    publishes no value. A missing value is neither zero nor a completed sample.

    Args:
        run (Run): A `Run` of a sampled-pair LCHS Plan.

    Returns:
        progress (LCHSPairProgress): Completed and missing rows, per-row
            score sums and their joint moments, in the scaled frame of the
            row table.

    Raises:
        ValueError: If the Run's Plan is not a sampled-pair LCHS Plan, or a
            data collection differs from its planned row, repeats one or does
            not belong to the Plan.
    """
    import numpy as np
    from nwqlib.core.records import FrozenArray
    from .primary_records import LCHSPairProgress
    plan,data = run.plan,run.data
    if plan.method.descriptor.method!="lchs" or getattr(plan.reconstruction,"pair_rows",None) is None:
        raise ValueError("sampled_pair_progress needs a Run of a sampled-pair LCHS Plan")
    moments,done,scores,returned = _pair_rows(plan,_chunks(plan,data,partial=True))
    rows = plan.reconstruction.pair_rows
    missing = np.setdiff1d(np.arange(rows.clusters,dtype=np.int64),np.asarray(done,dtype=np.int64))
    return LCHSPairProgress(plan_id=plan.content_id,run_id=data.trace.run_id,clusters=rows.clusters,
        repeats=rows.repeats,completed_rows=FrozenArray(np.asarray(done,dtype=np.int64)),
        missing_rows=FrozenArray(missing),row_sums=FrozenArray(np.asarray(scores,dtype=float).reshape(-1,5)),
        moments=moments,returned_shots=returned,status="incomplete" if missing.size else "complete")


def _pair_analysis(plan,data,chunks):
    """Reduce the counts of a sampled-pair Plan to signed pair moments, raw estimates and covering radii.

    The rows reduce as in `_pair_rows`. The covering radius of each
    requested quantity is its Hoeffding radius plus the deterministic bias
    of `LCHSPairBias.norm_bias` (times the observable bound for the
    quadratic form), for the failure share
    SampledPairs.fail_prob/len(quantities). A normalized value is
    published only when N_hat - e_N > 0, so that every N in the covering
    interval is positive, with the conservative error
    ``|Q_hat/N_hat - Q/N| <= (e_Q + C_O*e_N)/N_hat``, which follows from
    ``Q/N - Q_hat/N_hat = (Q - Q_hat)/N_hat + (Q/N)*(N_hat - N)/N_hat`` and
    |Q/N| <= ||O|| <= C_O. Its sampling and bias parts are the same
    expression with only the Hoeffding radii and only the biases. An
    accuracy request owns the confidence of its Plan, whose rows it chose
    (`sampled_plan.sampled_accuracy_shots`): its 1 - confidence replaces
    SampledPairs.fail_prob.
    """
    from .primary_records import LCHSPairEstimate
    from .sampled_moments import hoeffding_radius, pair_confidence, ratio_interval
    from .sampled_moments import norm_domain_interval
    rec,output,method = plan.reconstruction,plan.output,plan.method
    rows,bias = rec.pair_rows,rec.pair_bias
    moments,_,_,returned = _pair_rows(plan,chunks)
    quantities = ("n",) if isinstance(output,NormSquared) else ("q",) if isinstance(output,QuadraticForm) else ("q","n")
    bias_n = bias.norm_bias()
    bias_q = None if bias_n is None else bias.observable_bound*bias_n
    accuracy = plan.selection_accuracy
    failure = method.estimator.fail_prob if accuracy is None else 1.0-accuracy.confidence
    confidence = pair_confidence(moments,failure=failure,quantities=quantities,
                                 bias_q=bias_q,bias_n=bias_n)
    scale = rows.scale
    def physical(value):
        return None if value is None else physical_moment(value,scale)
    q_hat,n_hat = (None,None) if moments.completed==0 else (moments.cluster_q/moments.completed,
                                                           moments.cluster_n/moments.completed)
    estimate = dict(quadratic_form=physical(q_hat) if rec.terms else None,norm_squared=physical(n_hat))
    radii,samples = {},{}
    if confidence is not None:
        for name,(_,radius,share) in confidence.items():
            bound = moments.score_bound_q if name=="q" else moments.score_bound_n
            samples[name] = hoeffding_radius(bound,share,moments.clusters)
            radii[name] = radius
            estimate.update({f"sampling_radius_{name}":physical(samples[name]),f"radius_{name}":physical(radius),
                f"bias_{name}":physical(bias_q if name=="q" else bias_n),f"failure_{name}":share})
    if "n" in radii and radii["n"] is not None:
        interval = tuple(physical(value) for value in norm_domain_interval(n_hat,radii["n"]))
        estimate["norm_interval"] = None if None in interval else interval
    if "q" in radii and "n" in radii:
        estimate["ratio_interval"] = ratio_interval(q_hat,radii["q"],n_hat,radii["n"])
    notes = [f"{moments.completed} of {rows.clusters} rows completed with {rows.repeats} shots each",
             rows.qualification]
    if rec.pair_qsp is not None:
        notes += ["inner bias by the independent-member QSP bank relation 2*beta + beta**2 with the bank's "
                  "recovered mean polynomial errors", rec.pair_qsp.qualification]
    else:
        notes.append(f"product-formula bias by the {bias.inner_relation.replace('_',' ')} relation for the rounded "
                     "branch coefficients, whose difference from the exact coefficients has no bound")
    if rec.pair_recipe is not None:
        notes.append(rec.pair_recipe.qualification)
    if confidence is None:
        notes.append("no covering radius before every planned row is completed")
    if rec.psd_premise == "user_assertion":
        notes.append(ASSERTED_L_BOUNDS)
    notes.extend(bias.reasons)
    pair_estimate = LCHSPairEstimate(**estimate,qualification="; ".join(notes))
    frame = plan.error_model.frame
    value = unavailable = numerator = None
    sampling = algorithmic = None
    if confidence is None:
        unavailable = "sampled pairs publish a value only for the complete planned row table"
    elif isinstance(output,NormalizedExpectation):
        if radii["q"] is None or radii["n"] is None:
            unavailable = ("the normalized expectation needs known deterministic biases of the quadratic form and "
                           "the norm squared")
        elif not n_hat-radii["n"]>0:
            unavailable = "the covering interval of the norm squared is not strictly positive, so the ratio is unresolved"
        else:
            value = numerator = q_hat/n_hat
            c = bias.observable_bound
            sampling = (samples["q"]+c*samples["n"])/n_hat
            algorithmic = (bias_q+c*bias_n)/n_hat
    else:
        name = "n" if isinstance(output,NormSquared) else "q"
        value = estimate["norm_squared" if name=="n" else "quadratic_form"]
        numerator = value if name=="q" else None
        sampling = estimate[f"sampling_radius_{name}"]
        algorithmic = estimate[f"bias_{name}"]
        if value is None:
            unavailable = "the raw sampled estimate is not representable in binary64"
    facts = (fact("algorithmic_approximation",frame,algorithmic,premise=rec.psd_premise,
                  reason="; ".join(bias.reasons) or "the deterministic bias of this output is unavailable"),
             _sampling_fact(frame,sampling if value is not None else None,failure))
    return LCHSAnalysis(**result_fields(plan,data),value=value,unavailable=unavailable,numerator=numerator,
        numerator_frame='unit' if isinstance(output,NormalizedExpectation) else 'physical' if isinstance(output,QuadraticForm) else None,
        submitted_shots=rows.clusters*rows.repeats,returned_shots=returned,selected_shots=moments.shots,
        pair_moments=moments,pair_estimate=pair_estimate,facts=facts)


def analyze_quantum(plan,data,chunks):
    """Reduce the simulated LCHS vector, exact reduction or counts in original coordinates."""
    from nwqlib._quantum_readout import pair_float, pair_ratio, physical_shot_count, recover_scaled_pair, reduce_sample_arrays
    from nwqlib.core.records import FrozenArray
    from .primary_records import LCHSSamples
    rec,output = plan.reconstruction,plan.output
    if rec.pair_rows is not None:
        return _pair_analysis(plan,data,chunks)
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
            # Each array is copied into immutable storage and released before the
            # next, so the handoff holds at most 24 bytes per sampled index.
            indices = FrozenArray(indices)
            counts = FrozenArray(counts)
            samples = LCHSSamples(indices=indices,counts=counts)
            del indices,counts
            if selected==0:
                unavailable = 'no observed original-coordinate algorithm-success shots'
        elif isinstance(output,NormSquared):
            setting = rec.settings[0]
            submitted,returned = first.observation.shots,first.returned_shots
            selected = physical_shot_count(first,observed=observed_bits(rec),success=success,
                coordinates=rec.system_bits if setting.physical_projection else (),
                dimension=rec.dimension if setting.physical_projection else None)
            # A backend may return no shot, which leaves the physical mass undefined rather than zero.
            norm = value = physical_moment(selected/returned,rec.recovery) if returned else None
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
        for term in plan.error_model.terms if term.role=='listed_only' and adds_to_total(term.name))
    from .solution_error_budget import propagate_physical_error
    propagated = propagate_physical_error(output,component_values,radius=None if scale is None else scale.as_float(),
        observable_norm=fsum(abs(c) for _,c in rec.terms) if rec.terms else None)
    # Exact readout contributes a proved zero sampling error. Counts of a moment output carry their shot radius,
    # with the deterministic bias kept in algorithmic_approximation; Samples leave it unknown.
    sampling = exact_readout_sampling(plan,data.observations,random_draws=False)
    if not sampling and isinstance(output,(NormSquared,QuadraticForm,NormalizedExpectation)) and unavailable is None:
        # An accuracy request owns the confidence of its Plan (LCHS.sampling_shots), whose shots it chose.
        accuracy = plan.selection_accuracy
        # A sampled-pair Method reaches these counts only through its exact no-evolution readout, whose
        # radius does not depend on the estimator, so either estimator's fail_prob applies.
        failure = plan.method.estimator.fail_prob if accuracy is None else 1.0-accuracy.confidence
        radius,reason = _counts_radius(rec,output,groups,value,failure,returned=first.returned_shots)
        sampling = (_sampling_fact(frame,radius,failure,assumption=_COUNTS_SAMPLING,
                                   reason=reason or "the shot radius is not representable in binary64"),)
    facts = (fact('algorithmic_approximation',frame,propagated,premise=rec.psd_premise),
             *(sampling or (fact('sampling',frame),)))
    return LCHSAnalysis(**result_fields(plan,data),value=value,unavailable=unavailable,norm_squared=norm,physical_scale=scale,
        numerator=numerator,numerator_frame='unit' if isinstance(output,NormalizedExpectation) else 'physical' if isinstance(output,QuadraticForm) else None,
        artifact=artifact,samples=samples,reduction=reduction,groups=groups,submitted_shots=submitted,
        returned_shots=returned,selected_shots=selected,facts=facts)
