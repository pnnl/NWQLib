"""Explicit verification of a QHD result against finite-grid references (``QHDVerification``).

``grid_minimum`` evaluates the original objective once as an array on the
selected grid and reports the gaps of the candidate and of the most probable
point to the least evaluated binary64 value of that one reference table.
A fidelity comparison runs one restricted host evolution, ``schrodinger``
or ``ir_product``, and compares it with the stored state, restricted to the grid amplitudes of the one-hot or binary
register. Every fact is conditioned on the selected finite grid or on the
normalized valid state, and its relation text says whether the reference
shares the producer's model, so that a replay by the same producer is not
read as independent accuracy evidence. Work and bytes are admitted before
any evaluation or evolution.
"""

from math import fsum
from nwqlib.core.records import Float64, Unit
from nwqlib.evidence.error_model import CheckDomain, CheckSpec, ErrorFrame, FramedFact
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.evidence.verification import witness_check_facts
from nwqlib.execution import (
    KernelApplication,
    VerificationReceipt,
    verification_invocation,
)
from nwqlib.ir import Binding
from .records import QHDVerification
from .method import _admit_observations, _bits, _decoder, _generator_norms, _grid, restricted_sizes
from .objective import node_count


def verify(plan, result, *, checks):
    """Run one ``QHDVerification`` and return ``(receipt, facts)``.

    ``facts`` holds one receipt-witnessed fact per selected comparison, in
    the order of ``checks.verification_checks(result)``, ready for
    ``Certificate.with_verification`` with the same options. The receipt
    also keeps the companion facts, such as the reference minimum and the
    raw infidelity.
    """
    if type(checks) is not QHDVerification:
        raise TypeError("checks requires one QHDVerification")
    result.validate_plan(plan)
    return _verify(plan, result, checks)


def verification_checks(options, result):
    """Return one CheckSpec per selected comparison of ``options``.

    ``grid_minimum`` checks the candidate's objective gap to the grid
    minimum, in the Problem's objective unit, against
    ``objective_gap_tolerance``. A fidelity comparison checks the clipped
    infidelity ``max(0, 1 - F)`` of the named restricted reference against
    its infidelity tolerance. The check names equal the fact names
    ``grid_minimum``, ``schrodinger_infidelity`` and
    ``ir_product_infidelity``.
    """
    plan = result.plan
    result.validate_plan(plan)
    _admit_product(plan, options)
    source, relations = _source(plan, options)
    fields = dict(claim_id=plan.output.content_id, source=source, options_id=options.content_id,
        prerequisites=(), experiments=0, access=("observed candidate and, for fidelity, the stored state",),
        classical_work="original-objective grid evaluation or one restricted evolution per fidelity comparison",
        reference_work=source.reference, data_description="scalar gap, infidelity and their companion facts")
    checks = []
    for comparison in options.comparisons:
        if comparison == "grid_minimum":
            name, threshold, domain = comparison, options.objective_gap_tolerance, CheckDomain(lower=0.)
        else:
            name = comparison.replace("_fidelity", "_infidelity")
            threshold = (options.schrodinger_infidelity_tolerance if comparison == "schrodinger_fidelity"
                         else options.ir_product_infidelity_tolerance)
            domain = CheckDomain(lower=0., upper=1.)
        metric = "objective_gap" if comparison == "grid_minimum" else "infidelity"
        frame = _frame(plan, name, metric, comparison, relations[comparison])
        checks.append(CheckSpec(name=name, frame=frame, threshold=threshold, domain=domain, **fields))
    return tuple(checks)


def _admit_product(plan, options):
    """Reject an IR-product comparison when the Plan did not select that product."""
    if "ir_product_fidelity" in options.comparisons and not plan.reconstruction.compact_schedule_selected:
        raise ValueError("the IR product was not selected; create a new Plan to select it")


def _source(plan, options):
    """Return the options' Source, naming each comparison's relation, and the relations."""
    relations = {name: _comparison_relation(plan, name) for name in options.comparisons}
    return options.source.revise(reference="; ".join(relations.values())), relations


_OBJECTIVE_METRICS = frozenset({"objective_gap", "objective", "objective_difference"})


def _frame(plan, name, metric, comparison, relation, *, objective_unit=None):
    """Return the ErrorFrame of one fact of ``comparison``.

    The unit is the Problem's objective unit for objective metrics, or when
    ``objective_unit`` is true, and dimensionless otherwise. Facts of
    ``grid_minimum`` are conditioned on the selected finite grid, and facts
    of a fidelity comparison on the normalized valid one-hot state or the
    normalized binary register state, which holds only grid points. The
    comparison's producer and reference relation completes the
    conditioning.
    """
    objective = metric in _OBJECTIVE_METRICS if objective_unit is None else objective_unit
    unit = plan.problem.evidence_unit if objective else Unit(symbol="1", dimension="dimensionless")
    base = ("selected finite grid" if comparison == "grid_minimum" else "normalized valid one-hot state"
            if plan.method.encoding == "one_hot" else "normalized binary register state")
    return ErrorFrame(quantity=name, metric=metric, unit=unit, scope=plan.problem.evidence_scope,
                      conditioning=f"{base}; {relation}")


def _comparison_relation(plan, comparison):
    """Return text naming what produced the stored result and what the reference compares."""
    if comparison == "grid_minimum":
        return ("binary64 original-objective reevaluation, including the constant, on the same finite grid. "
                "The gap, threshold decision and success mass use evaluated values. Objective-evaluation "
                "error is not bounded by this check, and equal binary64 values are not separated")
    flavor = comparison.removesuffix("_fidelity")
    if plan.execution == "classical":
        producer = plan.method.theory_flavor
        relation = ("same-producer replay consistency; not independent accuracy evidence"
            if producer == flavor else "cross-model comparison; grid, objective and schedule are shared")
    else:
        producer = "quantum circuit"
        relation = ("circuit-versus-IR construction consistency; not independent discretization evidence"
            if flavor == "ir_product" else "circuit-versus-restricted evolution; grid, objective and schedule are shared")
        if flavor == "ir_product" and plan.method.encoding == "binary":
            relation += ("; the binary IR reference evaluates reconstructed diagonal phase arrays, whose "
                         "difference from the separate native rotations is not propagated into this comparison")
    if flavor == "schrodinger" and plan.method.kinetic_model != "finite_difference":
        # The schrodinger reference evolves the finite-difference stencil, so
        # its fidelity then compares two kinetic models.
        relation += f"; kinetic models differ, {plan.method.kinetic_model} producer and finite_difference reference"
    return f"producer={producer}; reference=restricted {flavor}; {relation}"


def _verify(plan, result, choice):
    """Compare the observed candidate or stored state with the requested finite-grid references.

    ``grid_minimum``: the explicit grid reference evaluates the original
    objective and required constraints on the declared original-variable
    grid. It chooses the lexicographically first point attaining the least
    computed feasible objective. Feasibility uses the normalized residual
    test at the stored tolerance. All reported reference values come from
    the same evaluated table. The result describes this finite numerical
    grid and supplies neither a continuous optimum nor an
    expression-evaluation error certificate. The table
    is one fresh array evaluation of the original objective at the original
    coordinates (``compiler.evaluate_support``), never the Plan's support
    tables. The first-index ``argmin`` of the C-order table is the
    lexicographically first minimum, and every grid
    point of this unconstrained reference is feasible. The candidate gap,
    the most probable point's gap (``most_probable_gap``), the reference
    minimum and ``reported_value_difference`` read that one table. The
    success mass is ``fsum`` of the observed valid weights whose rounded
    ``abs(table - minimum)`` passes ``minimum_tolerance``, the current
    subtraction rule; a difference that overflows cannot pass a finite
    tolerance. Vectorized expression evaluation may change the reference
    table's entries and can change a minimum or feasibility decision near a
    threshold, compared with the former scalar evaluation at each point.
    Objective-evaluation error is not bounded here. Each fidelity comparison runs one
    restricted evolution and compares it with the stored state. A native
    state is first restricted to its grid amplitudes in lexicographic order,
    its one-hot states (``decoding.onehot_register_indices``) or the whole
    binary register (``binary.lexicographic_register_array``),
    so the fidelity is conditional on the valid subspace. Work and bytes are
    admitted before any evaluation or evolution.
    """
    import numpy as np
    from .method import _evolve_restricted
    from .validation import _lambdify_objective

    if result.candidate is None:
        raise ValueError("verification requires an observed valid candidate")
    checks = choice.verification_checks(result)
    grid = _grid(plan)
    dimension = plan.reconstruction.restricted_dimension
    flavors = [
        name.removesuffix("_fidelity") for name in choice.comparisons if name != "grid_minimum"
    ]
    if flavors and result.artifact is None:
        raise ValueError("fidelity requires keep_state=True on the original acquisition")
    sizes = [_reference_sizes(plan, grid, f) for f in flavors]
    # Each fidelity evolution forms its start vector, the tensor product of the
    # stored factors or the direct uniform fill (initial_state.start_vector_work),
    # which _reference_sizes includes, and adds the sizes of its own evolution.
    # Without grid_minimum the base allowance over the D = K**d grid tuples
    # stays: per tuple d coordinates and an untuned 8 units for the
    # restriction of a stored native state and the comparison, and 8 bytes
    # per grid index plus an untuned 64. With grid_minimum the grid charge is
    # W_grid = D (d + N_f + 12) and _grid_minimum_bytes. With L = max_work // D and
    # complete tree size N_f, node_count returns n = min(N_f, L+1). n <= L
    # proves completion. Otherwise D*n > max_work already proves refusal,
    # but the resulting work is only a lower bound on the complete charge.
    nodes = 0
    work_is_lower_bound = False
    grid_minimum = "grid_minimum" in choice.comparisons
    fidelity_work, fidelity_bytes = sum(s[0] for s in sizes), sum(s[1] for s in sizes)
    chunk = None
    if grid_minimum:
        node_limit = choice.max_work // dimension
        nodes = node_count(plan.problem.objective, node_limit)
        work_is_lower_bound = nodes > node_limit
        work = dimension * (grid.num_variables + nodes + 12) + fidelity_work
        # The stored artifact and the restricted state stay live (B_held), and the fidelity
        # workspaces are added, not overlapped, with the grid phases. The one-hot ir_product
        # reference law already holds the restricted stored state, 16D bytes, as its held bytes
        # (_reference_sizes), which _held_state_bytes counts too, so that state is counted once.
        held = fidelity_bytes + _held_state_bytes(plan, result)
        if "ir_product" in flavors and plan.method.encoding != "binary":
            held -= 16 * dimension
        chunk, grid_bytes = _grid_minimum_bytes(grid.num_variables, grid.num_grid_points, nodes,
                                                max_bytes=choice.max_bytes, held_bytes=held)
        data_bytes = held + grid_bytes
    else:
        work = dimension * (grid.num_variables + 8) + fidelity_work
        data_bytes = dimension * (64 + 8 * grid.num_variables) + fidelity_bytes
    if work > choice.max_work or data_bytes > choice.max_bytes or (grid_minimum and chunk is None):
        work_requirement = f"at least {work}" if work_is_lower_bound else str(work)
        # The grid-minimum byte law grows with the node count, so a truncated count also bounds the bytes below.
        bytes_lower_bound = work_is_lower_bound and grid_minimum
        byte_requirement = f"at least {data_bytes}" if bytes_lower_bound else str(data_bytes)
        remedies = []
        if work_is_lower_bound:
            remedies.append(
                "The objective node count stopped at the current work limit. "
                "The displayed work and bytes are lower bounds on the full admission charges. "
                "Increase QHDVerification.max_work to count further. "
                "Using the displayed lower bound can still refuse"
            )
        elif work > choice.max_work:
            remedies.append(f"Set QHDVerification(max_work={work}) or larger to clear this work admission")
        if data_bytes > choice.max_bytes and not bytes_lower_bound:
            remedies.append(f"Set QHDVerification(max_bytes={data_bytes}) or larger to clear this byte admission")
        if len(choice.comparisons) > 1:
            remedies.append("Selecting fewer reference comparisons can reduce their charges")
        raise ValueError(
            f"QHD explicit verification requires {work_requirement} work units and {byte_requirement} bytes, "
            f"with QHDVerification(max_work={choice.max_work}, max_bytes={choice.max_bytes}). "
            + ". ".join(remedies)
            + ". These limits belong to QHDVerification, not QHD"
        )
    observations = result.data.observations
    _admit_observations(plan, observations, result=result)
    values = []
    source, relations = _source(plan, choice)

    def metric(name, value, kind, *, comparison="grid_minimum", objective_unit=None):
        """Append one framed fact of metric ``kind``, unknown when ``value`` is None.

        The frame's conditioning names the relation of its comparison, for
        example a same-producer replay. The fact carries no assumptions,
        because an assumption would leave its check INCONCLUSIVE in
        ``Certificate.with_verification``.
        """
        frame = _frame(plan, name, kind, comparison, relations[comparison], objective_unit=objective_unit)
        fact = Fact(
            quantity=name,
            unit=frame.unit,
            scope=frame.scope,
            availability="unknown" if value is None else "concrete",
            reason="required acquisition data unavailable" if value is None else None,
            value=None if value is None else Float64(value=value),
            evidence=None
            if value is None
            else Evidence(kind="numerical_estimate", source=source),
        )
        values.append(FramedFact(frame=frame, bindings=(), fact=fact))

    state = None
    if result.artifact is not None:
        state = result.data.artifact(result.artifact).array
        if plan.execution == "quantum":
            # One-hot states (decoding.onehot_register_indices) or the binary variable-axis
            # permutation (binary.lexicographic_register_array), in lexicographic order.
            if plan.method.encoding == "binary":
                from .binary import lexicographic_register_array

                state = lexicographic_register_array(state, grid.num_variables, _bits(plan.method))
            else:
                from .decoding import onehot_register_indices

                state = state[onehot_register_indices(grid.num_grid_points, grid.num_variables).reshape(-1)]
    # Evaluate the original objective on the selected finite grid. This
    # checks grid optimality and reported values, not the continuum minimum.
    if "grid_minimum" in choice.comparisons:
        from .compiler import evaluate_support

        evaluator = _lambdify_objective(plan.problem.variables, plan.problem.objective)
        # Original coordinates, filled one scalar at a time into the final axis arrays (A_axis = 0).
        axes = []
        for j in range(grid.num_variables):
            axis = np.empty(grid.num_grid_points, dtype=np.float64)
            for i in range(grid.num_grid_points):
                axis[i] = grid.grid_value(j, i)
            axes.append(axis)
        # One fresh C-order table of the original objective (the mutable evaluation output, not frozen).
        table = evaluate_support(evaluator, axes, chunk, expression=plan.problem.objective,
                                 support=tuple(range(grid.num_variables)))
        # Every grid cell is feasible, so the first-index argmin is the first lexicographic minimum.
        minimum = float(table[int(np.argmin(table))])
        shape = (grid.num_grid_points,) * grid.num_variables

        def value(indices):
            return float(table[int(np.ravel_multi_index(indices, shape))])

        metric("grid_minimum", value(result.candidate_indices) - minimum, "objective_gap")
        # The point that the outer layers read by default, from the same table.
        metric("most_probable_gap", value(result.most_probable_indices) - minimum, "objective_gap")
        metric("reference_minimum", minimum, "objective")
        metric(
            "reported_value_difference",
            result.objective - value(result.candidate_indices),
            "objective_difference",
        )

        def passing_weights():
            # The rounded abs(table - minimum) test at minimum_tolerance, in slabs of at most chunk entries.
            for start in range(0, dimension, chunk):
                stop = min(dimension, start + chunk)
                probabilities = np.abs(state[start:stop]) ** 2
                yield from probabilities[np.abs(table[start:stop] - minimum) <= choice.minimum_tolerance].tolist()

        success = None
        if state is not None:
            success = fsum(passing_weights())
        elif plan.execution == "quantum":
            success = _observed_success_mass(plan, observations, table, minimum, choice.minimum_tolerance)
        metric("minimum_success_mass", success, "probability")
    # Each requested fidelity comparison performs its own bounded host
    # evolution and compares normalized valid-sector states.
    for flavor in flavors:
        reference = _evolve_restricted(plan, flavor)
        raw, window = _infidelity(state, reference)
        comparison = flavor + "_fidelity"
        metric(flavor + "_infidelity", max(0.0, raw), "infidelity", comparison=comparison)
        metric(flavor + "_raw_infidelity", raw, "signed_difference", comparison=comparison)
        metric(flavor + "_roundoff_window", window, "infidelity_roundoff_window", comparison=comparison)
    measured = {
        item.fact.quantity: item.fact.value.value
        for item in values
        if item.fact.availability == "concrete"
    }
    for name in choice.comparisons:
        tolerance = (
            choice.objective_gap_tolerance
            if name == "grid_minimum"
            else choice.schrodinger_infidelity_tolerance
            if name == "schrodinger_fidelity"
            else choice.ir_product_infidelity_tolerance
        )
        key = name if name == "grid_minimum" else name.replace("_fidelity", "_infidelity")
        metric(name + ".threshold", tolerance, "threshold", comparison=name,
               objective_unit=name == "grid_minimum")
        metric(name + ".within_tolerance", float(measured[key] <= tolerance), "indicator", comparison=name)
    arguments = dict(
        grid_evaluations=dimension if "grid_minimum" in choice.comparisons else 0,
        schrodinger_evolutions=int("schrodinger" in flavors),
        ir_product_evolutions=int("ir_product" in flavors),
        native_executions=0,
        known_work=work,
        known_bytes=data_bytes,
    )
    application = KernelApplication(
        name="qhd_explicit_reference",
        implementation=source,
        arguments=tuple(Binding(parameter=name, value=value) for name, value in arguments.items()),
        facts=tuple(values),
    )
    receipt = VerificationReceipt(
        invocation_id=verification_invocation(),
        plan_id=plan.content_id,
        result_id=result.content_id,
        construction_id=result.construction_id,
        artifact_ids=() if result.artifact is None else (result.artifact.content_id,),
        reference=source,
        options_id=choice.content_id,
        applications=(application,),
    )
    return receipt, witness_check_facts(receipt, result, checks)


def _held_state_bytes(plan, result):
    """Return the bytes of the stored state that stay live during verification (``B_held``).

    The stored artifact's complex128 array, 16 bytes per entry, and, for a
    native Plan, its restriction to the ``K**d`` grid amplitudes, another
    16D bytes. Without a stored state nothing is held.
    """
    if result.artifact is None:
        return 0
    entries = result.data.artifact(result.artifact).array.size
    restricted = plan.reconstruction.restricted_dimension if plan.execution == "quantum" else 0
    return 16 * (entries + restricted)


def _grid_minimum_bytes(d, k, nodes, *, max_bytes, held_bytes):
    """Return ``(chunk, grid_bytes)`` of the grid-minimum reference table, or ``(None, minimum)`` when refused.

    Grid-minimum verification charges the selected original-expression
    evaluator and its bounded coordinate/selection slabs. Fidelity
    references have their separate evolution costs, and simultaneous stored
    states remain included in the verification peak.

    For a slab of b full-grid points,
    ``B_grid(b) = 8dK + H_eval + W_eval,f(b) + (8d+96)b`` and
    ``W_grid = D(d+N_f+12)``. The generous constraint-free 96-byte wrapper
    rate is shared with the constrained owner. It covers coordinate/index
    construction, the objective array, finite-real conversion,
    absolute/maximum/normalization temporaries, a mask and a ``where``
    result under a sequential evaluator schedule. A full D-entry objective
    table adds ``8D``. This table is the mutable evaluation output and is
    never independently frozen, so with ``F = 8dK + 8D + H_eval`` and
    ``S(b) = W_eval,f(b) + (8d+96)b`` the law is ``F + max(A_axis, S(b))``.
    The axes are filled one scalar at a time into their final
    ``np.empty(K, dtype=np.float64)`` arrays, so only a bounded number of
    Python scalar temporaries are live, covered by ``H_eval``, no ``40K``
    list population exists and ``A_axis = 0``. The success-mass selection
    reads the table and the state in slabs of at most b entries, so its
    growing temporaries stay within ``S(b)``. Adding the concurrently held
    data gives ``B_held + 8dK + 8D + H_eval + S(b) <= QHDVerification.max_bytes``.

    ``W_eval,f(b) = 8b(N_f+2)`` is an engineering allowance, not a derived
    bound: at most one slab-sized float64 temporary per expression node,
    plus the output and one broadcast input (``compiler.support_workspace``).
    ``H_eval`` is the completed header inventory, ``H0 + 256 N_objects`` with
    ``N_objects`` the ``N_f + 2`` evaluator arrays of ``support_workspace``,
    the d axes and their list, the table, the ``d + 3``
    coordinate/index/quotient arrays and the three conversion/validation
    arrays of an evaluation slab (``compiler.chunk_coordinates``,
    ``compiler.evaluate_support``), and the seven arrays of a success-mass
    selection slab, ``N_f + 2d + 16`` in all.

    With the rate ``a = 8(N_f+2) + 8d + 96`` and
    ``R = max_bytes - B_held - F``, the largest admitted slab is
    ``b = min(D, R // a)`` when ``R >= a``, and the minimum total byte limit
    for one-entry slabs is ``B_held + F + a``.
    """
    from .compiler import H0, OBJECT_HEADER_BYTES

    dimension = k**d
    h_eval = H0 + OBJECT_HEADER_BYTES * (nodes + 2 * d + 16)
    resident = 8 * d * k + 8 * dimension + h_eval
    rate = 8 * (nodes + 2) + 8 * d + 96
    if held_bytes + resident + rate > max_bytes:
        return None, resident + rate
    chunk = min(dimension, (max_bytes - held_bytes - resident) // rate)
    return chunk, resident + rate * chunk


def _observed_success_mass(plan, observations, table, minimum, tolerance):
    """Return the ``fsum`` of the observed valid weights whose points pass the rounded minimum test.

    Each chunk's histogram is decoded by array operations
    (``decoding.onehot_valid_words`` and ``decoding.onehot_local_indices``
    for one-hot words, the base-K digits of the register index for binary)
    and each passing weight is ``count/N`` over all returned shots N for
    counts, or ``value/C`` over C chunks, as before. A readout wider than one
    64-bit index is decoded entry by entry (``method._decoder``).
    """
    import numpy as np
    from .decoding import onehot_local_indices, onehot_valid_words

    grid = _grid(plan)
    d, k = grid.num_variables, grid.num_grid_points
    bits = None if plan.method.encoding != "binary" else _bits(plan.method)
    counts_total = sum(c.returned_shots or 0 for c in observations.chunks)
    passing = []
    for chunk in observations.chunks:
        histogram = chunk.histogram()
        counts = chunk.observation.kind == "counts"
        denominator = counts_total if counts else len(observations.chunks)
        if histogram.width > 64:
            decode = _decoder(plan)
            for index, weight in zip(histogram.index_list(), histogram.weights.tolist(), strict=True):
                point = decode(index)
                if point is not None and abs(table[int(np.ravel_multi_index(point, (k,) * d))] - minimum) <= tolerance:
                    passing.append(weight / denominator)
            continue
        words, weights = histogram.indices(), histogram.weights
        if bits is None:
            valid = onehot_valid_words(words, k, d)
            points = onehot_local_indices(words[valid], k, d)
        else:
            valid = np.ones(words.shape, dtype=bool)
            mask = np.uint64(k - 1)
            points = np.stack([((words >> np.uint64(j * bits)) & mask).astype(np.int64) for j in range(d)],
                              axis=-1).reshape(-1, d)
        flat = np.ravel_multi_index(tuple(points.T), (k,) * d) if d else np.zeros(0, dtype=np.int64)
        selected = np.abs(table[flat] - minimum) <= tolerance
        passing.extend(weight / denominator for weight in weights[valid][selected].tolist())
    return fsum(passing)


def _reference_sizes(plan, grid, flavor):
    """Return ``(work, bytes)`` of one restricted reference evolution of ``flavor``.

    The binary ``ir_product`` reference evaluates QFT operations and
    computed diagonal phase arrays (``theory.binary_product_sizes``). The
    one-hot ``ir_product`` reference applies each stored block directly
    (``theory.onehot_product_sizes``), with the start vector's explicit
    payload ``24 D + 8 d K`` (``initial_state.restricted_state``) and the
    restricted stored state of 16D bytes, which stays live for the
    comparison, as held bytes. The ``schrodinger`` reference uses
    ``expm_multiply`` calls (``method.restricted_sizes``). All include the
    start vector formed from the stored factors
    (``initial_state.start_vector_work``).
    """
    from .initial_state import start_vector_work

    r, method = plan.reconstruction, plan.method
    start = start_vector_work(method.initial_state, grid)
    if flavor == "ir_product" and method.encoding == "binary":
        from .theory import binary_product_sizes

        synthesis = method.binary_synthesis
        return binary_product_sizes(
            r.restricted_dimension, _bits(method), grid.num_variables,
            [grid.num_grid_points ** len(t.support) for t in r.support_values], r.steps,
            cutoff=synthesis.aqft_cutoff, swaps=synthesis.qft_bit_reversal == "swap",
            walsh=synthesis.potential != "dense_diagonal", start_work=start)
    if flavor == "ir_product":
        from .theory import onehot_product_sizes

        dimension, k = r.restricted_dimension, grid.num_grid_points
        return onehot_product_sizes(dimension, k, r.steps, start_work=start,
                                    start_bytes=24 * dimension + 8 * grid.num_variables * k,
                                    held_bytes=16 * dimension)
    return restricted_sizes(
        r.restricted_dimension, grid.num_variables, len(r.support_values), (r.step_weights, r.steps), flavor,
        _generator_norms(flavor, method, grid, r.support_values, r.steps, r.step_weights), start)


def _infidelity(left, right):
    """Return ``(raw, window)`` with ``raw = 1 - F`` for the fidelity F of two vectors.

    ``F = |<x,y>|**2 / (<x,x> <y,y>)`` and ``window`` come from
    ``nwqlib._numerics.normalized_fidelity_with_window``, whose docstring
    derives the window. The kernel raises when the computed F is not finite
    or exceeds one by more than ``window``, so ``raw >= -window``. The
    computed F is never negative, so ``raw <= 1`` for every accepted input,
    and no upper check on ``raw`` is needed. ``1 - F`` is exact near one by
    Sterbenz's lemma (Higham, *Accuracy and Stability of Numerical
    Algorithms*, 2nd ed., doi:10.1137/1.9780898718027, Sec. 2.5,
    Theorem 2.5). The window covers evaluation roundoff of the supplied
    vectors only, not the accuracy of the evolutions that produced them.
    """
    from nwqlib._numerics import normalized_fidelity_with_window

    fidelity, window = normalized_fidelity_with_window(left, right)
    return 1.0 - fidelity, window
