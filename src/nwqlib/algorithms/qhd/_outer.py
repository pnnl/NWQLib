"""Rules shared by the QHD layers that run a sequence of QHD Plans.

The augmented-Lagrangian layer (``constrained.solve_augmented_lagrangian``)
and box refinement (``refinement.refine_box``) each run one ordinary QHD Plan
per round or level through ``prepare`` and ``submit``. This module owns what
both do the same way: the arguments they share with ``nwqlib.solve``, the
random streams of each inner Plan, the remainder of the cumulative execution
limits, the counts read from an inner Run and its selected construction, the
sums of those counts, and the point that a point rule reads from an inner
result. Each layer keeps its own records and decides what it compares. When
the augmented-Lagrangian layer runs box refinement in each round, this module
also owns how the refinement's stop decides the round (``refinement_stop``).
"""

from dataclasses import dataclass

# Counts that an inner Run's trace supplies (``run_counts``) and that the
# cumulative limits subtract (``round_limits``).
RUN_COUNTS = ("circuit_preparations", "circuit_attempts", "completed_circuit_attempts", "shots",
              "completed_shots", "data_bytes", "evolution_work", "construction_work", "synthesis_work")
USED_COUNTS = ("circuit_preparations", "circuit_attempts", "shots", "data_bytes", "synthesis_work")


def check_arguments(qhd, execution, shots, seed, limits, *, executed=True):
    """Check the arguments that a layer shares with ``nwqlib.solve``, and QHD's type.

    ``execution`` None means ``"quantum"``, and ``limits`` None the
    ExecutionLimits defaults. A QHD with ``initial_state_preparation="none"``
    plans a resource-only construction that no round can execute, so it is
    refused before the first round, unless the caller only plans
    (``executed=False``, ``constrained.plan_augmented_lagrangian``).

    Returns:
        ``(execution, limits)`` with those defaults applied.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.execution import ExecutionLimits
    from nwqlib.scientist import _check_shots
    from .method import QHD

    if type(qhd) is not QHD:
        raise TypeError("qhd must be a configured QHD Method")
    execution = "quantum" if execution is None else execution
    if execution not in ("quantum", "classical"):
        raise ValueError("execution must be 'quantum' or 'classical'")
    _check_shots(shots)
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    limits = ExecutionLimits() if limits is None else limits
    if type(limits) is not ExecutionLimits:
        raise TypeError("limits must be ExecutionLimits")
    if executed and qhd.initial_state_preparation == "none":
        raise ApplicabilityError("initial_state_preparation='none' plans a resource-only QHD construction, "
                                 "which no round can execute")
    return execution, limits


def plan_round(problem, qhd, execution, shots, child):
    """Plan one round's ``problem`` with ``qhd`` from the random streams of ``child``.

    A layer draws ``root = numpy.random.SeedSequence(seed)`` once and gives
    every round its own ``child = root.spawn(1)[0]``, as
    ``nwqlib.scientist.compare`` gives every candidate its own child. The
    streams of one round then do not shift those of another, an integer seed
    reproduces every round, and the child's entropy and spawn key, which the
    layer records, identify the streams. ``_plan_with_streams`` is the internal
    entry that ``compare`` uses, since the public ``plan`` takes only an
    integer seed. The output is the problem's default, an
    ``OptimizationCandidate``.
    """
    from nwqlib.core.planning import RandomStreams
    from nwqlib.scientist import _plan_with_streams

    return _plan_with_streams(problem, qhd, problem.default_output(), execution, shots,
                              RandomStreams.from_sequence(child))


def round_limits(limits, used, execution, shots):
    """Return ``(ExecutionLimits, None)`` for the next round's Run, or ``(None, reason)`` when none is funded.

    ``limits`` caps the whole run cumulatively, and ``used`` holds the
    ``USED_COUNTS`` of the rounds so far. A Run caps its circuit preparations
    and its circuit attempts separately by ``max_total_circuits``, so the
    remainder subtracts the larger of the two totals. Shots, data bytes and
    synthesis work subtract their totals. The per-Run caps (qubits,
    simulator memory, completion metadata, direct amplitudes) pass through
    unchanged. A quantum round needs at least one circuit and its shots, and
    every cap of ExecutionLimits is positive, so every round needs a
    positive remainder of each capped quantity. A remainder that passes this
    test can still be too small for the Run's own reservations, which the
    Run then refuses.

    A refusal of the first round can end in two ways. When this test refuses
    it, for example because ``shots`` exceeds ``max_total_shots``, the layer
    knows before starting the round that it cannot be funded, and ends the
    run with ``budget_exhausted`` after zero completed rounds, reporting the
    short limits. When the first round's Run refuses its own reservations,
    the refusal is an error of the round's inner preparation or execution,
    and ``inner_failure`` states the rule for those. Before any outer round
    has completed, the original inner error propagates, which keeps its
    diagnosis. With box refinement the rule applies to the levels of the
    first round, so the error propagates only while no level of it has
    completed. A durable Run folder of the refused round keeps its journal
    with the charges it recorded.
    """
    from nwqlib.execution import ExecutionLimits

    circuits = max(used["circuit_preparations"], used["circuit_attempts"])
    left = dict(max_total_circuits=limits.max_total_circuits - circuits,
                max_total_shots=limits.max_total_shots - used["shots"],
                max_data_bytes=limits.max_data_bytes - used["data_bytes"],
                max_synthesis_work=limits.max_synthesis_work - used["synthesis_work"])
    needed = dict(max_total_circuits=1, max_total_shots=shots or 1, max_data_bytes=1, max_synthesis_work=1)
    short = [f"{name} has {left[name]} left of {getattr(limits, name)}"
             for name in left if left[name] < needed[name]]
    if short:
        return None, "the cumulative limits cannot fund another round: " + ", ".join(short)
    kept = {name: getattr(limits, name) for name in ExecutionLimits.model_fields
            if name not in ("schema_version", "parent_id")}
    return ExecutionLimits(**{**kept, **left}), None


def inner_failure(error, completed):
    """Return the ``inner_failed`` text of an inner error, or re-raise the error when nothing has completed.

    ``error`` was raised by the inner planning, preparation or execution of a
    round or level, and ``completed`` counts the rounds or levels of the whole
    run completed before it. In the augmented-Lagrangian layer with box
    refinement these are the earlier rounds plus the levels of the current
    round (``refinement._refine``). Once something has completed, the layer
    stops with ``inner_failed`` and keeps the completed ones, whose results
    and work remain useful. Before that nothing has completed, so a record
    would report nothing, and the error is most often a configuration that
    QHD rejects, such as ``keep_state`` with shots or a ``max_work`` below the
    admitted work of the first problem, or limits that the first round's Run
    refuses (``round_limits``), which should surface at once, so the
    original exception is re-raised. Call it from the ``except`` block that
    caught ``error``, and only for errors of those inner steps.

    Serialization can wrap a KeyboardInterrupt in an Exception. A real
    interrupt in its cause or context is re-raised before classifying the
    failure, even after completed work; exception text alone is not an
    interruption. The visited set also handles cyclic exception chains.

    The augmented-Lagrangian layer applies the same rule when f, h or g is
    not finite and real at the point that a round read
    (``constrained.solve_augmented_lagrangian``). ``completed`` then counts
    the completed rounds, since a round does not complete without a point,
    so the error propagates in round 0 and a later round
    ends the run with ``inner_failed``, keeping the completed rounds, which
    are valid work.

    Returns:
        ``"ExceptionType: message"``, the failure text that the layer records.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, KeyboardInterrupt):
            raise current
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
    if not completed:
        raise error
    return f"{type(error).__name__}: {error}"


def refinement_stop(termination, levels):
    """Return the status with which a round's box refinement ends the augmented-Lagrangian run, or None.

    ``termination`` is the refinement's stopping reason
    (``refinement_records.BoxRefinementResult``) and ``levels`` the number of
    levels it completed. No round and no level is retried.

    - A refinement that completed a level returns the point of the level
      with the least recorded relative value of the round's inner objective, and the round uses its
      projection onto the original variables: the layer evaluates f, h and g there and records the
      multiplier and penalty update. After ``level_limit``,
      ``box_unchanged``, ``split_limit``, ``width_floor``, ``no_improvement``,
      ``flat_objective`` or ``unresolved_objective`` the round's search ended
      normally, the result is None, and the run goes on under the layer's own
      tests. ``split_limit`` (a stall after the options' last stall split)
      follows a completed level, like ``box_unchanged``, whose place it takes
      once the split budget is spent.
    - After ``budget_exhausted``, ``inner_failed`` or ``no_valid_point`` with
      a completed level, the round keeps its point and update and the run
      stops after this round. The remainder that refused a level refuses the
      first level of the next round as well, since that round's remainder
      is no larger. An inner failure or a result without a valid point ends
      a run without refinement too, and no level is retried. The layer first
      applies its own stopping test to the round's point. When the test
      holds, the run ends with the layer's success status
      (``feasible_complementary`` or ``feasible``), since reporting a budget
      or failure status would say that the run ended before its stopping
      test held, and the refinement's reason stays in the round's
      refinement record. When the test fails, the run ends with the
      refinement's reason, which this function returns.
    - A refinement that completed no level has no point, so the round records
      no update and the run ends with the refinement's own reason:
      ``no_valid_point``, ``inner_failed`` (a first-level error in a round
      after the first, since in the first round it propagates,
      ``inner_failure``), ``flat_objective`` or ``unresolved_objective`` when
      the tables of L_k on the preprocessed box show no variation or none it
      can resolve, or ``width_floor`` when a side of the preprocessed box is
      at the refinement's width floor, where grid coordinates stop being
      distinct. Every round starts from the preprocessed box, so
      ``width_floor`` can end only the first round. The last three statuses
      are reachable only with refinement.

    Returns:
        The status that ends the run when the round's own stopping test does
        not hold, or None when the run goes on.
    """
    if not levels or termination in ("budget_exhausted", "inner_failed", "no_valid_point"):
        return termination
    return None


@dataclass(frozen=True)
class HeaderlessRun:
    """The folder of a durable Run whose creation raised before it committed its journal header.

    ``Run.__init__`` stores the selected inputs, the selection file and
    ``run.json`` before it commits the header, and nothing is prepared or
    acquired before that commit (``_durable.header_committed``). So every
    count of ``run_counts`` is a known zero except ``data_bytes``, the
    stored size of the files in its Run folder apart from the journal
    (``run.sqlite``, its lock and its rollback journal), which
    ``_durable.closed_trace`` reads, or None with ``reason`` when the folder
    cannot be read. The journal is left out because ``max_data_bytes``
    charges its rows, not its file (``_run_archive.stored_file_bytes``). The
    Run has no recorded identity, because only the header names it. When
    the journal itself cannot be read, ``closed_trace`` gives a reason in
    place of this record, and every count is unknown.
    """

    data_bytes: int | None
    reason: str | None = None


def run_counts(trace, reason=None):
    """Return ``(counts, unavailable)``: the ``RUN_COUNTS`` of one inner Run.

    With a trace, read after the Run finished or failed so that it holds
    every charge, including data recorded after the Result's own snapshot:
    circuit preparations are ``preparations - host_preparations``, circuit
    attempts and shots count the events of every status, the completed counts
    the events with status ``completed``, and construction work includes the
    classical kernel's setup (``host_preparation_work_reserved``). A
    ``HeaderlessRun`` in place of the trace gives zero for every count except
    its ``data_bytes``. Without a trace and without ``reason`` no Run was
    started, so every count is zero. Without a trace and with ``reason`` a
    Run was started but its counters could not be read, as when a
    preparation raises and closes an in-memory Run (``_durable.closed_trace``
    reads the counters of a durable one), so every count is None and
    ``unavailable`` names it with that reason.
    """
    if isinstance(trace, HeaderlessRun):
        counts = dict.fromkeys(RUN_COUNTS, 0)
        counts["data_bytes"] = trace.data_bytes
        return counts, () if trace.data_bytes is not None else (("data_bytes", trace.reason),)
    if trace is not None:
        circuits = [event for event in trace.events if event.execution == "quantum_circuit"]
        completed = [event for event in trace.events if event.status == "completed"]
        return dict(circuit_preparations=trace.preparations - trace.host_preparations,
                    circuit_attempts=len(circuits),
                    completed_circuit_attempts=sum(event.execution == "quantum_circuit" for event in completed),
                    shots=sum(event.shots for event in trace.events),
                    completed_shots=sum(event.shots for event in completed),
                    data_bytes=trace.data_bytes, evolution_work=trace.host_work_reserved,
                    construction_work=trace.construction_work_reserved + trace.host_preparation_work_reserved,
                    synthesis_work=trace.synthesis_work_reserved), ()
    if reason is None:
        return dict.fromkeys(RUN_COUNTS, 0), ()
    return dict.fromkeys(RUN_COUNTS), tuple((name, reason) for name in RUN_COUNTS)


def law_count(plan, metric, circuits, reason=None):
    """Return ``(value, reason)``: the law value for ``metric`` of the circuit that a round or level prepared.

    ``circuits`` is the ``circuit_preparations`` count of the round's or
    level's Run (``run_counts``), or None with ``reason`` when the count is
    unknown or the preparation raised. A static QHD Plan has one experiment,
    so its Run prepares at most one circuit. With zero preparations the round
    or level contributes 0, since it compiled no circuit, whether its
    planning raised, its Run's creation raised before the journal header
    (``HeaderlessRun``), it stopped before preparing, or its execution is
    classical. None gives None with ``reason``. A preparation that raised is
    counted as an attempt, and its record does not establish that the
    circuit was built, so the caller passes None and its law is not added
    as that of a prepared circuit. With one or more the value is the
    selected block's ``ResourceLaw`` with that metric in ``plan``, the CX
    upper bound for ``"cx"`` and the arbitrary-rotation count for
    ``"arbitrary_rotations"`` (``resources.rotation_law``, or the binary
    encoding's upper bound on rotation gates), for one circuit and not
    multiplied by shots. A Qiskit state preparation has neither law in the
    record, so the value is then None with that reason. Summed over the
    rounds or levels of a quantum run whose Runs each prepared at most one
    circuit, the values equal the body totals ``sum_e circuits_e R_e`` of
    ``resources.run_resources``, which is None as well when a round's value
    is None.
    """
    if circuits == 0:
        return 0, None
    if circuits is None:
        return None, reason
    laws = [law.value for selection in plan.construction.selections
            for law in selection.resource_laws if law.metric == metric]
    if len(laws) == 1 and isinstance(laws[0], int):
        return laws[0], None
    return None, f"the selected state preparation has no {metric} law"


def total_counts(entries, names):
    """Return ``(counts, unavailable)``: each named count summed over ``entries``, or None when any is unknown.

    ``entries`` pairs a label, such as ``"round 2"``, with a record whose
    counts may be None and whose ``unavailable`` names the reason. An unknown
    count is never replaced by zero or left out of a sum, so a total is None
    when any entry's count is, with the entries' reasons joined as
    ``"label: reason"``.
    """
    counts, reasons = {}, []
    for name in names:
        values = [(label, getattr(item, name), item) for label, item in entries]
        unknown = [(label, item) for label, value, item in values if value is None]
        if unknown:
            counts[name] = None
            reasons.append((name, "; ".join(f"{label}: {dict(item.unavailable)[name]}"
                                            for label, item in unknown)))
        else:
            counts[name] = sum(value for _, value, _ in values)
    return counts, tuple(reasons)


def grid_point(result, rule):
    """Return the grid point that ``rule`` reads from a QHD result, in the result's own coordinates.

    ``best_observed`` reads the candidate, the observed valid point with
    positive weight and the least evaluated binary64 value of the objective
    solved by that level. Under exact readout, when every grid point has
    nonzero probability, it is the grid point with the least binary64 table
    value of the inner objective, the smallest index on ties, whatever the
    evolution did, and with shots the best sampled point. Next to a large
    added constant the table values can tie at every grid point, so that
    point need not be an exact minimizer (docs/algorithms/qhd.md, "Box
    refinement", the example that adds ``2**60 + 1/7`` to the objective).
    ``most_probable`` and ``mode_or_mean`` read the most probable valid
    grid point, whose ties within the readout's tie window go to the
    lexicographically smallest index. It is the point that Wu et al.,
    arXiv:2605.12066v1, Sec. V, extract as a level's candidate. It shows
    where QHD concentrated probability and has no known distance to the grid
    minimizer. The objective value is the inner result's table value. Its
    ``mode_status`` (``QHDAnalysis``) says whether the tie window separates
    that point from every other positive observed valid point.

    Returns:
        A dict with ``kind="grid_point"``, ``indices``, ``point``,
        ``probability``, ``effective_value``, ``effective_value_source="table"``
        and, for the most probable point, ``tie_deficit``, ``tie_window`` and
        ``mode_status``.
    """
    if rule == "best_observed":
        return dict(kind="grid_point", indices=result.candidate_indices, point=result.candidate_coordinates,
                    probability=result.candidate_probability, effective_value=result.value,
                    effective_value_source="table")
    return dict(kind="grid_point", indices=result.most_probable_indices, point=result.most_probable_coordinates,
                probability=result.most_probable_probability, tie_deficit=result.most_probable_deficit,
                tie_window=result.most_probable_tie_window, mode_status=result.mode_status,
                effective_value=result.most_probable_objective, effective_value_source="table")


def mean_point(result, grid_value, value_at):
    """Apply the ``mode_or_mean`` comparison: whether the valid-mass mean position replaces the grid point.

    The mean is ``E[x_j] = sum_i x_j(i) m_j(i) / valid_mass`` from the
    marginals (``QHDAnalysis.position_mean``), in the result's own
    coordinates, and is usually off the grid. ``value_at(mean)`` returns
    ``(value, payload)``, the layer's comparison objective at the mean and
    whatever the layer keeps of that evaluation. The mean replaces the grid
    point only when its value is strictly smaller than ``grid_value``, so a
    tie keeps the grid point. A value that is not finite and real there
    (``value_at`` raising ValueError or OverflowError) keeps the grid point,
    and the reason is returned, since the mean lies off the grid on which the
    layer checked its objective. NumPy's floating-point warnings are silenced
    for that evaluation only.

    The mean is a convex combination of grid coordinates and therefore lies
    in the box in exact arithmetic. Feasibility and a smaller objective do
    not follow, which is why the value at the mean is evaluated and
    compared. When the two values nearly tie, the comparison of computed
    values does not certify the exact order. On a periodic grid the mean of
    coordinates depends on where the box puts the seam, the lower edge of
    each side, so probability that wraps across the seam can give a mean far
    from where the probability lies.

    The experiment code behind the results of Wu et al., arXiv:2605.12066v1,
    whose text describes only the most probable point (Sec. V), takes the
    mean unless the grid point's value is strictly smaller. An exact tie
    therefore takes the mean in that code and keeps the grid point here.
    That code evaluates both points with one function, while the layers
    compare the grid point's table value with a value evaluated at the mean,
    two computations that in the augmented-Lagrangian layer differ by
    roundoff (``constrained._normalized``).

    Returns:
        ``(mean, evaluated, unavailable)``. ``mean`` is the mean when it
        replaces the grid point, otherwise None. ``evaluated`` is
        ``(value, payload)`` of the evaluation, or None when it raised.
        ``unavailable`` is the error text, or None.
    """
    import numpy as np

    mean = tuple(result.position_mean)
    try:
        with np.errstate(all="ignore"):
            evaluated = value_at(mean)
    except (ValueError, OverflowError) as error:
        return None, None, str(error)
    return (mean if evaluated[0] < grid_value else None), evaluated, None


def range_bound(tables):
    """Return ``sum_S (max T_S - min T_S)`` rounded up to binary64, an upper bound on the grid range, or None.

    QHD tabulates an objective ``c + sum_S T_S`` by variable support S, here
    the stored binary64 values of each support table. At every grid point
    ``0 <= T_S - min T_S <= max T_S - min T_S`` for each S, and the sum over S
    gives ``0 <= F - c - sum_S min T_S <= sum_S (max T_S - min T_S)``. So the
    sum bounds ``max F - min F`` over the grid whatever the tables share.
    Tables whose supports overlap need not reach their extremes at the same
    grid point, so the bound can exceed the range. It needs only the tables,
    and no evaluation over the ``K**d`` grid points. QHD's classical
    admission bounds the potential range the same way
    (``method._schrodinger_step_bounds``). The sum is formed exactly from the stored
    values and rounded up, so the returned number is at least the exact sum
    and less than one unit in the last place above it. The result is None
    when the sum exceeds the largest finite binary64 number.
    """
    from fractions import Fraction
    from math import inf, isfinite, nextafter

    exact = sum((Fraction(max(values)) - Fraction(min(values)) for values in tables), Fraction(0))
    try:
        bound = float(exact)
    except OverflowError:
        return None
    # float(Fraction) rounds to nearest, so step up once when it rounded down.
    if Fraction(bound) < exact:
        bound = nextafter(bound, inf)
    return bound if isfinite(bound) else None
