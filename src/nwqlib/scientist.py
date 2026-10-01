"""Natural scientific planning and execution over one selected Plan."""

from dataclasses import dataclass

from nwqlib.algorithms.protocol import ApplicabilityError, Method
from nwqlib.core.planning import Plan, RandomStreams


@dataclass(frozen=True)
class ComparisonRow:
    """One configured candidate and its existing selection/evidence.

    Attributes:
        method: Original configured Method; a selected Plan must have this identity.
        plan: Already selected Plan, or None for an inapplicable candidate.
        reason: Concrete applicability limitation for a row without a Plan.
        estimate: Existing logical fold or profile forecast for this exact Plan.
            Reading a row does not evaluate a model or construct a circuit.
        allocation: Supplied allocation kept with its forecast, if any.
        facts: Immutable framed accuracy evidence; its original subject/point
            restrictions are not rebound to another candidate.
        reference: Supplied framed target reference, never an implicit solve.
        prior_work: Immutable supplied work facts, distinct from future execution.
    """

    method: Method
    plan: Plan | None
    reason: str | None = None
    estimate: object = None
    allocation: object = None
    facts: tuple = ()
    reference: object = None
    prior_work: tuple = ()

    def __post_init__(self):
        """Bind a candidate to its actual Plan, allocation and evidence, or an explicit
        applicability failure.
        """
        from nwqlib.backends.assessment import PlanEstimate
        from nwqlib.backends.profiles import Allocation
        from nwqlib.evidence import Fact
        from nwqlib.evidence.error_model import FramedFact, TargetReference
        from nwqlib.resources import WorkloadEstimate

        if not isinstance(self.method, Method):
            raise TypeError("comparison row requires its configured Method")
        if self.plan is None:
            if (
                not isinstance(self.reason, str)
                or not self.reason.strip()
                or self.estimate is not None
            ):
                raise ValueError("blocked row requires its reason and no fabricated estimate")
        elif (
            not isinstance(self.plan, Plan)
            or self.plan.method.content_id != self.method.content_id
            or self.reason is not None
        ):
            raise ValueError("comparison row must preserve its selected Method/Plan association")
        if self.allocation is not None and not isinstance(self.allocation, Allocation):
            raise TypeError("comparison row requires a supplied Allocation")
        if isinstance(self.estimate, PlanEstimate):
            if (
                self.estimate.plan_id != self.plan.content_id
                or self.estimate.allocation != self.allocation
                or self.estimate.resources.construction_id != self.plan._construction_id
            ):
                raise ValueError("comparison forecast differs from its Plan or Allocation")
        elif isinstance(self.estimate, WorkloadEstimate):
            if self.estimate.construction_id != self.plan._construction_id:
                raise ValueError("comparison resources differ from the selected construction")
        elif self.estimate is not None:
            raise TypeError("comparison estimate requires actual selected resource evidence")
        if type(self.facts) is not tuple or any(not isinstance(f, FramedFact) for f in self.facts):
            raise TypeError("comparison facts must be immutable framed evidence")
        if self.reference is not None and not isinstance(self.reference, TargetReference):
            raise TypeError("comparison reference must be a framed TargetReference")
        if type(self.prior_work) is not tuple or any(
            not isinstance(f, Fact) for f in self.prior_work
        ):
            raise TypeError("comparison prior work must preserve immutable supplied occurrences")


@dataclass(frozen=True)
class Comparison:
    """Explicit candidates in their original order; no implicit best-method rule.

    Attributes:
        problem: Original scientific Problem shared by every selected row.
        rows: Finite immutable ComparisonRow tuple in caller order. Selecting
            an index returns that existing row without ranking, planning or
            executing it. Original Plan/evidence associations remain intact.
    """

    problem: object
    rows: tuple[ComparisonRow, ...]

    def __post_init__(self):
        """Require one shared Problem, so rows compare methods on the same scientific target."""
        from nwqlib.problems.records import ProblemRecord

        if (
            not isinstance(self.problem, ProblemRecord)
            or type(self.rows) is not tuple
            or not self.rows
        ):
            raise TypeError("comparison requires one original Problem and finite immutable rows")
        if any(
            not isinstance(row, ComparisonRow)
            or row.plan is not None
            and row.plan.problem.content_id != self.problem.content_id
            for row in self.rows
        ):
            raise ValueError("comparison rows must address the same original scientific Problem")

    def select(self, index):
        """Return any original row; execution separately rejects a blocked row."""
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise IndexError("comparison selection requires an original nonnegative row index")
        return self.rows[index]


def _check_shots(shots):
    """Refuse shots that are not a positive Python int or None, naming the value and its type."""
    if shots is not None and (type(shots) is not int or shots < 1):
        raise ValueError(f"shots must be a positive int or None for exact readout, got {shots!r}")


def plan(
    problem, *, method, output=None, accuracy=None, execution="quantum", shots=None, seed=None
):
    """Select science once. Native execution belongs to prepare and submit.

    The returned Plan fixes the Problem, configured Method, output, readout and the
    RNG position after the Method's planning draws. Later steps execute that
    selection and never reselect it. A different choice requires a new Plan. When
    ``accuracy`` is given, the Method's ``sampling_shots`` connector chooses the
    shots and the Plan records the criterion that chose them.
    """
    if not isinstance(method, Method):
        raise TypeError("method must be an immutable configured Method")
    if execution not in {"quantum", "classical"}:
        raise ValueError("execution must be 'quantum' or 'classical'")
    _check_shots(shots)
    if output is None:
        output = problem.default_output()
    if accuracy is not None:
        connector = getattr(method, "sampling_shots", None)
        if not callable(connector):
            raise ApplicabilityError(
                f"{type(method).__name__} has no selected accuracy connector; "
                "use its explicit scientific controls. Method authors can supply "
                "sampling_shots(self, problem, *, output, accuracy, execution, shots); "
                "see docs/algorithm_protocol.md"
            )
        shots = connector(
            problem, output=output, accuracy=accuracy, execution=execution, shots=shots
        )
    return _plan_with_streams(
        problem, method, output, execution, shots, RandomStreams(seed), accuracy=accuracy
    )


def _plan_with_streams(problem, method, output, execution, shots, streams, *, accuracy=None):
    """Call ``Method.plan`` and check that the Plan keeps what the caller supplied.

    The Plan must hold the same Problem, Method and output objects, the RNG
    snapshot taken after planning draws and the accuracy criterion that selected
    its acquisition. A Plan that substituted any of them would attach later data
    to a question the caller did not ask.
    """
    # An output unit that conflicts with the Problem's unit would relabel an
    # unconverted value. Reject it before the Method selects anything. The
    # check belongs to the library's OutputRecord kinds. An output record that
    # a Method defines under the algorithm protocol states its own frame.
    from nwqlib.problems.records import OutputRecord
    if isinstance(output, OutputRecord):
        output.frame(problem)
    selected = method.plan(
        problem,
        output=output,
        execution=execution,
        shots=shots,
        rng=streams,
        **({"accuracy": accuracy} if accuracy is not None else {}),
    )
    if not isinstance(selected, Plan):
        raise TypeError("Method.plan must return its selected Plan")
    if (
        selected.problem is not problem
        or selected.method is not method
        or selected.output is not output
    ):
        raise ValueError(
            "Method.plan must preserve the original Problem, configured Method and selected output"
        )
    if selected.randomness != streams.snapshot():
        raise ValueError("Plan randomness must snapshot the actual streams after planning draws")
    if selected.selection_accuracy != accuracy:
        raise ValueError("Plan must preserve the criterion that actually selected its acquisition")
    return selected


def prepare(plan, *, backend=None, limits=None, progress=None, directory=None, settings="first"):
    """Admit the chosen execution and prepare its first setting or every static setting, without submitting.

    A new Run is created for the Plan (or a selected ComparisonRow, whose forecast
    and Allocation travel with it). A static Method admits every circuit, shot and
    preparation of the whole Plan against the Run's limits before it prepares
    anything. The returned Prepared owns the live Run, which the caller closes. If
    preparation fails, the Run is closed and its durable folder, if any, is named
    in the error.

    Args:
        plan (Plan | ComparisonRow): Selected Plan, or a selected comparison row.
        backend (object | None): Execution connection. None selects local Aer for
            quantum execution and host kernels for classical execution.
        limits (ExecutionLimits | None): Cumulative Run limits, or None for the defaults.
        progress (object | None): Progress callback, False to disable, or None for
            the notebook display.
        directory (str | Path | None): New durable Run folder.
        settings (str): ``"first"`` prepares the setting that the Method
            submits first, the first experiment of a static Plan, and ``submit``
            prepares each later one when it reaches it. ``"all"``
            prepares every static setting now, in Plan order, so
            ``Prepared.circuits``, ``Prepared.setting_names``, inspection and
            compilation by index cover the whole Plan. Each setting is
            prepared and charged once, and a durable Run journals it. ``submit``
            uses these preparations. The admission above already covers them, so
            ``"all"`` adds no limit and prepares nothing that a completed submit
            would not. In this version ``"all"`` needs local Aer, local NWQ-Sim
            or classical host execution. Another backend is refused before a Run
            exists, because its preparation can be a remote compilation with a
            provider charge. This restriction lasts until the cost disclosure
            and the handling of pending remote preparations are decided
            (docs/ROADMAP.md, "Preparing every setting on remote backends").
            A Method whose later settings depend on earlier
            outcomes (RWPE, Lanczos with SensitivitySampling, ADAPT) refuses
            ``"all"`` before a Run exists and says what it can prepare instead
            (``Method.prepare_all_refusal``).

    Returns:
        prepared (nwqlib._prepared_execution.Prepared): The preparations and their
            live Run. Nothing has been submitted.
    """
    from nwqlib.backends.assessment import PlanEstimate

    if settings not in ("first", "all"):
        raise ValueError("settings must be 'first' or 'all'")
    if settings == "all" and backend is not None:
        from nwqlib._prepared_execution import LOCAL_SIMULATORS

        kind = getattr(backend, "kind", type(backend).__name__)
        if kind not in LOCAL_SIMULATORS:
            raise ValueError(
                "prepare(..., settings='all') needs local Aer, local NWQ-Sim or classical host "
                f"execution in this version, and backend {kind!r} is none of them. On other "
                "backends a preparation can be a remote compilation with a provider charge. The "
                "selected construction is the same on every backend, so for counts or exact "
                "readout, prepare(plan, settings='all') with the default Aer backend shows every "
                "circuit within ExecutionLimits.max_simulation_qubits for inspection or "
                "compilation. Only the backend's own compilation differs"
            )
    forecast = allocation = None
    if isinstance(plan, ComparisonRow):
        # ComparisonRow construction already bound its Method, Plan and estimate.
        row = plan
        if row.plan is None:
            raise ApplicabilityError(row.reason)
        plan, allocation = row.plan, row.allocation
        forecast = row.estimate if isinstance(row.estimate, PlanEstimate) else None
    return _prepare_plan(
        plan,
        backend=backend,
        limits=limits,
        progress=progress,
        directory=directory,
        forecast=forecast,
        allocation=allocation,
        settings=settings,
    )


def _prepare_plan(
    plan, *, backend, limits, progress, directory=None, forecast=None, allocation=None,
    settings="first",
):
    """The same preparation owner serves a new or explicitly repeated run.

    A repeated ``submit`` prepares the first setting of its new Run, and
    ``execute`` prepares the rest as it reaches them.
    """
    from nwqlib._prepared_execution import Prepared, Run

    if not isinstance(plan, Plan):
        raise TypeError("prepare requires an already selected Plan")
    if plan.requirements:
        raise ApplicabilityError("; ".join(plan.requirements))
    # A refusal of "all" comes before the Run, so it leaves no Run folder.
    refusal = plan.method.prepare_all_refusal() if settings == "all" else None
    if refusal is not None:
        raise ValueError(refusal)
    run = Run(
        plan,
        backend=backend,
        limits=limits,
        progress=progress,
        directory=directory,
        forecast=forecast,
        allocation=allocation,
    )
    try:
        run.progress("input", 1, 1)
        # Only an explicit "all" reaches the hook, so a Method that overrides
        # prepare without the settings keyword keeps its default behavior.
        plan.method.prepare(plan, run=run, **({"settings": "all"} if settings == "all" else {}))
    except BaseException as error:
        if run.directory is not None:
            error.add_note(f"Original preparation data is saved at {run.directory}")
        run.close()
        raise
    return Prepared(run)


def submit(prepared):
    """Start one explicit run. Later continuation uses that Run's resume method.

    The first call starts the Prepared's own Run and returns it; that Run stays
    the caller's through ``prepared.run``, also when ``submit`` raises, so enter
    ``with prepared.run as run:`` before submitting to close it. A second call on
    the same Prepared starts a new Run with the same backend, current limits and
    the Plan's original randomness. A started Run's exposure cannot be reset,
    so repeating work needs a separate owner. The new Run draws the same seeds, so
    its new identity does not establish that its samples are independent of the
    first Run's.
    """
    from nwqlib._prepared_execution import Prepared

    if not isinstance(prepared, Prepared):
        raise TypeError("submit requires the opaque object returned by prepare")
    run = prepared.run
    internal_owner = run._state["started"]
    if internal_owner:
        prepared = _prepare_plan(
            run.plan,
            backend=run.backend,
            limits=run.limits,
            progress=run._state["progress"],
            forecast=run.forecast,
            allocation=run.allocation,
        )
        run = prepared.run
    run._state["started"] = True
    try:
        return run.resume()
    except BaseException as error:
        # A repeated submit created a new owner that the caller never received.
        # The original Prepared.run remains theirs; do not close or cancel it.
        if internal_owner:
            if run.directory is not None:
                error.add_note(f"Original execution data is saved at {run.directory}")
            run.close()
        raise


def solve(
    problem_or_plan,
    *,
    method=None,
    output=None,
    accuracy=None,
    backend=None,
    execution=None,
    shots=None,
    seed=None,
    limits=None,
    progress=None,
):
    """Plan when needed, execute the selected experiment, and return its Result.

    Args:
        problem_or_plan (object): Original Problem, selected Plan, or selected ComparisonRow.
            A Plan/row already fixes science and acquisition; method, output,
            accuracy, execution, shots and seed cannot override that selection.
        method (Method | None): Configured Method, required for a raw Problem.
        output (object | None): Scientific output requested from a raw Problem; its default
            output is used when omitted.
        accuracy (object | None): Optional criterion used by a Method's supported selection
            connector. It is not an assertion of achieved total accuracy.
        backend (object | None): Explicit compatible execution connection; omitted selects
            the local default appropriate to the selected execution.
        execution (str | None): For a raw Problem, quantum by default or explicit classical.
            Classical execution evaluates that Method's selected numerical model.
        shots (int | None): Positive requested shots for supported sampled readout, or None
            for exact readout. Method-specific connectors can select acquisition.
        seed (int | None): Nonnegative seed for a raw Problem's planning/acquisition streams.
            A seed does not establish statistical independence.
        limits (object | None): Optional cumulative Run limits for preparation and execution.
        progress (object | None): Optional progress callback passed to the Run;
            called as callback(stage, done, total). False disables it, and None
            permits the default notebook display.

    Returns:
        result (nwqlib.core.analysis.Result): The Method's concrete scientific Result
            with its Plan and actual RunData. This call prepares and acquires
            data and waits for completion; it does not automatically run reference
            verification. Use prepare/submit for explicit lifecycle control and
            durable continuation.
    """
    if isinstance(problem_or_plan, (Plan, ComparisonRow)):
        overrides = [name for name, value in (
            ("method", method), ("output", output), ("accuracy", accuracy),
            ("execution", execution), ("shots", shots), ("seed", seed),
        ) if value is not None]
        if overrides:
            raise ValueError(
                f"a selected Plan cannot be overridden with {', '.join(overrides)}; "
                f"supply {', '.join(overrides)} to a new plan(...) to change science or acquisition"
            )
        selected = problem_or_plan
    else:
        if method is None:
            raise TypeError("solve requires method= when given a raw Problem")
        selected = plan(
            problem_or_plan,
            method=method,
            output=output,
            accuracy=accuracy,
            execution="quantum" if execution is None else execution,
            shots=shots,
            seed=seed,
        )
    prepared = prepare(selected, backend=backend, limits=limits, progress=progress)
    with prepared.run:
        return submit(prepared).wait()


def estimate(plan, *, context=None, profile=None, allocation=None, assessed_at=None,
             facts=(), reference=None, max_assessments=4096):
    """Read selected laws and supplied models at an explicit assessment time.

    Framed facts/reference require a profile assessment and its selected
    accuracy criterion. max_assessments bounds forecast rows, not logical folds.
    """
    if not isinstance(plan, Plan):
        raise TypeError("estimate requires a selected Plan")
    if type(facts) is not tuple:
        raise TypeError("forecast facts must be an immutable tuple")
    if profile is None and allocation is None:
        if facts or reference is not None or assessed_at is not None:
            raise ValueError("assessment time/evidence require a profile or allocation; logical folding has no assessment")
        from nwqlib.resources import estimate as fold

        return fold(plan.construction, context=context)
    from nwqlib.backends.assessment import estimate_plan

    return estimate_plan(plan, context=context, profile=profile, allocation=allocation,
                         assessed_at=assessed_at, facts=facts, reference=reference,
                         max_assessments=max_assessments)


def compare(
    problem,
    *,
    methods,
    output=None,
    accuracy=None,
    execution="quantum",
    shots=None,
    seed=None,
    profile=None,
    allocation=None,
    context=None,
    assessed_at=None,
    facts=(),
    reference=None,
    max_assessments=4096,
):
    """Plan explicit candidates without acquisition, reference solves or ranking.

    Each candidate gets its own child of one ``SeedSequence(seed)``, so no
    candidate's planning draws shift another's, and with an integer ``seed`` every
    row is reproducible. A candidate outside its Method's domain becomes a row with
    a reason and no Plan. Rows keep the caller's order.
    """
    if not isinstance(methods, (tuple, list)) or not methods:
        raise TypeError("methods must be a nonempty finite list or tuple")
    if type(facts) is not tuple:
        raise TypeError("forecast facts must be an immutable tuple")
    if profile is None and (facts or reference is not None):
        raise ValueError("supplied accuracy evidence requires a profile assessment")
    if profile is None and allocation is None and assessed_at is not None:
        raise ValueError("assessment time requires a profile or allocation")
    if profile is not None or allocation is not None:
        from nwqlib.backends.profiles import Allocation, DeviceProfile

        if profile is not None and (
            not isinstance(profile, DeviceProfile) or not isinstance(allocation, Allocation)
        ):
            raise ValueError(
                "device predictions require a supplied DeviceProfile and explicit Allocation"
            )
        if allocation is not None and not isinstance(allocation, Allocation):
            raise TypeError("allocation must be a supplied Allocation")
        from nwqlib.backends.assessment import _assessment_time, _forecast_rows

        at = _assessment_time(assessed_at)
        if profile is not None:
            _forecast_rows(0, profile, max_assessments)  # Validate the cap before candidate planning.
    import numpy as np

    _check_shots(shots)
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    children = np.random.SeedSequence(seed).spawn(len(methods))
    rows = []
    for method, child in zip(methods, children, strict=True):
        try:
            if accuracy is not None:
                connector = getattr(method, "sampling_shots", None)
                if not callable(connector):
                    raise ApplicabilityError(f"{type(method).__name__} has no selected accuracy connector; "
                        "use its explicit scientific controls. Method authors can supply "
                        "sampling_shots(self, problem, *, output, accuracy, execution, shots); "
                        "see docs/algorithm_protocol.md")
                selected_shots = connector(
                    problem,
                    output=output or problem.default_output(),
                    accuracy=accuracy,
                    execution=execution,
                    shots=shots,
                )
            else:
                selected_shots = shots
            selected = _plan_with_streams(
                problem,
                method,
                output or problem.default_output(),
                execution,
                selected_shots,
                RandomStreams.from_sequence(child),
                accuracy=accuracy,
            )
        except ApplicabilityError as error:
            rows.append(ComparisonRow(method, None, str(error), allocation=allocation))
        else:
            rows.append(ComparisonRow(method, selected, allocation=allocation,
                                      facts=facts, reference=reference))
    if profile is not None or allocation is not None:
        from dataclasses import replace
        from nwqlib.backends.assessment import (
            _forecast_rows,
            _profile_points,
            estimate_plan,
        )

        # Each row's points are enumerated once and passed to estimate_plan.
        evaluated = [None if profile is None or row.plan is None else _profile_points(row.plan) for row in rows]
        if profile is not None:
            points = sum(len(pair[0]) for pair in evaluated if pair is not None)
            _forecast_rows(points, profile, max_assessments)
        rows = [
            row
            if row.plan is None
            else replace(
                row,
                estimate=estimate_plan(
                    row.plan,
                    context=context,
                    profile=profile,
                    allocation=allocation,
                    assessed_at=at,
                    accuracy=accuracy,
                    facts=row.facts,
                    reference=row.reference,
                    max_assessments=max_assessments,
                    profile_points=pair,
                ),
            )
            for row, pair in zip(rows, evaluated, strict=True)
        ]
    else:
        from dataclasses import replace

        rows = [
            row if row.plan is None else replace(row, estimate=estimate(row.plan, context=context))
            for row in rows
        ]
    return Comparison(problem, tuple(rows))


def load_result(path, *, method=None):
    """Load a saved Result, with its published arrays as lazy read-only mappings.

    Loading checks the saved records and array headers but does not read the
    array values. An external Method must be supplied as ``method``. See
    ``saved_evidence.load_result``.
    """
    from nwqlib.saved_evidence import load_result as load

    return load(path, method=method)


def load_run(path, *, backend, method=None, progress=None):
    """Open a saved Run for inspection or explicit continuation.

    Args:
        path (str | Path): Existing Run directory, distinct from a standalone Result folder.
            Keep it available and writable for journal/controller transitions.
        backend (object): Execution connection matching the saved backend configuration.
            This does not authorize replacement of uncertain acquisitions.
        method (type | None): Explicit external Method class when required by its saved
            implementation. Built-in Methods are resolved by the library.
        progress (object | None): Progress callback of the reopened Run, False to disable,
            or None for the notebook display, as for ``prepare``.

    Returns:
        run (nwqlib._prepared_execution.Run): An open Run owning the directory's
            controller lock. Close it when done. Loading restores selected data
            without planning or acquiring again; resume/wait separately advance
            the original work.
    """
    from nwqlib._run_archive import load

    return load(path, backend=backend, method=method, progress=progress)


def methods():
    """Return the tuple of built-in Registration records without executing them.

    Read each record's source.name/source.version for a compact inventory;
    declarations do not establish scientific or backend qualification. The CLI
    algorithms/card/options commands provide formatted discovery and schemas.
    """
    from nwqlib.algorithms.registry import builtin_registrations

    return builtin_registrations()
