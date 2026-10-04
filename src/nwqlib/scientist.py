"""Natural scientific planning and execution over one selected Plan."""

from dataclasses import dataclass

from nwqlib.algorithms.protocol import ApplicabilityError, Method
from nwqlib.core.planning import Plan, RandomStreams


@dataclass(frozen=True)
class ComparisonRow:
    """One Method of a comparison, with its Plan and resource estimate.

    [`compare`][nwqlib.scientist.compare] and [`scan`][nwqlib.search.scan]
    build the rows, and `Comparison.select(index)` returns one. Pass a row to
    [`solve`][nwqlib.scientist.solve] or [`prepare`][nwqlib.scientist.prepare]
    to run its Plan, together with its device forecast and Allocation when it
    has them. A Method that cannot handle the Problem gives a row with a
    `reason` and no Plan, and running that row raises `ApplicabilityError`.
    The fields below are read-only. Reading a row evaluates no model and
    builds no circuit.

    Attributes:
        method: The configured Method. The row's Plan was made by this Method.
        plan: The Method's Plan, or `None` when the Method cannot handle the
            Problem.
        reason: Why the Method cannot handle the Problem, or `None` for a row
            with a Plan.
        estimate: The resource estimate of this Plan: a `WorkloadEstimate`,
            or a `PlanEstimate` when a device profile or Allocation was given,
            with device predictions when a profile was given. `None` for a
            row without a Plan.
        allocation: The Allocation given to `compare` or `scan`, kept with
            the forecast, or `None`.
        facts: Accuracy evidence (`FramedFact` records) supplied for this
            row. Each keeps the subject and point it was stated for and is
            not applied to another row.
        reference: The supplied target reference (`TargetReference`), or
            `None`. No reference value is computed.
        prior_work: Records (`Fact`) of work done earlier, as supplied. They
            are kept apart from the work of a later run.
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
    """The Plans and resource estimates of several Methods for one Problem.

    [`compare`][nwqlib.scientist.compare] returns it, with one row per Method
    in the order the Methods were given. `select(index)` returns a row to pass
    to [`solve`][nwqlib.scientist.solve] or
    [`prepare`][nwqlib.scientist.prepare]. A Comparison ranks nothing and
    names no best Method. [`scan`][nwqlib.search.scan] ranks rows by stated
    objectives. The fields below are read-only.

    Attributes:
        problem: The Problem shared by every row.
        rows: The [`ComparisonRow`][nwqlib.scientist.ComparisonRow] records,
            in the order the Methods were given. Selecting a row returns it
            unchanged, with its Plan, estimate and evidence, and does not
            rank, plan or run anything.
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
        """Return the row at `index`, unchanged.

        Any row can be selected, including one without a Plan. Running a row
        without a Plan raises `ApplicabilityError` with the row's `reason`.

        Args:
            index (int): Row position, from 0 to `len(rows) - 1`.

        Returns:
            row (ComparisonRow): The row at that position.

        Raises:
            IndexError: If `index` is not an integer in that range.
        """
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise IndexError("comparison selection requires an original nonnegative row index")
        return self.rows[index]


def _check_shots(shots):
    """Require a positive count for a sampled readout."""
    if shots is not None and shots < 1:
        raise ValueError(f"shots must be a positive int or None for exact readout, got {shots!r}")


def plan(
    problem, *, method, output=None, accuracy=None, execution="quantum", shots=None, seed=None
):
    """Plan how a Method computes the requested output of a Problem, without running it.

    The Method chooses its construction, circuits and readout once, and the
    returned [`Plan`][nwqlib.core.planning.Plan] fixes the Problem, Method,
    output, readout and the state of the random streams after the Method's
    planning draws. [`solve`][nwqlib.scientist.solve],
    [`prepare`][nwqlib.scientist.prepare] and
    [`submit`][nwqlib.scientist.submit] run this Plan and never plan again, so
    a different choice needs a new Plan. Planning computes what the Method
    needs to build its circuits and takes no measurement.
    [`estimate`][nwqlib.scientist.estimate] reads the Plan's resource counts.

    Args:
        problem (ProblemRecord): A Problem such as `Eigenproblem`.
        method (Method): Configured Method such as `Lanczos()`. Required.
        output (OutputRecord | None): Requested output, such as `Eigenvalue()`.
            Omitted, the Problem's default output is used.
        accuracy (Accuracy | None): Optional accuracy request. A Method that
            defines `sampling_shots` uses it to choose the shots, and the Plan
            records the request in `selection_accuracy`. Any other Method
            raises `ApplicabilityError`. It does not assert the accuracy
            achieved.
        execution (str): `"quantum"` (the default) or `"classical"`, which
            evaluates the Method's numerical model on the host.
        shots (int | None): Positive number of shots for sampled readout, or
            `None` (the default) for exact readout.
        seed (int | None): Nonnegative seed of the random streams. Omitted,
            NumPy draws fresh entropy, and the Plan records it in
            `randomness`.

    Returns:
        plan (Plan): The Plan. Its `content_id` identifies it, and every
            Result computed from it names it in `plan_id`.

    Raises:
        TypeError: If `method` is not a Method.
        ValueError: If `execution` is neither `"quantum"` nor `"classical"`,
            if a supplied `shots` count is below 1, if `seed` is
            negative or not an integer, or if `output` has a unit whose
            symbol differs from the one the Problem or the output kind
            defines.
        ApplicabilityError: If the Method cannot compute this output of this
            Problem, or if `accuracy` is given to a Method without
            `sampling_shots`.

    Examples:
        The eigenvalues of `[[2, 1], [1, 2]]` are 1 and 3:

        >>> from nwqlib import Eigenproblem, plan, solve
        >>> from nwqlib.algorithms import Lanczos
        >>> problem = Eigenproblem(A=[[2.0, 1.0], [1.0, 2.0]])
        >>> method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
        >>> selected = plan(problem, method=method, seed=7)
        >>> print(type(selected.output).__name__, selected.execution, selected.shots)
        Eigenvalue quantum None
        >>> print(round(solve(selected).eigenvalue, 10))
        1.0
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
    """Build the circuits of a Plan for a backend, without submitting them.

    `prepare` creates a new [`Run`][nwqlib._prepared_execution.Run] for the
    Plan and builds its first circuit, or every circuit with `settings="all"`.
    Pass the result to [`submit`][nwqlib.scientist.submit] to run it, or
    inspect the circuits first with `Prepared.circuits` and
    `Prepared.inspect_resources`. When the Plan's experiments are fixed in
    advance, rather than chosen from earlier outcomes, every circuit, shot and
    preparation of the whole Plan is checked against the Run's limits before
    anything is built. The returned Prepared holds the open Run, which the
    caller closes, for example with `with prepared.run as run:`. If
    preparation fails, the Run is closed and the error names its saved
    folder, if any.

    Args:
        plan (Plan | ComparisonRow): A Plan, or a row from `compare` or
            `scan`. A row's device forecast and Allocation stay with the Run
            and its Result.
        backend (object | None): Backend to run on. Omitted, local Aer runs
            quantum execution and the host runs classical execution.
        limits (ExecutionLimits | None): Limits on the Run's total
            preparations, circuits, shots and stored data. Omitted,
            the defaults of
            [`ExecutionLimits`][nwqlib.execution.ExecutionLimits] apply.
        progress (object | None): Progress callback, called as
            `callback(stage, done, total)`. `False` disables progress
            reports, and `None` permits the default notebook display.
        directory (str | os.PathLike | None): New folder in which the Run
            saves its state as it runs, so that
            [`load_run`][nwqlib.scientist.load_run] can reopen it.
        settings (str): `"first"` (the default) prepares the setting that the
            Method submits first, which is the first experiment when the
            experiments are fixed, and `submit` prepares each later one when
            it reaches it. `"all"` prepares every setting of fixed
            experiments now, in Plan order, so that `Prepared.circuits`,
            `Prepared.setting_names`, inspection and compilation by index
            cover the whole Plan. Each setting is prepared and counted once,
            a Run with a folder saves it, and `submit` uses these
            preparations. The limit check above already covers them, so
            `"all"` adds nothing to the limits and prepares nothing that a
            completed submit would not. `"all"` needs local Aer, local
            NWQ-Sim or classical host execution. Another backend is refused
            before a Run exists, because its preparation can be a remote
            compilation that the provider charges for. This restriction lasts
            until the cost disclosure and the handling of pending remote
            preparations are decided ([Preparing every setting on remote
            backends](../ROADMAP.md#preparing-every-setting-on-remote-backends)).
            A Method whose later settings depend on earlier outcomes (RWPE,
            Lanczos with `SensitivitySampling`, ADAPT) refuses `"all"` before
            a Run exists, and its message says what it can prepare instead.

    Returns:
        prepared (Prepared): The prepared circuits and their open Run
            (`prepared.run`). Nothing has been submitted.

    Raises:
        TypeError: If `plan` is neither a Plan nor a row of a comparison.
        ValueError: If `settings` is neither `"first"` nor `"all"`, or if
            `"all"` is requested on another backend or for a Method that
            refuses it.
        ApplicabilityError: If `plan` is a row without a Plan, or the Plan
            lists unmet requirements.

    Examples:
        See [`submit`][nwqlib.scientist.submit].
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
    """Start running prepared circuits and return their Run.

    The first call starts the Run of `prepared` and returns it when the run
    is complete or the Method has to wait for results from the backend.
    `run.result` holds the Result once the run is complete, and
    [`Run.wait`][nwqlib._prepared_execution.Run.wait] waits for it. The Run
    stays the caller's through `prepared.run`, also when `submit` raises, so
    enter `with prepared.run as run:` before submitting to close it. A
    second call on the same `prepared` starts a new Run with the same
    backend, the current limits and the Plan's original random state,
    because the work counted against a started Run cannot be reset. The new
    Run draws the same seeds, so its new `run_id` does not establish that
    its samples are independent of the first Run's.

    Args:
        prepared (Prepared): The return value of
            [`prepare`][nwqlib.scientist.prepare].

    Returns:
        run (Run): The started Run.

    Raises:
        TypeError: If `prepared` is not the return value of `prepare`.
        RunFailed: If the run fails or an attempt's outcome cannot be
            recovered.

    Examples:
        The state `|0>` gives `<Z> = 1` in every shot:

        >>> from nwqlib import Expectation, plan, prepare, submit
        >>> from nwqlib.algorithms import ExpectationMethod
        >>> problem = Expectation(state=[1.0, 0.0],
        ...                       observable=[[1.0, 0.0], [0.0, -1.0]])
        >>> selected = plan(problem, method=ExpectationMethod(), shots=64,
        ...                 seed=7)
        >>> prepared = prepare(selected)
        >>> print(prepared.setting_names)
        ('group_0',)
        >>> with prepared.run:
        ...     result = submit(prepared).wait()
        >>> print(result.value)
        1.0
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
    """Solve a Problem with a Method and return its Result.

    `solve` plans the computation (when given a Problem), prepares and runs
    the circuits or the classical model, waits for completion and returns
    the Method's Result. It does not run a reference check of the answer.
    Use `plan`, `prepare` and `submit` to control each step, or to save a
    run and continue it after an interruption.

    Args:
        problem_or_plan (object): A Problem such as `Eigenproblem`, a `Plan`
            from `plan`, or a `ComparisonRow` from `compare`. A Plan or row
            already fixes the method, output and measurement settings, so
            `method`, `output`, `accuracy`, `execution`, `shots` and `seed`
            must then be omitted.
        method (Method | None): Configured Method such as `Lanczos()`,
            required with a Problem.
        output (OutputRecord | None): Output requested from a Problem, such as
            `Eigenvalue()`. Omitted, the Problem's default output is used.
        accuracy (Accuracy | None): Optional accuracy request. A Method that
            defines `sampling_shots` uses it to choose the shots, and any
            other Method raises `ApplicabilityError`. It does not assert the
            accuracy achieved.
        backend (object | None): Backend to run on. Omitted, the local
            default for the execution mode is used.
        execution (str | None): `"quantum"` (the default for a Problem) or
            `"classical"`, which evaluates that Method's numerical model.
        shots (int | None): Positive number of shots for sampled readout, or
            `None` for exact readout. A Method setting can choose the
            measurement instead, for example `Lanczos(sampling=...)`.
        seed (int | None): Nonnegative seed of the random streams that
            planning and measurement of a Problem use. A seed does not
            establish statistical independence.
        limits (ExecutionLimits | None): Optional
            [`ExecutionLimits`][nwqlib.execution.ExecutionLimits] for the
            Run's preparation and execution.
        progress (object | None): Optional progress callback, called as
            `callback(stage, done, total)`. `False` disables progress
            reports, and `None` permits the default notebook display.

    Returns:
        result (nwqlib.core.analysis.Result): The Method's Result, with its
            Plan and run data attached. For example, a
            [`LanczosResult`][nwqlib.algorithms.lanczos.records.LanczosResult]
            holds the answer in `eigenvalue`.

    Raises:
        TypeError: If a Problem is given without `method`.
        ValueError: If a Plan or row is given with `method`, `output`,
            `accuracy`, `execution`, `shots` or `seed`. Pass them to a new
            `plan(...)` instead.

    Examples:
        The smallest eigenvalue of `[[1.5, -1], [-1, 0.5]]` is
        `(2 - sqrt(5)) / 2 = -0.1180339887...`. Lanczos with a full
        two-dimensional trial space recovers it from exact probabilities:

        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import Lanczos
        >>> problem = Eigenproblem(A=[[1.5, -1], [-1, 0.5]])
        >>> method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
        >>> result = solve(problem, method=method, seed=7)
        >>> print(round(result.eigenvalue, 10))
        -0.1180339887
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
    """Estimate the resources of a Plan, and optionally its run time on a device, without running it.

    Without a device profile or Allocation, `estimate` adds up the counting
    formulas of the Plan's construction (qubits, operations, shots and the
    other metrics) and returns a `WorkloadEstimate`. A metric without a
    formula is reported as unavailable, never as zero. With a profile, it
    also evaluates the profile's device models at each planned execution
    whose parameters are already fixed, and returns a `PlanEstimate`. With
    an Allocation and no profile, the `PlanEstimate` keeps the Allocation and
    makes no device prediction. Nothing is executed or compiled. The
    [Estimate resources](../resources.md) and [Check device fit and run
    time](../profiles.md) guides explain how to read both.

    Args:
        plan (Plan): The Plan to estimate.
        context (ResourceContext | None): Counting options, such as the gate
            basis (`ResourceContext(basis="cx")`) and the schedule of
            independent executions. Omitted, `ResourceContext()` applies.
        profile (DeviceProfile | None): Device models to evaluate. It needs
            an `allocation`.
        allocation (Allocation | None): The devices granted to the run. Given
            without a profile, it is kept with the estimate and no device
            prediction is made.
        assessed_at (datetime | None): Time-zone-aware time at which the
            device models are evaluated. Omitted, the current time is used.
            It needs a profile or Allocation.
        facts (tuple[FramedFact, ...]): Accuracy evidence to assess with the
            forecast. It needs a profile, an error model in the Plan and an
            accuracy request in the Plan (`plan(..., accuracy=...)`).
        reference (TargetReference | None): Target reference to assess with
            the forecast, with the same requirements as `facts`.
        max_assessments (int): Default `4096`. Largest number of forecast
            rows, one assessment per planned execution and one prediction
            per device model for each, checked before the first model
            evaluation. Adding up the construction's counts is not limited
            by it.

    Returns:
        estimate (WorkloadEstimate | PlanEstimate): `WorkloadEstimate`
            without a profile or Allocation, `PlanEstimate` with one. Read a
            count of a `WorkloadEstimate` with `estimate.quantity(metric)`.

    Raises:
        TypeError: If `plan` is not a Plan or `facts` is not a tuple.
        ValueError: If `assessed_at`, `facts` or `reference` is given without
            a profile or Allocation, if `facts` or `reference` is given
            without the profile, error model and accuracy request it needs,
            if a profile is given without an Allocation, if `assessed_at` has
            no time zone, or if the forecast needs more than
            `max_assessments` rows.

    Examples:
        The Plan requests 64 shots of a one-qubit circuit:

        >>> from nwqlib import Expectation, estimate, plan
        >>> from nwqlib.algorithms import ExpectationMethod
        >>> problem = Expectation(state=[1.0, 0.0],
        ...                       observable=[[1.0, 0.0], [0.0, -1.0]])
        >>> selected = plan(problem, method=ExpectationMethod(), shots=64,
        ...                 seed=7)
        >>> workload = estimate(selected)
        >>> print(workload.quantity("shots").fact.value.numerator)
        64
        >>> width = workload.quantity("logical_width",
        ...                           location="logical_device")
        >>> print(width.fact.value.numerator)
        1
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
    """Plan one Problem with several Methods and estimate the resources of each Plan.

    `compare` returns a [`Comparison`][nwqlib.scientist.Comparison] with one
    row per Method, in the order given. Each row holds the Method's Plan and
    its resource estimate, as [`estimate`][nwqlib.scientist.estimate] gives
    it. A Method that cannot handle the Problem gives a row with a `reason`
    and no Plan instead of an error. Nothing is measured, no reference value
    is computed and the rows are not ranked. Pass a row to
    [`solve`][nwqlib.scientist.solve] or [`prepare`][nwqlib.scientist.prepare]
    to run it, or rank the rows with [`scan`][nwqlib.search.scan]. Each Method
    plans with its own child of one `numpy.random.SeedSequence(seed)`, so no
    Method's planning draws shift another's, and with an integer `seed` every
    row is reproducible. The [Plan, compare and solve](../scientist.md) guide
    shows a comparison.

    Args:
        problem (ProblemRecord): The Problem every Method plans.
        methods (list | tuple): Configured Methods, at least one.
        output (OutputRecord | None): Requested output. Omitted, the
            Problem's default output is used.
        accuracy (Accuracy | None): Optional accuracy request, as for
            [`plan`][nwqlib.scientist.plan]. A Method without
            `sampling_shots` then gives a row with a reason.
        execution (str): `"quantum"` (the default) or `"classical"`.
        shots (int | None): Positive number of shots for sampled readout, or
            `None` (the default) for exact readout.
        seed (int | None): Nonnegative seed shared by the rows, as described
            above. Omitted, NumPy draws fresh entropy.
        profile (DeviceProfile | None): Device models to evaluate for every
            row, as for `estimate`. It needs an `allocation`.
        allocation (Allocation | None): The devices granted to a run, kept
            with every row.
        context (ResourceContext | None): Counting options for the resource
            estimates. Omitted, `ResourceContext()` applies.
        assessed_at (datetime | None): Time-zone-aware time of the device
            forecasts. It needs a profile or Allocation.
        facts (tuple[FramedFact, ...]): Accuracy evidence assessed with the
            forecast of every row and kept with it. It needs a profile and
            `accuracy`, and each Plan needs an error model.
        reference (TargetReference | None): Target reference, with the same
            use and requirements as `facts`.
        max_assessments (int): Default `4096`. Largest number of forecast
            rows, as for `estimate`, summed over the rows of the comparison
            and checked before the first model evaluation.

    Returns:
        comparison (Comparison): One row per Method, in the order given.

    Raises:
        TypeError: If `methods` is not a nonempty list or tuple, or `facts`
            is not a tuple.
        ValueError: If `facts`, `reference` or `assessed_at` is given without
            what it needs, if a profile is given without an Allocation, if
            `shots` or `seed` is invalid, or if the forecast needs more than
            `max_assessments` rows.

    Examples:
        For `Z + I`, Lanczos from the state `|+>` with one Krylov vector
        projects onto that state, whose energy is 1. FixedGCIM with the
        trial states `|+>` and `|+i>` spans the whole space, whose
        eigenvalues are 0 and 2:

        >>> from nwqlib import Eigenproblem, compare, solve
        >>> from nwqlib.algorithms import FixedGCIM, Lanczos
        >>> problem = Eigenproblem(A=[[2.0, 0.0], [0.0, 0.0]])
        >>> plus, plus_i = [1.0, 1.0], [1.0, 1j]
        >>> comparison = compare(problem, methods=(
        ...     Lanczos(initial_state=plus, krylov_dimension=1),
        ...     FixedGCIM(basis=(plus, plus_i)),
        ... ), seed=7)
        >>> for row in comparison.rows:
        ...     print(row.method.descriptor.method, row.reason)
        chebyshev_lanczos None
        fixed_gcim None
        >>> for index in range(2):
        ...     result = solve(comparison.select(index))
        ...     print(round(result.eigenvalue, 10))
        1.0
        0.0
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
    """Reopen a Result saved with `Result.save`, with its Plan and run data.

    The loaded Result has the same fields as the saved one, and
    `Result.analyze` can recompute it from the saved data with other
    settings, without new measurements. Loading plans nothing, runs nothing
    and computes no reference. It checks the saved records, the content hash
    of the Plan and the header of every saved array, and it reads the array
    values only when they are used, as read-only memory maps, so keep the
    folder available while using the Result. The [Save, load and reanalyze
    results](../saved_evidence.md) guide describes the saved folder.

    Args:
        path (str | os.PathLike): Folder written by `Result.save`.
        method (type | Method | None): The Method class, or an instance of
            it, for a Method that is not built into NWQLib. Built-in Methods
            are found from the saved record.

    Returns:
        result (Result): The saved Result, of the Method's own Result type,
            with its Plan and run data attached.

    Raises:
        ValueError: If the saved records fail their checks, or a Method
            outside NWQLib is saved and `method` is not given or differs
            from it.

    Examples:
        >>> import os, tempfile
        >>> from nwqlib import Expectation, load_result, solve
        >>> from nwqlib.algorithms import ExpectationMethod
        >>> problem = Expectation(state=[1.0, 0.0],
        ...                       observable=[[1.0, 0.0], [0.0, -1.0]])
        >>> result = solve(problem, method=ExpectationMethod(), seed=7)
        >>> path = result.save(os.path.join(tempfile.mkdtemp(), "result"))
        >>> print(load_result(path).value)
        1.0
    """
    from nwqlib.saved_evidence import load_result as load

    return load(path, method=method)


def load_run(path, *, backend, method=None, progress=None):
    """Reopen a saved Run to inspect it or continue it.

    `load_run` opens the folder of a Run, either the `directory` given to
    `prepare` or a copy written by `Run.save`, and returns the open Run with
    its Plan, limits, counted work, random state and the locators of its
    pending jobs. Loading plans nothing, contacts no backend and runs
    nothing. `run.resume()` or `run.wait()` then continues the same work, and
    a completed Run returns its saved Result. The Run takes an exclusive lock
    on the folder until it is closed. The [Continue an interrupted
    run](../run_archives.md) guide describes saving and reopening.

    Args:
        path (str | os.PathLike): A Run folder. A folder written by
            `Result.save` is not a Run folder. Keep it available and writable
            while the Run is open, because the Run records its progress
            there.
        backend (object): Backend with the configuration that the Run was
            saved with. Reopening never submits a replacement for an attempt
            whose outcome is unknown.
        method (type | Method | None): The Method class, or an instance of
            it, for a Method that is not built into NWQLib. Built-in Methods
            are found from the saved record.
        progress (object | None): Progress callback of the reopened Run,
            `False` to disable it, or `None` for the notebook display, as for
            [`prepare`][nwqlib.scientist.prepare].

    Returns:
        run (Run): The open Run. Close it when done, for example with
            `with load_run(path, backend=backend) as run:`.

    Raises:
        ValueError: If the saved records fail their checks, or a Method
            outside NWQLib is saved and `method` is not given or differs from
            it.
        BlockingIOError: If another open Run holds the folder. Close that
            Run, for example `prepared.run`, first.

    Examples:
        >>> import os, tempfile
        >>> from nwqlib import Expectation, load_run, plan, prepare, submit
        >>> from nwqlib.algorithms import ExpectationMethod
        >>> from nwqlib.backends import AerBackend
        >>> problem = Expectation(state=[1.0, 0.0],
        ...                       observable=[[1.0, 0.0], [0.0, -1.0]])
        >>> selected = plan(problem, method=ExpectationMethod(), shots=64,
        ...                 seed=7)
        >>> folder = os.path.join(tempfile.mkdtemp(), "run")
        >>> prepared = prepare(selected, directory=folder)
        >>> with prepared.run:
        ...     result = submit(prepared).wait()
        >>> with load_run(folder, backend=AerBackend()) as run:
        ...     print(run.wait().value)
        1.0
    """
    from nwqlib._run_archive import load

    return load(path, backend=backend, method=method, progress=progress)


def methods():
    """List the built-in Methods.

    Each entry is a registration record. Its `source.name` and
    `source.version` name the Method. A registration declares what the Method
    is for. It does not establish that the Method or a backend has been
    validated for a problem. Nothing is run. The command-line `algorithms`,
    `card` and `options` commands print the same information and each
    Method's settings ([Use the command line](../cli.md)).

    Returns:
        registrations (tuple[Registration, ...]): One record per built-in
            Method.

    Examples:
        >>> from nwqlib import methods
        >>> print([entry.source.name for entry in methods()])
        ['adapt_gcim', 'chebyshev_lanczos', 'finite_pauli_expectation', 'fixed_gcim', 'lchs', 'qcels', 'qhd', 'qls', 'rfe', 'rwpe', 'spe']
    """
    from nwqlib.algorithms.registry import builtin_registrations

    return builtin_registrations()
