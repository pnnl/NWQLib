"""The Method base class and its descriptor."""

from dataclasses import dataclass
from typing import ClassVar

from nwqlib._limits import ResourceLimitError  # noqa: F401 - public as nwqlib.algorithms.ResourceLimitError
from nwqlib.core.records import Record, Source, Text


@dataclass(frozen=True)
class RowTemplate:
    """Points of a static Plan whose circuits differ only in rotation angles (`Method.row_templates`).

    A point is an experiment, or one range-axis point of an experiment with
    axes (`nwqlib.core.planning.StaticPoint`). The points of one template must also share one readout, their
    `ObservationSpec` with its shots included, because they run circuits
    compiled with the readout of the template's first prepared point.

    Attributes:
        key: Identifier of the shared gate sequence and readout, equal for
            the same sequence and readout each time the Method is asked about
            the same Plan. Points with different readouts need different
            keys.
        block_id: Content hash of the selected block whose gate the template
            replaces in each point's circuit.
        circuit: That block's circuit with one Qiskit parameter per
            continuous angle, on the block's ports in signature order.
        parameters: The circuit's parameters, in the order of each row's
            values.
        rows: ``(point key, values)`` per point, with the `StaticPoint.key`
            the Method received. Binding the values to `parameters` gives the
            point's own block circuit.
    """

    key: str
    block_id: str
    circuit: object
    parameters: tuple
    rows: tuple

    # A class attribute, not a field: an ordinary template has no stated size.
    _retained_bytes = None


class _DeferredRowTemplate(RowTemplate):
    """A `RowTemplate` that builds its circuit when ``circuit`` is read, and states the size to compare first.

    `Method.row_templates` receives no Run and returns before the Run can compare
    anything, so a Method that built every template there would allocate them all
    before any comparison with what the Run holds. A deferred template carries
    ``retained_bytes`` instead, a declared allowance for one built template, not a
    measurement. ``prepare_template_rows`` compares it, reads ``circuit`` once for
    each new template and releases the result after compiling. The ``circuit``
    property does not cache, so that a template no longer in use keeps no circuit,
    and each read builds again.

    ``dataclasses.replace(template, circuit=...)`` gives an ordinary template with
    that circuit, because a circuit given to the constructor is used as it is and
    the stated size belongs only to a build. Replacing any other field reads
    ``circuit``, so it builds. ``repr``, equality and hashing build nothing.
    """

    def __init__(self, *, key, block_id, parameters, rows, circuit=None, build=None, retained_bytes=None):
        for name, value in (("key", key), ("block_id", block_id), ("parameters", parameters), ("rows", rows),
                            ("_given", circuit), ("_build", build), ("_retained_bytes", retained_bytes)):
            object.__setattr__(self, name, value)

    @property
    def circuit(self):
        return self._build() if self._given is None else self._given

    def __repr__(self):
        return f"_DeferredRowTemplate(key={self.key!r}, rows={len(self.rows)}, retained_bytes={self._retained_bytes})"

    def __eq__(self, other):
        return self is other

    __hash__ = object.__hash__


class AlgorithmDescriptor(Record):
    """What a Method declares about itself: name, version, the problems and outputs it handles, its coverage, limitations and references.

    Build it with keyword arguments, for example
    `AlgorithmDescriptor(method="hadamard_expectation", version="1", problem_families=("expectation",))`,
    and set it as the class attribute `descriptor` of a
    [`Method`][nwqlib.algorithms.protocol.Method] subclass. `method` and `version`
    are required. The descriptor states scope and references. Whether a Method
    applies to a given problem is decided by its `plan`, and a descriptor neither
    proves that nor qualifies a backend.

    Attributes:
        method: Required. Registered method name, such as `"lchs"`.
        version: Required. Method version. The pair (`method`, `version`)
            identifies the implementation.
        problem_families: Default `()`. Problem kinds the Method declares, such as
            `"linear_dynamics"`.
        output_families: Default `()`. Output kinds it declares, such as
            `"solution"`.
        access_families: Default `()`. Input representations it declares, such as
            `"dense"`.
        resource_coverage: Default `()`. What its resource estimates cover.
        evidence_coverage: Default `()`. What its error evidence covers.
        limitations: Default `()`. Known limitations stated to users.
        references: Default `()`. Sources of the implemented method, such as its
            paper.
        maintenance: Default `"experimental research method"`. Maintainer or
            status label.
    """

    method: Text
    version: Text
    problem_families: tuple[Text, ...] = ()
    output_families: tuple[Text, ...] = ()
    access_families: tuple[Text, ...] = ()
    resource_coverage: tuple[Text, ...] = ()
    evidence_coverage: tuple[Text, ...] = ()
    limitations: tuple[Text, ...] = ()
    references: tuple[Source, ...] = ()
    maintenance: Text = "experimental research method"

    @property
    def source(self):
        """The Source that names this method and version.

        Registrations and saved results use it to identify the implementation.
        """
        return Source(name=self.method, version=self.version,
                      domain="selected scientific method", reference=self.method)


class Method(Record):
    """Base class of every method: an immutable configuration that plans, runs and analyzes one kind of problem.

    Subclass it to add a method. Declare the class attribute
    `descriptor: ClassVar[AlgorithmDescriptor]` and the method's configuration
    fields, and implement `plan` and `analyze`. Users pass an instance as
    `method=` to `solve`, `plan` or `compare`. A Method is kept separate from the
    Problem it solves, so one problem can be approached with different initial
    states, trial spaces, degrees or estimators and compared on equal terms. A
    Method is an immutable Record, so its content hash enters every Plan it makes,
    and changing a field gives a different Plan. Running, preparation,
    submission, saving and reports call the hooks below and never branch on the
    method's family, so a new Method needs no change elsewhere in NWQLib.
    [Add a Method](../algorithm_protocol.md) states every requirement,
    including the protected hooks of Plan and Result below, and
    [Run your own circuit](../own_circuit.md) shows the smallest Method.

    Hooks a Method implements or may override:

    - `plan`: choose and bind the Plan. Required.
    - `analyze`: turn existing run data into the Result. Required.
    - `prepare` and `execute`: defaults for a fixed set of circuits. A Method
      that chooses later circuits from earlier results overrides them.
    - `prepare_all_refusal`: why `settings="all"` cannot be served, asked before
      a Run exists.
    - `has_row_templates` and `row_templates`: groups of points of a static
      Plan whose circuits differ only in rotation angles, built once per group.
    - `query_catalog_reason`: why the Plan's experiments are templates of
      queries whose parameter values the iteration chooses, asked by the
      resource estimate of a whole Plan.
    - `error_model`: the known and unavailable error sources of a Plan.
    - `verify` and `recover_analysis`: explicit extra operations, unavailable by
      default.
    - `reduction_allowance(self, plan, point, *, observation, width, run) -> int | None`,
      optional, for a Method whose reductions of data while it is collected
      count against an explicit work limit of its own. It receives the
      Plan, the point's `Realization`, its resolved trajectory
      `ObservationSpec`, the Program's logical width and the Run. It may
      check the reduction's workspace, raising `ValueError` if the point
      exceeds the limit, and returns the remaining nonnegative work
      allowance in the registered reducers' work units, or None when the
      Method sets no work limit. It reads the work already used without
      changing it. The base Method supplies no implementation, and without
      it the reductions have no work limit.
    - `before_submit`: check data collection before its submission is recorded.
    - `validate_point`, `specialize_experiment` and `selected_kernels`: check and
      derive one resolved point without planning again.

    Attributes:
        descriptor: Class attribute that every subclass sets: the
            [`AlgorithmDescriptor`][nwqlib.algorithms.protocol.AlgorithmDescriptor].
    """

    descriptor: ClassVar[AlgorithmDescriptor]

    def plan(self, problem, *, output, execution, shots, rng, accuracy=None):
        """Choose the circuits and readout for a problem and return the bound Plan.

        `nwqlib.plan` calls it. The Plan must keep the original problem, this
        Method, the output, the execution mode, the requested accuracy and the
        state of each named random stream after the planning draws. `nwqlib.plan`
        rejects a Plan that replaced the problem, Method or output object, or
        whose random state or accuracy differs. `accuracy` is a requested
        criterion, and the Plan must not record it as an achieved error bound.
        Raise
        [`ApplicabilityError`][nwqlib.algorithms.protocol.ApplicabilityError] when this
        Method cannot solve the problem or output.

        Args:
            problem (Record): The problem.
            output (Record): The requested output.
            execution (str): The requested execution, such as `"quantum"` or
                `"classical"`.
            shots (int | None): Requested shots. `None` selects exact readout
                or the Method's default. Exact readout provides probabilities,
                Pauli expectation values, amplitudes or classically computed
                scalars without shot sampling. Numerical errors can remain.
            rng (RandomStreams): The Plan's named random streams. Draw from them, so
                that one seed reproduces the same choice.
            accuracy (Accuracy | None): Requested accuracy criterion. It is passed
                only when the caller gives one.

        Returns:
            plan (Plan): The bound Plan.
        """
        raise NotImplementedError("a Method must select its scientific experiment")

    def prepare(self, plan, *, run, settings="first"):
        """Prepare circuits of the Plan in the Run without submitting them.

        The default prepares the first setting, or every setting for
        `settings="all"`. A Method that chooses its next circuit from earlier results
        overrides it to prepare that circuit. NWQLib passes `settings` on every call,
        and passes `"all"` only after `prepare_all_refusal` returned `None`.

        Args:
            plan (Plan): The Plan.
            run (Run): The Run that holds the prepared circuits.
            settings (str): `"first"` or `"all"`.
        """
        from nwqlib._prepared_execution import prepare_static
        return prepare_static(plan, run=run, settings=settings)

    def prepare_all_refusal(self):
        """Return why `prepare(plan, settings="all")` cannot serve this Method, or None when it can.

        `nwqlib.prepare` asks before it creates a Run, so a refusal leaves no Run
        folder. A Method whose later settings depend on earlier outcomes overrides
        it to return its reason and what can be prepared instead. Otherwise the
        inherited preparation would prepare, and count against the Run's limits,
        every setting of the Plan, including ones the Method never submits. The
        default returns `None`.

        Returns:
            reason (str | None): The reason, or `None`.
        """
        return None

    def has_row_templates(self, plan):
        """Return whether `row_templates` can group points of `plan`.

        A Run on a backend that binds parameterized circuits asks this
        before it collects the remaining points of a static Plan for
        `row_templates`. When it returns False, the Run generates, prepares
        and runs the points one at a time, so a Plan with many range-axis
        points starts without listing them all first. The default returns
        whether the Method's class overrides `row_templates`. A Method
        overrides this hook to return False for its Plans whose points are
        never grouped.

        Args:
            plan (Plan): The Plan.

        Returns:
            groups (bool): Whether `row_templates` can return groups for `plan`.
        """
        return type(self).row_templates is not Method.row_templates

    def row_templates(self, plan, points):
        """Group static points whose circuits differ only in rotation angles, or return None.

        A point is an experiment of the Plan, or one range-axis point of an
        experiment whose batch has axes. The Run asks only on a backend
        that binds parameterized circuits (`supports_template_pubs`: IBM
        Runtime and Aer), when `has_row_templates` returns True, before it
        prepares the `points` of a static Plan. Each group's circuit is
        compiled once, and each point keeps its own values, preparation
        record and attempt, on which its shots are counted. A Method
        overrides it when its points share gates, and its groups must cover
        `points`, each by its key. The points of a group must share one readout
        (`ObservationSpec`, shots included), because they run circuits
        compiled with the readout of the group's first prepared point.
        Put points with different readouts in groups with different keys.

        Both backends group counts readouts only. IBM Runtime sends every
        grouped point that has not run yet in one Sampler job, which uses
        one shot count, so on IBM Runtime all groups of a Plan need the same
        shots. Aer asks only about the points with counts readouts and runs
        each grouped point as its own attempt with its own seed, so the
        groups' shots can differ. Aer prepares a point with any other
        readout on its own, and so the points of a group whose
        parameterized operations are not all Qiskit standard gates, such as
        a `HamiltonianGate` whose time is a parameter, because translating
        such an operation can need its value. On both backends a group
        without parameters is prepared point by point. The default returns
        None, and each point is prepared as its own bound circuit and run
        as its own attempt. A Method whose points use different shot counts
        keeps this default on IBM Runtime.

        Preparation raises `ValueError`, before it prepares any point of
        the group, when a point's readout differs from the group's,
        including a group that an earlier preparation in the same Run
        compiled. When the groups of one IBM Runtime job have different
        shots, the Run raises `ValueError` after preparing their points and
        before sending the job.

        Args:
            plan (Plan): The Plan.
            points (tuple[StaticPoint, ...]): The points to prepare
                (`nwqlib.core.planning.StaticPoint`), each with its key, its
                Experiment and the bindings of its range axes.

        Returns:
            templates (tuple[RowTemplate, ...] | None): The groups, or `None`.
        """
        return None

    def query_catalog_reason(self):
        """Return why the experiments of this Method's Plans are templates of queries rather than the workload of a Run, or None.

        A Method overrides it when one experiment of its Plan can serve any
        number of queries whose parameter values its iteration chooses from
        earlier outcomes, so that the counts of the Plan's construction are
        not the totals of a Run. The reason says why and how to estimate one
        query. By returning a reason, the Method asserts that every query it
        runs is a point of one experiment of the Plan, bound by
        `Plan.resolve`, and that the construction of that point keeps the
        experiment's measurement batch, observation kind and repetitions.
        The resource estimate of a whole Plan, from `nwqlib.estimate(plan)`
        with or without a profile, `compare` or `scan`, then passes the
        reason to [`nwqlib.resources.estimate`][nwqlib.resources.fold.estimate]
        as `catalog`. That estimate reports a count as zero only when no
        query can contribute to it. It leaves the other counts unavailable
        with the reason, except `root_setting_declarations` and
        `root_repetitions`, the setting declarations and body repetitions of
        a `MeasurementBatch` at the root of the Program, which keep their
        values. It labels each width
        and byte peak as a footprint of the templates. The forecast of one
        point with every parameter value
        bound estimates that point's construction without the reason. The
        default returns `None`, and the resource estimate of a Plan adds up
        its construction as the workload of a Run.

        Returns:
            reason (str | None): The reason, or `None`.
        """
        return None

    def execute(self, plan, *, run):
        """Advance the Run's work and return the Result when it is complete, or None while it is pending.

        The default runs the Plan's experiments in order, one attempt each, never
        submits an existing attempt again, and analyzes once every experiment has
        its data. A Method that chooses later circuits from earlier results
        overrides it.

        Args:
            plan (Plan): The Plan.
            run (Run): The Run.

        Returns:
            result (Result | None): The Result, or `None` while work is pending.
        """
        from nwqlib._prepared_execution import execute_static
        return execute_static(plan, run=run)

    def analyze(self, plan, data, *, settings):
        """Turn the collected data of a Plan into its Result, with explicit settings and without collecting new data.

        `Result.analyze` and the end of a Run call it. Return an instance of the
        Method's Result class computed from this Plan and the exact data used.
        The caller attaches it with `Result._attach`, which runs the supported
        Plan and data checks, including `validate_plan`, so `analyze` need not
        call `validate_plan` itself. A Result that `analyze` has already
        attached to the same Plan and data is accepted as well.

        Args:
            plan (Plan): The Plan.
            data (RunData): The collected data.
            settings (dict): Analysis settings.

        Returns:
            result (Result): The Result for `plan` and `data`, attached or not
                yet attached.
        """
        raise NotImplementedError("a Method must interpret its acquired observations")

    def recover_analysis(self, plan, *, run):
        """Finish an analysis that a saved Run interrupted, without submitting earlier work again.

        Only a Method that saves an interrupted analysis overrides it. The default
        raises.

        Args:
            plan (Plan): The Plan.
            run (Run): The reopened Run.

        Raises:
            ValueError: Always, in the default.
        """
        raise ValueError("this Method has no interrupted controller analysis to recover")

    def before_submit(self, plan, prepared, *, run):
        """Check the prepared circuits or classical routines (host kernels in the code) of one submission before the Run records it, and raise to refuse it.

        The Run calls it with the tuple of `PreparedHandle` objects that the
        submission sends, before it records the submission, so a refusal
        leaves no attempt behind. The default accepts every submission.

        Args:
            plan (Plan): The Plan.
            prepared (tuple[PreparedHandle, ...]): The prepared work of the
                submission.
            run (Run): The Run that submits them.
        """

    def error_model(self, plan):
        """Return the known and unavailable error sources of a Plan, each with the quantity and unit it refers to.

        The default returns the error model recorded in the Plan (`plan.error_model`).

        Args:
            plan (Plan): The Plan.

        Returns:
            model (ErrorModel | None): The error model, or `None`.
        """
        return plan.error_model

    def validate_point(self, plan, experiment, values):
        """Check the parameter values of one experiment point against the Method's own rules, and raise to refuse the point.

        It runs when `Plan.resolve` or a preparation resolves a point, after
        each value has passed the checks of its Program parameter and, in a
        measurement batch, of its range axis. The default accepts every
        point.

        Args:
            plan (Plan): The Plan.
            experiment (Experiment): The experiment of the point.
            values (Mapping[str, object]): Read-only mapping from each bound
                Program parameter name to its value.
        """

    def specialize_experiment(self, plan, experiment, values):
        """Return the experiment of one point with its readout fitted to the point's parameter values.

        It is called after `validate_point`, with the same values. A Method
        whose readout depends on those values, such as the labels a point
        measures, returns a revised `Experiment` with the same name, setting,
        batch and setting index. The default returns the experiment
        unchanged.

        Args:
            plan (Plan): The Plan.
            experiment (Experiment): The planned experiment of the point.
            values (Mapping[str, object]): Read-only mapping from each bound
                Program parameter name to its value.

        Returns:
            experiment (Experiment): The experiment to run at this point.
        """
        return experiment

    def selected_kernels(self, plan, experiment, program):
        """Return the declarations of the classical routines that one point runs on this computer.

        It is called when the construction of a point is built. A Method may
        revise a declaration of the Plan's `construction.kernels` for the
        point, for example its scalar labels and its work per invocation,
        but each returned declaration must be a revision of the Plan's
        declaration of the same name, with the same implementation, inputs
        and dependencies, so that a point cannot substitute another
        routine or another input. The default returns the Plan's
        declarations unchanged.

        Args:
            plan (Plan): The Plan.
            experiment (Experiment): The experiment of the point.
            program (Program): The point's Program, with every parameter
                bound.

        Returns:
            kernels (tuple[SelectedKernel, ...]): The declarations of the
                point.
        """
        return plan.construction.kernels

    def verify(self, plan, result, *, checks):
        """Run checks the caller selected explicitly on a Result, at their own cost.

        `Result.verify(checks=...)` calls it. Override it only for checks the Method
        supports. The default refuses.

        Args:
            plan (Plan): The Plan.
            result (Result): The Result to check.
            checks (object): The selected checks, as `Result.verify` received them.

        Raises:
            ValueError: Always, in the default.
        """
        raise ValueError("this method has no selected verification operation")


class ApplicabilityError(ValueError):
    """Raised when a configured Method cannot solve the given problem or output.

    A Method's `plan` raises it, for example for an unsupported problem type,
    input representation or output. It is a `ValueError`. A refusal because a
    memory or work limit stops the planning is `ResourceLimitError` instead.
    """
