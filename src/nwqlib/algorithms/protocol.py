"""Immutable scientific Method configuration and direct execution behavior."""

import inspect
from typing import ClassVar

from nwqlib.core.records import Record, Source, Text


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
    - `error_model`: the known and unavailable error sources of a Plan.
    - `verify` and `recover_analysis`: explicit extra operations, unavailable by
      default.
    - `reduction_allowance`, for a Plan that reduces data while it is collected:
      check the reduction's workspace and return the Method's remaining work
      allowance.
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
            shots (int | None): Requested shots, or `None` for exact readout or the
                Method's default.
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
        overrides it to prepare that circuit. The hook receives `"all"` only after
        `prepare_all_refusal` returned `None`.

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
        default returns `None` when `prepare` accepts the `settings` keyword, and
        otherwise a refusal, because a `prepare` override without that keyword never
        receives `"all"`.

        Returns:
            reason (str | None): The reason, or `None`.
        """
        parameters = inspect.signature(self.prepare).parameters.values()
        if any(item.name == "settings" or item.kind is item.VAR_KEYWORD for item in parameters):
            return None
        return (
            f"{type(self).__name__}.prepare does not accept the settings keyword, so it can prepare "
            "only the setting it submits first. prepare(plan) with the default settings='first' prepares "
            "that setting. An override that accepts settings and passes it to Method.prepare also "
            "serves settings='all'"
        )

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
        Method's Result class attached to this Plan and the exact data used.
        `Result._attach` runs the supported Plan and data checks.

        Args:
            plan (Plan): The Plan.
            data (RunData): The collected data.
            settings (dict): Analysis settings.

        Returns:
            result (Result): The Result attached to `plan` and `data`.
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
        """Admit a tuple of original acquisitions before their submission intent."""

    def error_model(self, plan):
        """Return the known and unavailable error sources of a Plan, each with the quantity and unit it refers to.

        The default returns the error model stored in the Plan (`plan.error_model`).

        Args:
            plan (Plan): The Plan.

        Returns:
            model (ErrorModel | None): The error model, or `None`.
        """
        return plan.error_model

    def validate_point(self, plan, experiment, values):
        """Admit method-specific argument relations after generic IR admission."""

    def specialize_experiment(self, plan, experiment, values):
        """Resolve an adaptive readout from its actual admitted arguments."""
        return experiment

    def selected_kernels(self, plan, experiment, program):
        """Resolve selected host declarations without replacing native inputs."""
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
    input representation or output. It is a `ValueError`.
    """
